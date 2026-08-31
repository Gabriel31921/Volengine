"""The ``Calibrator`` port as an executable contract, run against every implementer this build has.

Two calibrators compete behind one output contract, and in F3 there will be three. What makes them
comparable is not that they fit the same way -- they emphatically do not -- but that whatever they
produce leaves through ``CalibratedSurface`` and lands in a risk report that never learns which one
was on the other end. That claim is architecture, and architecture that is only asserted in prose
decays: the day a new calibrator returns its slices in its own order, or publishes a grid on axes
of its own choosing, nothing in its own test file would notice.

So the tests below are written **against the port**, parametrised over the implementers, and
mention no optimiser anywhere. Nothing here asserts accuracy: how close a fit lands is a claim
about one calibrator and lives in that calibrator's own tests, where the tolerance can be argued
against the method. What is asserted is what a *consumer* is entitled to assume, and the last two
tests carry it all the way through Risk -- the executable equivalent of the context map (Design
8.2).

**Adding F3's calibrators is meant to be one line**: an entry in :data:`CALIBRATORS`. If a new
implementer needs a test body changed to pass, the port has grown a second meaning and that is the
finding, not the failure.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from tests.parametric_pricing.builders import (
    FORWARD,
    NEAR,
    make_grid_spec,
    make_market_snapshot,
    make_weighting,
)
from tests.risk.builders import make_bumps
from volengine.contracts.calibrated_surface import CalibratedSurface, SurfaceStatus
from volengine.parametric_pricing.adapters.flat_vol import FlatVolCalibrator
from volengine.parametric_pricing.adapters.scipy_calibrator import ScipyCalibrator
from volengine.parametric_pricing.application.acl import to_calibrated_surface, to_calibration_task
from volengine.parametric_pricing.domain.calibration import CalibrationTask
from volengine.parametric_pricing.domain.ports import Calibrator
from volengine.risk.application.acl import to_surface_view
from volengine.risk.domain.portfolio import Position
from volengine.risk.domain.pricing import OptionKindR
from volengine.risk.domain.valuation import position_risk

pytestmark = pytest.mark.contract

CALIBRATORS: tuple[Callable[[], Calibrator], ...] = (ScipyCalibrator, FlatVolCalibrator)
"""Every implementer of the port this build can construct, as zero-argument factories.

Two today, three when F3-A lands its JAX calibrator and F3-D its learner. A factory rather than an
instance so that each test gets a calibrator that has never been called: warm-start state is the
one thing a calibrator is allowed to keep between cycles, and a shared instance would let one
test's history decide another test's answer.
"""

IDS = tuple(factory.__name__ for factory in CALIBRATORS)
"""Test ids that name the class, so a failure says which implementer broke the contract."""


def task() -> CalibrationTask:
    """One published snapshot, translated into the problem every implementer is handed.

    Built through the real ACL rather than assembled by hand: the contract is about what a
    calibrator receives *in the engine*, and a task written directly in the test could be one no
    inversion would ever produce.
    """
    translated = to_calibration_task(make_market_snapshot(), make_weighting())
    assert translated is not None, "the builder's snapshot must produce a fittable task"
    return translated


def published(calibrator: Calibrator) -> CalibratedSurface:
    """Fit the task and publish the whole result, exactly as the use case would.

    Every slice is accepted here. Acceptance is the use case's judgement and it is tested there;
    what this module needs is the translation, which must work for whatever the fit came back
    with.
    """
    fitted = task()
    result = calibrator.calibrate(None, fitted)
    surface = to_calibrated_surface(
        task=fitted,
        accepted=result.slices,
        n_iterations=result.n_iterations,
        duration_ms=result.duration_ms,
        grid=make_grid_spec(),
        producer_id=calibrator.producer_id,
        surface_id=f"{fitted.snapshot_id}/{calibrator.producer_id}",
        ts_calibrated=fitted.ts_snapshot,
        status=SurfaceStatus.OK,
    )
    assert surface is not None, "an accepted fit must be expressible on the published grid"
    return surface


@pytest.mark.parametrize("build", CALIBRATORS, ids=IDS)
def test_the_producer_names_itself_and_keeps_the_name(build: Callable[[], Calibrator]) -> None:
    """``producer_id`` is what every metric, topic and report is tagged with.

    A blank or shifting name would route two producers' surfaces onto one topic, which is the one
    mistake the comparison this project exists for could not survive.
    """
    calibrator = build()

    assert calibrator.producer_id.strip()
    assert calibrator.producer_id == build().producer_id


@pytest.mark.parametrize("build", CALIBRATORS, ids=IDS)
def test_every_slice_of_the_task_comes_back_once_in_the_task_order(
    build: Callable[[], Calibrator],
) -> None:
    """One result per slice, in the order they were handed over, which is ascending in tenor.

    ``CalibrationResult`` refuses to hold slices out of order, so an implementer that fitted the
    expiries concurrently and collected them as they finished would fail here rather than publish
    a surface whose term structure runs backwards.
    """
    fitted = task()

    result = build().calibrate(None, fitted)

    assert [one.expiry for one in result.slices] == [one.expiry for one in fitted.slices]
    assert [one.tenor_years for one in result.slices] == [one.tenor_years for one in fitted.slices]


@pytest.mark.parametrize("build", CALIBRATORS, ids=IDS)
def test_the_same_task_twice_gives_the_same_answer(build: Callable[[], Calibrator]) -> None:
    """Determinism, which ADR-004's replay rests on and no random restart may break.

    Two runs over one recording have to produce the same report or the recording proves nothing.
    A multi-start optimiser is welcome; a multi-start optimiser seeded from the clock is not.
    """
    fitted = task()

    first = build().calibrate(None, fitted)
    second = build().calibrate(None, fitted)

    assert [one.params for one in first.slices] == [one.params for one in second.slices]


@pytest.mark.parametrize("build", CALIBRATORS, ids=IDS)
def test_a_warm_start_is_accepted_and_still_answers_for_every_slice(
    build: Callable[[], Calibrator],
) -> None:
    """The previous cycle's parameters are an offer, not an instruction.

    A calibrator may seed its search from them, and one that has no search to seed may ignore
    them entirely -- but neither may fail on them, because the use case chains the state through
    on every cycle after the first and does not know which kind it holds.
    """
    fitted = task()
    calibrator = build()
    previous = {one.expiry: one.params for one in calibrator.calibrate(None, fitted).slices}

    warmed = calibrator.calibrate(previous, fitted)

    assert len(warmed.slices) == len(fitted.slices)


@pytest.mark.parametrize("build", CALIBRATORS, ids=IDS)
def test_what_it_fitted_publishes_as_a_surface_a_risk_report_can_value(
    build: Callable[[], Calibrator],
) -> None:
    """The whole point of the port: a position is valued without knowing who fitted the surface.

    Snapshot to task to fit to ``CalibratedSurface`` to ``SurfaceView`` to one report line, with
    no branch anywhere on the producer. The bounds are deliberately loose -- this asserts that a
    number came out and that it is a volatility rather than a NaN, which is a statement about the
    wiring; how *good* the number is belongs to the implementer's own tests.
    """
    surface = published(build())

    line = position_risk(
        view=to_surface_view(surface),
        position=Position(
            underlying="BTC",
            expiry=NEAR,
            strike=FORWARD,
            kind=OptionKindR.CALL,
            quantity=1.0,
        ),
        bumps=make_bumps(),
    )

    assert math.isfinite(line.vol) and 0.0 < line.vol < 5.0
    assert math.isfinite(line.value) and line.value > 0.0


def test_every_implementer_publishes_the_same_axes_for_the_same_snapshot() -> None:
    """Interchangeable means valued on one coordinate system, not merely valued.

    Risk interpolates on the published grid, so two producers whose surfaces sat on different
    moneyness nodes would be two books rather than two opinions about one -- and the comparison
    this project exists to make would be comparing the interpolation as much as the fit.

    Not parametrised: the claim is about the implementers *against each other*, which is exactly
    the assertion no single-implementer run can make.
    """
    surfaces = [published(build()) for build in CALIBRATORS]

    axes = {(one.grid.log_moneyness, one.grid.tenors, one.grid.expiries) for one in surfaces}
    assert len(axes) == 1
