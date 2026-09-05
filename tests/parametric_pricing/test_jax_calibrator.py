"""What the JAX calibrator recovers, what it never recompiles, and what the mask cannot touch.

Three groups of claims, and only the first is about mathematics.

* **Known truth.** Quotes are generated from SVI slices the calibrator's own cold start is nowhere
  near, and the assertions are on the recovered *parameters* rather than only on the RMSE -- the
  trap ``Plan.md`` names for a generator and a fitter that are both ours: a shared mistake cancels
  and a known-truth test passes with the system broken.
* **No recompilation**, which is risk 5 of Design 11 and the permanent guard ADR-009 asks for. The
  failure is silent: every number stays correct and the engine simply gets slow enough to defeat
  the reason for using JAX at all. It is watched here through JAX's own compilation events, with a
  companion test that makes the counter rise so the guard cannot pass vacuously.
* **The mask is inert.** Perturbing what sits under the padding changes nothing, asserted on the
  objective and on both searches.

Several tests come in pairs: one asserts that a mechanism protects the fit, and its neighbour
asserts that the thing it protects against really would have wrecked it.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.parametric_pricing.builders import (
    FAR,
    FAR_TENOR,
    FORWARD,
    NEAR,
    NEAR_TENOR,
    make_calibration_task,
    make_slice_task,
)
from tests.parametric_pricing.jax_builders import TEST_SETTINGS, TEST_SHAPE, make_jax_calibrator
from volengine.parametric_pricing.adapters.jax_black76 import DTYPE
from volengine.parametric_pricing.adapters.jax_calibrator import (
    PRODUCER_ID,
    JaxCalibrator,
    JaxFitSettings,
    _adam_fit,
    _at_bound,
    _compiled,
    _durrleman_g,
    _lbfgs_fit,
    _objective,
    _to_batch,
)
from volengine.parametric_pricing.adapters.padding import PadShape, pad
from volengine.parametric_pricing.domain.calibration import CalibrationTask, SliceTask
from volengine.parametric_pricing.domain.durrleman import durrleman_g
from volengine.parametric_pricing.domain.errors import CalibrationError
from volengine.parametric_pricing.domain.ports import Calibrator
from volengine.parametric_pricing.domain.svi_slice import SVIParams

TRUTH = SVIParams(a=0.0035, b=0.060, rho=-0.35, m=-0.02, sigma=0.10)
"""The generating slice: a one-month crypto smile at roughly 20% at-the-money vol, with the
downside wing lifted. The same one the scipy baseline is measured on, so the two stages' numbers
are readable against each other, and deliberately nowhere near the cold start's guess."""

ARBITRAGEABLE = SVIParams(a=0.001, b=0.35, rho=-0.90, m=0.0, sigma=0.02)
"""A slice that really does price a negative density: steep, sharply skewed, nearly kinked."""

K_AXIS: tuple[float, ...] = tuple(np.linspace(-0.6, 0.6, 21))
"""Twenty-one strikes spanning +-60% in log-forward-moneyness: a well-populated crypto slice, and
enough points that five parameters are comfortably over-determined."""

VOL_NOISE = 0.002
"""Twenty basis points of volatility noise, the order of a real bid-ask on a liquid crypto wing."""

COMPILE_EVENT = "/jax/compilation_cache/compile_requests_use_cache"
"""JAX's own event for a compilation request. It fires when a specialisation is *built* and stays
silent on a cache hit, which is exactly the counter ADR-009 asks to be watched."""

_events: list[str] = []
"""Every monitoring event JAX has emitted since this module was imported.

A process-wide listener registered once, and read through deltas rather than reset: JAX offers no
way to unregister one, and a counter that could be zeroed would be a counter another test could
zero halfway through this one.
"""


def _record(event: str, **_: str | int) -> None:
    """JAX's monitoring callback. Its signature is JAX's, not ours."""
    _events.append(event)


jax.monitoring.register_event_listener(_record)


def compilations_since(mark: int) -> int:
    """How many compilations JAX has performed since ``mark``, process-wide."""
    return sum(1 for name in _events[mark:] if name == COMPILE_EVENT)


def vols_of(params: SVIParams, k: tuple[float, ...], tenor_years: float) -> tuple[float, ...]:
    """The volatilities a slice implies at each point of an axis: the answer a fit must recover."""
    return tuple(params.implied_vol(one, tenor_years) for one in k)


def noisy_vols(
    params: SVIParams, k: tuple[float, ...], tenor_years: float, seed: int
) -> tuple[float, ...]:
    """The same, perturbed by a seeded normal draw. Seeded so a failure is reproducible rather
    than a story about one unlucky afternoon."""
    rng = np.random.default_rng(seed)
    return tuple(vol + float(rng.normal(0.0, VOL_NOISE)) for vol in vols_of(params, k, tenor_years))


def svi_task(
    params: SVIParams = TRUTH,
    k: tuple[float, ...] = K_AXIS,
    tenor_years: float = NEAR_TENOR,
    expiry: datetime = NEAR,
    weights: tuple[float, ...] | None = None,
    implied_vol: tuple[float, ...] | None = None,
) -> SliceTask:
    """One slice quoted exactly on a known SVI curve, evenly weighted unless a test says so."""
    vols = vols_of(params, k, tenor_years) if implied_vol is None else implied_vol
    return make_slice_task(
        expiry=expiry,
        tenor_years=tenor_years,
        forward=FORWARD,
        log_moneyness=k,
        implied_vol=vols,
        weights=weights if weights is not None else tuple(1.0 / len(k) for _ in k),
    )


def one_slice(task: SliceTask) -> CalibrationTask:
    """The smallest whole task: one expiry, so a test reads one result."""
    return make_calibration_task(slices=(task,))


def fit(task: SliceTask, calibrator: Calibrator | None = None) -> SVIParams:
    """Cold-fit one slice and hand back the parameters, which is what most tests assert on."""
    engine = make_jax_calibrator() if calibrator is None else calibrator
    return engine.calibrate(None, one_slice(task)).slices[0].params


# --- known truth


def test_recovers_the_generating_parameters_from_clean_quotes() -> None:
    """The cold cycle, on a slice it has no history of. Tolerances are single precision's, which
    is what this adapter runs in -- three orders of magnitude tighter than any fit uncertainty."""
    fitted = fit(svi_task())

    assert fitted.a == pytest.approx(TRUTH.a, abs=1e-4)
    assert fitted.b == pytest.approx(TRUTH.b, abs=1e-3)
    assert fitted.rho == pytest.approx(TRUTH.rho, abs=1e-2)
    assert fitted.m == pytest.approx(TRUTH.m, abs=1e-2)
    assert fitted.sigma == pytest.approx(TRUTH.sigma, abs=1e-2)


def test_fits_clean_quotes_to_within_a_basis_point() -> None:
    """A tenth of a bid-ask is the bar a fit has to clear to be worth publishing."""
    result = make_jax_calibrator().calibrate(None, one_slice(svi_task())).slices[0]

    # An explicit `abs` because `pytest.approx` passes on `rel` *or* `abs` and its default `abs`
    # is 1e-12: below that scale a relative bound never binds and the assertion accepts anything.
    assert result.rmse_vol_bp == pytest.approx(0.0, abs=1.0)
    assert result.max_err_vol_bp == pytest.approx(0.0, abs=2.0)
    assert result.converged
    assert not result.at_bound


def test_recovers_the_generating_parameters_under_realistic_noise() -> None:
    """Loose bounds on purpose: twenty basis points of noise on twenty-one quotes cannot pin five
    parameters to more than a few percent, and a test that demanded more would be asserting the
    seed rather than the mathematics."""
    fitted = fit(svi_task(implied_vol=noisy_vols(TRUTH, K_AXIS, NEAR_TENOR, seed=7)))

    assert fitted.a == pytest.approx(TRUTH.a, abs=5e-4)
    assert fitted.b == pytest.approx(TRUTH.b, rel=0.10)
    assert fitted.rho == pytest.approx(TRUTH.rho, rel=0.10)
    assert fitted.m == pytest.approx(TRUTH.m, abs=0.02)
    assert fitted.sigma == pytest.approx(TRUTH.sigma, rel=0.15)


def test_fits_every_expiry_of_a_surface_in_one_batched_call() -> None:
    """``vmap`` over the padded slice axis: sixteen expiries cost what one costs, and each is still
    an independent fit (ADR-008). A batch that had coupled them would show up as a near expiry
    dragged towards a far one's parameters."""
    task = make_calibration_task(
        slices=(
            svi_task(expiry=NEAR, tenor_years=NEAR_TENOR),
            svi_task(expiry=FAR, tenor_years=FAR_TENOR),
        )
    )

    result = make_jax_calibrator().calibrate(None, task)

    assert [one.expiry for one in result.slices] == [NEAR, FAR]
    for one in result.slices:
        assert one.params.b == pytest.approx(TRUTH.b, abs=1e-3)
        assert one.rmse_vol_bp < 1.0


# --- no recompilation (ADR-009, risk 5 of Design 11)


@pytest.mark.no_recompilation
def test_a_changing_chain_composition_never_recompiles() -> None:
    """**The permanent guard.** Strikes are born and die and whole expiries roll off between
    snapshots; every one of those chains is fitted by the specialisation the constructor built.

    The counter is JAX's own compilation event rather than a wall clock, because the failure this
    watches is invisible in the answers: a recompiled fit is correct and hundreds of milliseconds
    late, which in streaming is the same as broken.
    """
    calibrator = make_jax_calibrator()
    compositions = (
        (K_AXIS[:4],),
        (K_AXIS[:11],),
        (K_AXIS, K_AXIS[:7]),
        (K_AXIS[:9], K_AXIS[:9], K_AXIS[:5]),
    )

    mark = len(_events)
    for chain in compositions:
        task = make_calibration_task(
            slices=tuple(
                svi_task(
                    k=axis,
                    expiry=datetime(2026, 9, 1 + index, 8, tzinfo=UTC),
                    tenor_years=NEAR_TENOR * (index + 1),
                )
                for index, axis in enumerate(chain)
            )
        )
        result = calibrator.calibrate(None, task)
        calibrator.calibrate({one.expiry: one.params for one in result.slices}, task)

    assert compilations_since(mark) == 0


@pytest.mark.no_recompilation
def test_the_compilation_counter_really_does_rise_for_a_new_shape() -> None:
    """The guard on the guard. Without it, the test above would pass just as happily against a
    counter that never moves -- a listener on the wrong event, or an event JAX has renamed."""
    mark = len(_events)

    JaxCalibrator(settings=TEST_SETTINGS, shape=PadShape(max_slices=2, max_quotes=6, mesh_nodes=7))

    assert compilations_since(mark) > 0


def test_the_constructor_pays_for_the_compilation_and_the_first_fit_does_not() -> None:
    """ADR-009's timing claim, stated as a test: compiled once at start-up, executed forever. A
    calibrator that compiled lazily would put hundreds of milliseconds inside the first snapshot's
    latency budget, which is the one place they cannot go."""
    calibrator = make_jax_calibrator()

    mark = len(_events)
    calibrator.calibrate(None, one_slice(svi_task()))

    assert compilations_since(mark) == 0


# --- the mask is inert


def batch_with_fill(task: CalibrationTask, fill: float) -> object:
    """The padded task with every masked cell overwritten by ``fill``.

    Reaching past ``calibrate`` on purpose: the padding is filled with benign values by
    construction, so the only way to assert that the *loss* ignores it is to put something else
    there. What a caller could do accidentally -- a chain that shrinks between snapshots leaving
    yesterday's numbers in the buffer -- is what this simulates deliberately.
    """
    padded = pad(task, TEST_SHAPE, TEST_SETTINGS.durrleman_mesh_margin)
    quotes = ~padded.quote_mask
    log_moneyness = padded.log_moneyness.copy()
    implied_vol = padded.implied_vol.copy()
    log_moneyness[quotes] = fill
    implied_vol[quotes] = abs(fill) + 1.0
    return _to_batch(
        type(padded)(
            log_moneyness=log_moneyness,
            implied_vol=implied_vol,
            weights=padded.weights,
            quote_mask=padded.quote_mask,
            tenor_years=padded.tenor_years,
            slice_mask=padded.slice_mask,
            mesh=padded.mesh,
            n_slices=padded.n_slices,
        )
    )


@given(fill=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False))
@settings(max_examples=25, deadline=None)
def test_perturbing_the_values_under_the_mask_does_not_change_the_objective(fill: float) -> None:
    """The property ADR-009 asks for, at the loss. ``where(mask, err, 0)`` is only half of what
    makes a padded cell inert; the other half is that its weight is zero, and both are asserted
    here at once by moving the cell anywhere in a thousand units of log-moneyness."""
    task = one_slice(svi_task(k=K_AXIS[:5]))
    start = jnp.zeros(5)
    pinned = jnp.asarray([False] * 5)

    plain = jax.tree.map(lambda a: a[0], _to_batch(pad(task, TEST_SHAPE, 0.5)))
    poisoned = jax.tree.map(lambda a: a[0], batch_with_fill(task, fill))

    assert float(_objective(start, start, pinned, plain, TEST_SETTINGS)) == float(
        _objective(start, start, pinned, poisoned, TEST_SETTINGS)
    )


def test_perturbing_the_values_under_the_mask_does_not_change_either_search() -> None:
    """The same property one level up, where it actually matters: both cycles have to land on the
    same point, not merely see the same cost once."""
    task = one_slice(svi_task(k=K_AXIS[:5]))
    start = jnp.asarray([-3.0, -2.0, -0.3, 0.0, -2.2])

    plain = jax.tree.map(lambda a: a[0], _to_batch(pad(task, TEST_SHAPE, 0.5)))
    poisoned = jax.tree.map(lambda a: a[0], batch_with_fill(task, 250.0))

    for search in (_adam_fit, _lbfgs_fit):
        one = search(start, plain, TEST_SETTINGS)
        other = search(start, poisoned, TEST_SETTINGS)
        assert np.array_equal(np.asarray(one.x), np.asarray(other.x))
        assert float(one.cost) == float(other.cost)


def test_a_perturbation_of_a_visible_quote_really_would_have_moved_the_fit() -> None:
    """The guard on the pair above: an inertness test proves nothing unless the same perturbation,
    applied where the mask does *not* hide it, changes the answer."""
    clean = svi_task(k=K_AXIS[:5])
    poisoned_vols = (clean.implied_vol[0] + 0.05, *clean.implied_vol[1:])

    moved = fit(svi_task(k=K_AXIS[:5], implied_vol=poisoned_vols))

    assert abs(moved.rho - fit(clean).rho) > 1e-3


# --- the warm start


def test_a_warm_start_on_an_unchanged_market_is_never_degraded() -> None:
    """**The regression this adapter needs most.** Adam's step is set by its own moment ratio and
    not by the size of the gradient, so at an optimum -- where a warm start begins in a calm market
    -- it keeps stepping about a learning rate away and settles wherever the tolerance catches it.
    A hot cycle that returned the last iterate would hand back a *worse* fit than it was given,
    every snapshot, and the surface would drift with no market behind the movement.
    """
    calibrator = make_jax_calibrator()
    task = one_slice(svi_task())
    cold = calibrator.calibrate(None, task).slices[0]

    warm = calibrator.calibrate({cold.expiry: cold.params}, task).slices[0]

    assert warm.rmse_vol_bp <= cold.rmse_vol_bp + 1e-6


def test_a_warm_start_costs_less_than_a_cold_one() -> None:
    """The whole argument for a hot cycle: a calm market is nearly free. Measured in objective
    evaluations rather than in wall time, because a clock in a test is a flake generator."""
    calibrator = make_jax_calibrator()
    task = one_slice(svi_task())
    cold = calibrator.calibrate(None, task)

    warm = calibrator.calibrate({cold.slices[0].expiry: cold.slices[0].params}, task)

    assert 0 < warm.n_iterations < cold.n_iterations


def test_a_warm_start_from_a_stale_basin_is_retried_cold() -> None:
    """Design 5.6's "cold cycle after a failure". The previous cycle's parameters can sit in a
    basin the market has since left; from there the search runs out of budget, says so, and the
    multi-start is what recovers the fit rather than publishing the wreck."""
    stale = SVIParams(a=0.5, b=2.0, rho=0.85, m=0.9, sigma=1.5)
    task = one_slice(svi_task())

    result = make_jax_calibrator().calibrate({NEAR: stale}, task)

    assert result.slices[0].rmse_vol_bp < 5.0
    assert result.slices[0].params.b == pytest.approx(TRUTH.b, abs=1e-2)


def test_an_expiry_with_no_history_is_cold_started_beside_one_that_has_it() -> None:
    """A tenor born since the last cycle (ADR-013) is an ordinary event, not an error, and the row
    it occupies still has to be filled with something -- which is what the per-row warm flag is
    for."""
    calibrator = make_jax_calibrator()
    task = make_calibration_task(
        slices=(
            svi_task(expiry=NEAR, tenor_years=NEAR_TENOR),
            svi_task(expiry=FAR, tenor_years=FAR_TENOR),
        )
    )
    first = calibrator.calibrate(None, task)

    result = calibrator.calibrate({NEAR: first.slices[0].params}, task)

    assert len(result.slices) == 2
    assert all(one.rmse_vol_bp < 1.0 for one in result.slices)


# --- reporting a failed fit


STARVED = JaxFitSettings(hot_steps=1, cold_steps=1)
"""A budget of one step in either cycle: enough to produce a fit, nowhere near enough to converge.

The cheapest way to reach the failure paths deliberately. A calibrator cannot be *asked* to fail on
real data without inventing a market nobody quotes, and starving the budget fails it for the one
reason ``converged`` exists to name.
"""

AT_THE_EDGE = SVIParams(a=0.0035, b=0.060, rho=-0.35, m=-2.0, sigma=0.10)
"""``TRUTH`` with its smile minimum pushed out to the practical bound on ``m``.

A slice a market could quote -- the vols are ordinary and monotone across the band -- and one whose
parameters a fit is not allowed to publish, which is what makes it the case ``at_bound`` is for.
"""


def test_a_starved_budget_returns_an_unconverged_fit_rather_than_raising() -> None:
    """**ADR-006's precondition.** A poor fit is a return value, never an exception: the use case
    is the layer that decides publication, and it republishes the last good surface as
    ``STALE_REPUBLISH`` on the strength of exactly these flags. A calibrator that raised, or that
    quietly reported success, would take that decision away from the layer that owns it and leave
    nothing to report."""
    result = make_jax_calibrator(settings=STARVED).calibrate(None, one_slice(svi_task()))

    fitted = result.slices[0]
    assert fitted.converged is False
    assert fitted.rmse_vol_bp > 1.0
    assert fitted.n_quotes_used == len(K_AXIS)
    assert fitted.params.min_total_variance >= 0


def test_a_fit_that_ends_on_a_practical_bound_says_so() -> None:
    """The other half of ADR-006's rule, observed where it matters -- coming out of ``calibrate``
    rather than out of the private predicate. The warm start sits exactly on the bound and the
    budget is too short to leave it, so the attempt has an excellent residual and is still
    unpublishable: a fit whose wing is held up by the constraint rather than by the market.

    It also pins ``_prefer``'s tie-break. Both attempts are unhealthy, the warm one has by far the
    lower residual, and it is the one that comes back -- with ``at_bound`` intact.
    """
    task = one_slice(
        svi_task(params=AT_THE_EDGE, implied_vol=vols_of(AT_THE_EDGE, K_AXIS, NEAR_TENOR))
    )

    fitted = make_jax_calibrator(settings=STARVED).calibrate({NEAR: AT_THE_EDGE}, task).slices[0]

    assert fitted.at_bound is True
    assert fitted.converged is False
    assert fitted.rmse_vol_bp < 1.0


def test_when_both_attempts_fail_the_lower_residual_is_the_one_reported() -> None:
    """``_prefer``'s other branch: two unhealthy attempts, and the cold retry wins because the
    stale warm start never got near the market.

    The guard against reading this as a coincidence is the last assertion: the parameters the warm
    start began from are five thousand vol points away from the quotes, and one step cannot have
    moved them to the answer that came back.
    """
    stale = SVIParams(a=0.5, b=2.0, rho=0.85, m=0.9, sigma=1.5)
    calibrator = make_jax_calibrator(settings=STARVED)
    task = one_slice(svi_task())

    cold = calibrator.calibrate(None, task).slices[0]
    warm = calibrator.calibrate({NEAR: stale}, task).slices[0]

    assert warm.converged is False
    assert warm.rmse_vol_bp <= cold.rmse_vol_bp
    assert warm.params.b == pytest.approx(cold.params.b, abs=1e-6)
    assert abs(warm.params.b - stale.b) > 1.0


# --- weights, pinning and bounds


POISONED_VOLS: tuple[float, ...] = (
    TRUTH.implied_vol(K_AXIS[0], NEAR_TENOR) + 0.30,
    *vols_of(TRUTH, K_AXIS, NEAR_TENOR)[1:],
)
"""The same clean slice with its deepest downside quote thirty vol points too high: a stale or
crossed wing, which is what a real chain produces several times an hour."""


def test_a_zero_weight_quote_does_not_steer_the_fit() -> None:
    """Ingestion flags, the calibrator weights or excludes. A flagged quote stays in the slice, is
    visible in the residuals, and moves nothing."""
    weights = (0.0, *(1.0 / 20 for _ in K_AXIS[1:]))

    fitted = fit(svi_task(implied_vol=POISONED_VOLS, weights=weights))

    assert fitted.b == pytest.approx(TRUTH.b, abs=1e-3)
    assert fitted.rho == pytest.approx(TRUTH.rho, abs=1e-3)


def test_that_same_quote_at_full_weight_really_does_move_the_fit() -> None:
    """The guard: without it, the test above passes on a calibrator that ignores its wings.

    The movement is small -- a hundredth of a unit of skew rather than a ruined slice -- and that
    is Huber doing its own job beside the weights: thirty vol points is three thousand basis
    points, far past the scale where the loss stops charging quadratically, so one absurd quote
    buys a bounded amount of movement instead of the whole smile. Both mechanisms are needed and
    this is where they can be told apart.
    """
    weighted = fit(svi_task(implied_vol=POISONED_VOLS))
    ignored = fit(
        svi_task(
            implied_vol=POISONED_VOLS,
            weights=(0.0, *(1.0 / 20 for _ in K_AXIS[1:])),
        )
    )

    assert abs(weighted.rho - ignored.rho) > 1e-3


def test_a_thin_slice_keeps_its_shape_parameters_at_the_starting_values() -> None:
    """ADR-008's answer to risk 3 of Design 11: five parameters through four points is an
    interpolation with a spare degree of freedom, and the shape parameters are the two that absorb
    it. Below the threshold they are pinned and the level, wings and skew are still fitted."""
    thin = svi_task(k=K_AXIS[:3])

    result = make_jax_calibrator().calibrate(None, one_slice(thin)).slices[0]

    assert result.n_quotes_used == 3
    assert result.params.sigma == pytest.approx(0.10, abs=1e-5)
    assert not result.at_bound


@pytest.mark.parametrize(
    "params",
    [
        SVIParams(a=0.0, b=4.0, rho=0.0, m=0.0, sigma=1.0),
        SVIParams(a=0.0, b=0.5, rho=-0.9995, m=0.0, sigma=1.0),
        SVIParams(a=0.0, b=0.1, rho=0.0, m=2.0, sigma=1.0),
        SVIParams(a=0.0, b=0.1, rho=0.0, m=0.0, sigma=5.0),
    ],
)
def test_a_parameter_at_or_past_a_practical_bound_is_reported(params: SVIParams) -> None:
    """The bounds are the baseline's, imported rather than restated, because ``at_bound`` is half
    of ADR-006's acceptance rule and the rule has to be read against one ruler. The search here is
    unconstrained, so a fit can finish *past* a bound rather than pressed against it -- both mean
    what ADR-006 refuses."""
    assert _at_bound(params, pinned=False) is True


def test_a_healthy_slice_is_not_reported_as_pinned() -> None:
    """The other half: a rule that fired on every fit would refuse every surface."""
    assert _at_bound(TRUTH, pinned=False) is False


def test_a_pinned_shape_parameter_is_not_counted_against_the_fit() -> None:
    """A pinned ``m`` never moved, so reporting it as pinned by the optimiser would mean something
    ADR-006's rule does not: it refuses a fit that was *stopped* at the edge, not one held at a
    value chosen before the search began."""
    at_the_edge = SVIParams(a=0.0, b=0.1, rho=0.0, m=2.0, sigma=1.0)

    assert _at_bound(at_the_edge, pinned=True) is False


# --- the butterfly penalty


PENALTY_MESH: tuple[float, ...] = tuple(np.linspace(-1.1, 1.1, 41))
"""A mesh wider than any quoted band, so both wings and the bottom of the smile are sampled.

Wide on purpose: a sign slip in ``w'`` is antisymmetric about the minimum, so it cancels on a mesh
that only looks at one side of it or that is too narrow to let the wings diverge.
"""


def batched_g(params: SVIParams, mesh: tuple[float, ...]) -> np.ndarray:
    """This adapter's Durrleman function, evaluated the way the loss evaluates it."""
    return np.asarray(
        _durrleman_g(
            jnp.asarray(params.a, dtype=DTYPE),
            jnp.asarray(params.b, dtype=DTYPE),
            jnp.asarray(params.rho, dtype=DTYPE),
            jnp.asarray(params.m, dtype=DTYPE),
            jnp.asarray(params.sigma, dtype=DTYPE),
            jnp.asarray(mesh, dtype=DTYPE),
        ),
        dtype=np.float64,
    )


@pytest.mark.parametrize(("name", "params"), [("healthy", TRUTH), ("arbitrageable", ARBITRAGEABLE)])
def test_the_batched_durrleman_agrees_with_the_domain_pointwise(
    name: str, params: SVIParams
) -> None:
    """**The oracle test this copy of the formula exists under.**

    ``domain/durrleman.py`` cannot be differentiated by JAX (rule 3 keeps the library out of the
    domain) and a penalty an optimiser cannot differentiate is not a penalty, so the closed form is
    written twice. A duplication that is allowed to drift is a fork, and the whole loss rests on
    this one: a sign slip in ``w'`` or ``w''`` still produces a smooth, positive-on-a-healthy-slice
    ``g``, so nothing that looks at the *fit* would notice -- the surface would simply be pushed
    away from arbitrage it was not committing and towards arbitrage it was.

    Pointwise across the mesh rather than on a summary, because that is what pins the derivatives:
    the two curves have to agree at every ``k``, not merely have the same minimum.
    """
    ours = batched_g(params, PENALTY_MESH)
    theirs = durrleman_g(params, np.asarray(PENALTY_MESH, dtype=np.float64))

    for k, mine, oracle in zip(PENALTY_MESH, ours, theirs, strict=True):
        assert mine == pytest.approx(oracle, rel=1e-4, abs=1e-4), f"{name} at k={k}"


def test_the_mesh_the_oracle_runs_on_really_does_see_both_regimes() -> None:
    """The guard on the pair above. Two functions agree trivially on a mesh where neither varies,
    and a comparison that never reached a negative ``g`` would be checking the healthy branch of a
    formula whose whole purpose is the other one."""
    healthy = batched_g(TRUTH, PENALTY_MESH)
    arbitrageable = batched_g(ARBITRAGEABLE, PENALTY_MESH)

    assert healthy.min() > 0.0
    assert arbitrageable.min() < 0.0
    assert healthy.max() - healthy.min() > 0.5


def test_a_sign_slip_in_the_first_derivative_would_have_been_caught() -> None:
    """And the guard on *that*: the tolerance above has to be tight enough to reject the mistake
    it is watching for. Flipping the skew reverses the sign of ``w'`` about the minimum, which is
    exactly the shape of the slip, and it moves ``g`` by far more than 1e-4 somewhere on the mesh.
    """
    flipped = SVIParams(a=TRUTH.a, b=TRUTH.b, rho=-TRUTH.rho, m=TRUTH.m, sigma=TRUTH.sigma)

    difference = np.abs(batched_g(TRUTH, PENALTY_MESH) - batched_g(flipped, PENALTY_MESH))

    assert difference.max() > 1e-2


def test_the_penalty_charges_an_arbitrageable_slice_and_leaves_a_healthy_one_alone() -> None:
    """The soft constraint of Design 5.3, asserted on the objective rather than through a fit, so
    that it costs no second compilation. Soft rather than hard because the constraint is what the
    *fit* should be pushed towards, not what the *type* should refuse: ``SVIParams`` deliberately
    admits an arbitrageable slice so the violation stays measurable."""
    task = one_slice(
        svi_task(params=ARBITRAGEABLE, implied_vol=vols_of(ARBITRAGEABLE, K_AXIS, NEAR_TENOR))
    )
    data = jax.tree.map(lambda a: a[0], _to_batch(pad(task, TEST_SHAPE, 0.5)))
    free = jnp.asarray(_free_of(ARBITRAGEABLE))
    without = JaxFitSettings(durrleman_penalty_bp=0.0)

    charged = float(_objective(free, free, jnp.asarray([False] * 5), data, TEST_SETTINGS))
    plain = float(_objective(free, free, jnp.asarray([False] * 5), data, without))

    assert charged > plain
    healthy = jnp.asarray(_free_of(TRUTH))
    clean = jax.tree.map(lambda a: a[0], _to_batch(pad(one_slice(svi_task()), TEST_SHAPE, 0.5)))
    assert float(
        _objective(healthy, healthy, jnp.asarray([False] * 5), clean, TEST_SETTINGS)
    ) == pytest.approx(
        float(_objective(healthy, healthy, jnp.asarray([False] * 5), clean, without)), rel=1e-6
    )


def _free_of(params: SVIParams) -> tuple[float, ...]:
    """The free coordinates of a slice, through the adapter's own encoding."""
    from volengine.parametric_pricing.adapters.jax_calibrator import _encode

    return _encode(params)


# --- the port's promises


def test_the_producer_names_itself() -> None:
    calibrator = make_jax_calibrator()

    assert calibrator.producer_id == PRODUCER_ID
    assert calibrator.producer_id == "svi-jax"


def test_an_empty_producer_id_is_refused() -> None:
    with pytest.raises(ValueError, match="producer id"):
        JaxCalibrator(producer_id="   ", settings=TEST_SETTINGS, shape=TEST_SHAPE)


def test_the_same_task_twice_gives_the_same_answer_bit_for_bit() -> None:
    """ADR-004's replay rests on it, and a multi-start is exactly where a seeded restart would
    creep in. The offsets are fixed for that reason."""
    task = one_slice(svi_task())
    calibrator = make_jax_calibrator()

    first = calibrator.calibrate(None, task).slices[0]
    second = calibrator.calibrate(None, task).slices[0]

    assert first.params == second.params
    assert first.rmse_vol_bp == second.rmse_vol_bp


def test_the_duration_is_zero_because_no_clock_is_read() -> None:
    """Reading a clock is the one thing the port forbids outright: it would make two runs over one
    recording differ. What a consumer sees is the use case's own measurement."""
    result = make_jax_calibrator().calibrate(None, one_slice(svi_task()))

    assert result.duration_ms == 0.0


def test_the_iteration_count_is_objective_evaluations_and_is_never_zero() -> None:
    """ADR-027 left this stage the choice of what ``n_iterations`` carries, and it carries the same
    physical quantity the baseline reports: how many times the objective of a slice was evaluated,
    summed over the slices fitted. Never zero, because the starting point is evaluated before
    anything is decided."""
    result = make_jax_calibrator().calibrate(None, one_slice(svi_task()))

    assert result.n_iterations > 0


def test_the_iteration_count_does_not_grow_with_the_reservation() -> None:
    """A padded lane's search is an artefact of the fixed shape. Charging the caller for it would
    make the reported cost depend on how much margin the grid was given rather than on the market.
    Three times the rows, the same one real slice, and what the caller is charged stays put.

    **A tolerance and not an equality**, and the reason is a property of the platform rather than
    of this adapter. The searches run inside a ``vmap`` over the rows, and XLA vectorises a row's
    own reduction across that batch axis -- so how many rows were reserved decides how that row's
    thirty-two-lane sum is associated, and float32 addition is not associative. The same slice at
    the same point comes back with different bits under two reservations, the two searches part
    company after some thirty L-BFGS iterations, and the counts settle a couple of percent apart.
    Which reservations agree is a fact about the host's vector width, not about this code: capped
    at SSE4.2 every height agrees bit for bit, and under AVX2 one row already disagrees with two.
    The seam is in ``docs/SEAMS.md``.

    The tolerance is wide next to that drift and narrow next to the bug it exists to catch: the
    reserved rows are worth hundreds of evaluations, so a count that included them would read 1046
    against 2278 for this pair -- more than double, where ten percent is the gate. What the
    exclusion is asserted *exactly* against is the test below.
    """
    task = one_slice(svi_task())
    narrow = make_jax_calibrator().calibrate(None, task)

    tall = JaxCalibrator(
        settings=TEST_SETTINGS, shape=PadShape(max_slices=12, max_quotes=32, mesh_nodes=21)
    ).calibrate(None, task)

    assert tall.n_iterations == pytest.approx(narrow.n_iterations, rel=0.1)

    fitted, baseline = tall.slices[0].params, narrow.slices[0].params
    assert (fitted.a, fitted.b, fitted.rho, fitted.m, fitted.sigma) == pytest.approx(
        (baseline.a, baseline.b, baseline.rho, baseline.m, baseline.sigma), rel=1e-2
    )


def test_the_iteration_count_excludes_the_reserved_rows() -> None:
    """The same property where it is exact: one reservation, one compilation, and the adapter's own
    array of evaluations rather than two fits compared across shapes.

    Reaching past the port for ``_compiled`` is what makes the claim checkable at all. Reading the
    exclusion off two reservations -- the test above -- can only see the padding through arithmetic
    that is not bit-stable across shapes, which is why that one states the property loosely. Here
    both numbers come off one rectangle and nothing rounds between them.

    The second assertion is what stops the first from passing vacuously: if a reserved lane's
    search were free, masking it out would be indistinguishable from leaving it in.
    """
    shape = PadShape(max_slices=12, max_quotes=32, mesh_nodes=21)
    task = one_slice(svi_task())
    padded = pad(task, shape, TEST_SETTINGS.durrleman_mesh_margin)
    _, cold = _compiled(TEST_SETTINGS, shape)

    spent = np.asarray(cold(_to_batch(padded)).evaluations)
    result = JaxCalibrator(settings=TEST_SETTINGS, shape=shape).calibrate(None, task)

    assert result.n_iterations == int(spent[np.asarray(padded.slice_mask)].sum())
    assert result.n_iterations < int(spent.sum())


def test_a_reserved_row_really_does_run_a_search_of_its_own() -> None:
    """The guard on the test above. If an empty lane cost nothing, the count would be unchanged
    whether or not ``_result`` excluded the padding, and the property would be untested.

    It costs several evaluations -- an all-masked slice has a zero objective from the first step,
    and the early stop still needs its run of quiet steps to notice -- so eight reserved rows are
    worth a couple of hundred evaluations across the multi-start. Every one of them is excluded.
    """
    padded = pad(one_slice(svi_task()), TEST_SHAPE, TEST_SETTINGS.durrleman_mesh_margin)
    empty = jax.tree.map(lambda row: row[1], _to_batch(padded))
    start = jnp.asarray([-3.0, -2.0, -0.3, 0.0, -2.2])

    assert bool(padded.slice_mask[1]) is False
    for search in (_adam_fit, _lbfgs_fit):
        spent = search(start, empty, TEST_SETTINGS)
        assert int(spent.evaluations) > 0
        assert float(spent.cost) == 0.0


def test_a_chain_wider_than_the_reservation_is_refused_rather_than_truncated() -> None:
    """The seam ADR-009 leaves open: growing the reservation is a new compilation, so it is a
    restart. Until ``ChainCompositionChanged`` is handled, the honest answer is to refuse the task
    and let the use case republish the last good surface."""
    task = one_slice(svi_task(k=K_AXIS))

    with pytest.raises(CalibrationError, match="reserves 8 quotes"):
        JaxCalibrator(
            settings=TEST_SETTINGS, shape=PadShape(max_slices=2, max_quotes=8, mesh_nodes=7)
        ).calibrate(None, task)


def test_every_fitted_parameter_is_finite_and_the_slice_is_a_surface() -> None:
    """The reparametrisation's central claim: every point of R^5 decodes to a valid slice, so no
    search can produce one ``SVIParams`` would refuse. Asserted through the constructor, which is
    the thing that would have raised."""
    result = make_jax_calibrator().calibrate(None, one_slice(svi_task()))
    params = result.slices[0].params

    assert math.isfinite(params.a)
    assert params.b >= 0
    assert abs(params.rho) < 1
    assert params.sigma > 0
    assert params.min_total_variance >= 0
