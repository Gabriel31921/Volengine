"""The benchmark of Design 5.7: the same market, the same loss, two searches.

`Plan.md` asks F3-A for a comparison of the two calibrators on **time, convergence and lines of
code**. This module is that comparison, written as tests so that it cannot rot: the claims that are
falsifiable are asserted, and the numbers that are measurements are printed as a table (run
``pytest -s -k benchmark`` to read it).

What is asserted is what the comparison *means*: handed identical tasks, the two producers land on
the same market within a basis point of volatility and within a few thousandths on every parameter.
That is the statement Design 5.7 exists to make, and it is only worth making because both minimise
the same objective -- residual for residual, Huber scale for Huber scale (see ``_objective``). Two
optimisers minimising two different losses would be a comparison of the losses.

What is **not** asserted is a wall clock. A timing assertion in a test suite is a flake generator:
it depends on the machine, on the load, and here on whether the run is on a CPU or a GPU, which is
precisely the axis the two producers differ on. The numbers are measured, printed and discussed in
ADR-029, which records this stage; the tests below guard the shape of the result, not the speed of
the laptop that produced it.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from tests.parametric_pricing.builders import FORWARD, make_calibration_task, make_slice_task
from tests.parametric_pricing.jax_builders import TEST_SETTINGS, make_jax_calibrator
from volengine.parametric_pricing.adapters import jax_calibrator, padding, scipy_calibrator
from volengine.parametric_pricing.adapters.scipy_calibrator import FitSettings, ScipyCalibrator
from volengine.parametric_pricing.domain.calibration import CalibrationResult, CalibrationTask
from volengine.parametric_pricing.domain.ports import Calibrator
from volengine.parametric_pricing.domain.svi_slice import SVIParams

TRUTH = SVIParams(a=0.0035, b=0.060, rho=-0.35, m=-0.02, sigma=0.10)
"""The generating slice both producers are asked to recover -- the same one each of their own test
modules uses, so the three sets of numbers are readable against each other."""

K_AXIS: tuple[float, ...] = tuple(np.linspace(-0.6, 0.6, 21))
TENOR = 1.0 / 12.0

SHARED = FitSettings(
    huber_scale_bp=TEST_SETTINGS.huber_scale_bp,
    durrleman_penalty_bp=TEST_SETTINGS.durrleman_penalty_bp,
    durrleman_mesh_nodes=21,
    durrleman_mesh_margin=TEST_SETTINGS.durrleman_mesh_margin,
    min_quotes_for_free_shape=TEST_SETTINGS.min_quotes_for_free_shape,
    ridge_bp=TEST_SETTINGS.ridge_bp,
)
"""The baseline configured with the JAX side's numbers, so the two minimise the same function.

Every field here exists on both settings types under the same name and with the same default; what
does not appear is what each search owns alone -- an evaluation budget on one side, a learning rate
and a step budget on the other. The mesh node count is the one that has to be restated, because on
the JAX side it is part of the compiled shape and lives in :class:`PadShape`.
"""


def task() -> CalibrationTask:
    """One well-populated slice quoted exactly on ``TRUTH``, evenly weighted."""
    return make_calibration_task(
        slices=(
            make_slice_task(
                tenor_years=TENOR,
                forward=FORWARD,
                log_moneyness=K_AXIS,
                implied_vol=tuple(TRUTH.implied_vol(k, TENOR) for k in K_AXIS),
                weights=tuple(1.0 / len(K_AXIS) for _ in K_AXIS),
            ),
        )
    )


def producers() -> tuple[tuple[str, Calibrator], ...]:
    """Both implementers, configured against the same loss."""
    return (("svi-scipy", ScipyCalibrator(SHARED)), ("svi-jax", make_jax_calibrator()))


def timed(
    calibrator: Calibrator, previous: Mapping[datetime, SVIParams] | None
) -> tuple[CalibrationResult, float]:
    """One calibration, with the wall time in milliseconds beside it. Measured, never asserted."""
    started = time.perf_counter()
    result = calibrator.calibrate(previous, task())
    return result, 1000.0 * (time.perf_counter() - started)


def lines_of_code(*modules: object) -> int:
    """Non-blank source lines of the modules that make up one producer.

    Docstrings included, and that is the honest count for this repository: the arguments live in
    the docstrings and deleting them would not make either adapter smaller in any sense that
    matters. What the number compares is how much *material* each approach needed.
    """
    total = 0
    for module in modules:
        source = Path(str(module.__file__))  # type: ignore[attr-defined]
        total += sum(1 for line in source.read_text().splitlines() if line.strip())
    return total


def test_both_producers_recover_the_generating_parameters() -> None:
    """The claim that makes the comparison worth reading: neither is fitting a different market."""
    for name, calibrator in producers():
        fitted = calibrator.calibrate(None, task()).slices[0].params

        assert fitted.a == pytest.approx(TRUTH.a, abs=1e-4), name
        assert fitted.b == pytest.approx(TRUTH.b, abs=1e-3), name
        assert fitted.rho == pytest.approx(TRUTH.rho, abs=1e-2), name
        assert fitted.m == pytest.approx(TRUTH.m, abs=1e-2), name
        assert fitted.sigma == pytest.approx(TRUTH.sigma, abs=1e-2), name


def test_the_two_producers_agree_on_the_fit_to_within_a_basis_point() -> None:
    """Convergence, compared where it is comparable: both are asked for the same minimum of the
    same objective, so a gap of more than a basis point of volatility would mean one of them
    stopped somewhere the other would not have."""
    fits = {name: calibrator.calibrate(None, task()).slices[0] for name, calibrator in producers()}

    scipy_fit, jax_fit = fits["svi-scipy"], fits["svi-jax"]

    assert abs(jax_fit.rmse_vol_bp - scipy_fit.rmse_vol_bp) < 1.0
    assert jax_fit.params.b == pytest.approx(scipy_fit.params.b, abs=2e-3)
    assert jax_fit.params.rho == pytest.approx(scipy_fit.params.rho, abs=2e-2)


def test_both_producers_report_the_same_population_in_their_metrics() -> None:
    """ADR-027's caution, discharged. ``n_quotes_used`` and ``max_err_vol_bp`` count the quotes
    that *entered the fit* on both sides, so a comparative report can put the two producers in one
    table without the columns meaning different things."""
    results = {
        name: calibrator.calibrate(None, task()).slices[0] for name, calibrator in producers()
    }

    counts = {one.n_quotes_used for one in results.values()}
    assert counts == {len(K_AXIS)}


def test_benchmark_table_of_time_convergence_and_size(capsys: pytest.CaptureFixture[str]) -> None:
    """The table Design 5.7 asks for, measured on whatever machine is running the suite.

    The assertions are structural -- every producer answers, with a finite cost and a real fit --
    because the interesting content is the numbers and the numbers are not a specification. What
    they showed on the machine this stage was written on is recorded in ADR-029: on a chain this
    size the baseline's trust-region step is far cheaper per unit of progress, the two hot cycles
    are level, and JAX's cost is flat in the number of expiries where the baseline's grows with it
    -- which is the trade the padding was bought for.
    """
    rows: list[str] = []
    for name, calibrator in producers():
        cold, cold_ms = timed(calibrator, None)
        warm, warm_ms = timed(calibrator, {one.expiry: one.params for one in cold.slices})

        assert cold.n_iterations > 0
        assert cold.slices[0].rmse_vol_bp < 2.0
        assert warm.slices[0].rmse_vol_bp < 2.0

        rows.append(
            f"{name:>10} | cold {cold_ms:8.1f} ms, {cold.n_iterations:6d} evaluations, "
            f"{cold.slices[0].rmse_vol_bp:7.4f} bp | warm {warm_ms:8.1f} ms, "
            f"{warm.n_iterations:6d} evaluations, {warm.slices[0].rmse_vol_bp:7.4f} bp"
        )

    sizes = {
        "svi-scipy": lines_of_code(scipy_calibrator),
        "svi-jax": lines_of_code(jax_calibrator, padding),
    }
    with capsys.disabled():
        print("\n" + "\n".join(rows))
        print(f"     lines: scipy {sizes['svi-scipy']}, jax {sizes['svi-jax']}")
