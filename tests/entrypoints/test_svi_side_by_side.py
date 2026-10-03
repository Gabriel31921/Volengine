"""The two SVI engines on one market, from the file F3-W1 ships: ``examples/svi-scipy-vs-jax.toml``.

The configuration is the deliverable here: ``svi-jax`` reachable from TOML, its tuning in
``[calibration.jax]`` beside the baseline's ``[calibration.fit]``, and the metrics persisted. So the
file itself is loaded, and the run below is the file's market and calibration with only the far
ends swapped -- a writer the test can read and a sink it can query -- and the feed's timings
shortened so the session takes a fraction of a second rather than twenty.

Everything that needs the ``jax`` extra is skipped without it, except the one behaviour that only
exists without it: the refusal that names the extra, which is observed on every installation by
hiding the extra from the loader's lookup.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path

import pytest

from tests.entrypoints.builders import RecordingWriter, make_risk_config
from tests.support import RecordingMetrics
from volengine.entrypoints import config as config_module
from volengine.entrypoints.config import (
    JAX_CALIBRATOR,
    JAX_EXTRA_LIBRARIES,
    AppConfig,
    ConfigError,
    MetricsSinkKind,
    load_config,
)
from volengine.entrypoints.pipeline import Adapters, build_pipeline, default_adapters
from volengine.market_data.adapters.synthetic import SyntheticProvider
from volengine.parametric_pricing.adapters.scipy_calibrator import PRODUCER_ID as SCIPY_ID
from volengine.platform.bus import InProcessConflatingBus
from volengine.platform.clock import SystemClock
from volengine.risk.domain.portfolio import Position
from volengine.risk.domain.pricing import OptionKindR

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "svi-scipy-vs-jax.toml"
"""The configuration shipped for F3-W1. Its own header holds the commands."""

HAS_JAX_EXTRA = all(find_spec(name) is not None for name in JAX_EXTRA_LIBRARIES)

needs_jax = pytest.mark.skipif(not HAS_JAX_EXTRA, reason="needs the jax extra")


@needs_jax
def test_the_shipped_example_names_only_adapters_this_build_registers() -> None:
    """The guard on the file a reader runs, without pinning it to the date of its position."""
    config = load_config(EXAMPLE)
    registry = default_adapters()

    assert list(config.calibration.calibrators) == [SCIPY_ID, JAX_CALIBRATOR]
    assert all(name in registry.calibrators for name in config.calibration.calibrators)
    assert config.markets[0].provider in registry.providers
    assert config.risk.writer in registry.writers
    assert config.metrics.sink is MetricsSinkKind.CSV


@needs_jax
def test_the_two_tables_state_one_loss() -> None:
    """The numbers that define the objective agree, so the comparison is of two searches.

    Design 5.7 compares methods, not problems: a different Huber scale, penalty, ridge or pinning
    threshold in one table would make the distance between the two surfaces a measure of the
    configuration rather than of the optimisers.
    """
    calibration = load_config(EXAMPLE).calibration
    fit, jax = calibration.fit, calibration.jax

    assert fit is not None
    assert jax is not None
    assert (
        fit.huber_scale_bp,
        fit.durrleman_penalty_bp,
        fit.durrleman_mesh_margin,
        fit.ridge_bp,
        fit.min_quotes_for_free_shape,
    ) == (
        jax.huber_scale_bp,
        jax.durrleman_penalty_bp,
        jax.durrleman_mesh_margin,
        jax.ridge_bp,
        jax.min_quotes_for_free_shape,
    )


def test_the_shipped_example_without_the_extra_is_refused_with_the_remedy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What a reader without ``uv sync --extra jax`` meets: the command, not a traceback."""
    monkeypatch.setattr(
        config_module, "find_spec", lambda name: None if name == "jax" else find_spec(name)
    )

    with pytest.raises(ConfigError, match="uv sync --extra jax"):
        load_config(EXAMPLE)


def runnable(config: AppConfig, writer: RecordingWriter) -> tuple[AppConfig, Adapters]:
    """The example, sped up and given a book on an expiry the feed really quotes.

    The feed's interval goes from a second to a twentieth and its cycles from twenty to six, which
    changes how long the session takes and nothing about what it quotes; its start is pinned so the
    expiry the position names is the one the pipeline's own provider will place. The writer is the
    only adapter swapped, so both calibrators come out of the real registry.
    """
    market = config.markets[0]
    assert market.synthetic is not None
    start = datetime.now(UTC)
    synthetic = replace(
        market.synthetic,
        config=replace(market.synthetic.config, interval_seconds=0.05, cycles=6),
        start=start,
    )
    market = replace(market, synthetic=synthetic)
    expiry = SyntheticProvider(market.conventions, synthetic.config, start).instruments[0].expiry
    position = Position(
        underlying=market.underlying,
        expiry=expiry,
        strike=synthetic.config.forward0,
        kind=OptionKindR.CALL,
        quantity=1.0,
    )
    registry = default_adapters()
    adapters = Adapters(
        providers=registry.providers,
        calibrators=registry.calibrators,
        writers={"recording": lambda _risk: writer},
    )
    risk = make_risk_config(positions=(position,))
    return replace(config, markets=(market,), risk=risk), adapters


@needs_jax
@pytest.mark.e2e
async def test_both_engines_publish_and_are_compared_on_one_market() -> None:
    """Surfaces from both producers reach Risk, and each arrival measures the distance between them.

    The comparison is the reason two engines run side by side (Design 7.3): the first producer
    listed is the baseline, the second the challenger, and ``risk.comparison.*`` is only emitted
    once the cache holds a surface from each.
    """
    writer, metrics = RecordingWriter(), RecordingMetrics()
    config, adapters = runnable(load_config(EXAMPLE), writer)

    pipeline = build_pipeline(
        config, adapters, SystemClock(), InProcessConflatingBus(metrics), metrics
    )
    await pipeline.run()

    assert {report.producer_id for report in writer.reports} == {SCIPY_ID, JAX_CALIBRATOR}
    compared = [tags for name, _, tags in metrics.gauges if name.startswith("risk.comparison.")]
    assert compared
    assert {tags.get("challenger") for tags in compared} == {JAX_CALIBRATOR}
