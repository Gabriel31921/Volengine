"""The torch learner, judged as an implementer of the port and as a producer of surfaces.

Four groups. **Configuration**: every guard on the two settings objects, NaN included, because a
NaN learning rate that slipped through would produce a network of NaN on the first step and a NaN
penalty weight would switch the soft tier off without anyone noticing. **The port's promises**:
purity, determinism, a new object every time, the previous surface untouched, and the refusals a
learner owes the use case instead of a silent cold start. **The soft tier**: that the penalties
reduce the violations they are named after, measured against the same learner with the term
switched off -- an assertion that "the penalty is on" would be decoration without the off case
beside it. **Accuracy and the update regime**: the known-truth bound of Design 8.2, the warm step
that must not wreck a converged fit, the fine-tune that must follow a market, and one cycle through
the real use case.

The whole module is skipped from collection when the ``neural`` extra is absent
(``tests/neural_surface/conftest.py``).
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest
import torch

from tests.neural_surface.builders import (
    NOW,
    FlatVolSurface,
    make_buffer,
    make_grid_spec,
    make_mesh,
    make_sample,
    make_schedule,
    make_thresholds,
    make_weighting,
)
from tests.neural_surface.torch_builders import (
    DENSE_MONEYNESS,
    TEST_SETTINGS,
    TEST_SPEC,
    make_snapshot_batch,
    make_torch_learner,
)
from tests.parametric_pricing.builders import make_market_snapshot, make_params
from tests.support import RecordingMetrics, replace_field
from volengine.contracts.events import SurfaceCalibrated
from volengine.neural_surface.adapters.torch_learner import (
    PRODUCER_ID,
    Activation,
    NetworkSpec,
    TorchFitSettings,
    TorchLearner,
    TorchSurface,
    _durrleman_g,
)
from volengine.neural_surface.application.acl import fit_metrics
from volengine.neural_surface.application.train_on_snapshot import TrainOnSnapshot
from volengine.neural_surface.application.training_state import TrainingState
from volengine.neural_surface.domain.errors import NeuralSurfaceError
from volengine.neural_surface.domain.invariants import check_surface, durrleman_g
from volengine.neural_surface.domain.learned_surface import evaluate_total_variance
from volengine.neural_surface.domain.ports import SurfaceLearner
from volengine.neural_surface.domain.training_batch import TrainingBatch
from volengine.platform.clock import ManualClock

K_AXIS = np.linspace(-0.5, 0.5, 7)
TENOR_AXIS = np.asarray([1.0 / 12.0, 0.25, 0.5])
"""A rectangular evaluation mesh, seven strikes by three tenors, so a transposed answer cannot
hide."""

GENEROUS_GATE = make_thresholds(butterfly=1e-3, calendar=1e-4)
"""Tolerances a soft-constrained fit can honestly meet: a thousandth of a unit of ``g`` and a
ten-thousandth of a unit of total variance. The soft tier is a preference, not a guarantee
(ADR-010), and a gate at exactly zero would be asserting the guarantee the design says not to."""


def grid(surface: TorchSurface) -> np.ndarray:
    return surface.total_variance(K_AXIS, TENOR_AXIS)


def rmse_bp(surface: TorchSurface, batch: TrainingBatch) -> float:
    metrics = fit_metrics(surface=surface, fresh=batch.samples, n_iterations=1, duration_ms=0.0)
    return metrics.rmse_vol_bp


# --- configuration


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hidden", ()),
        ("hidden", (32, 0)),
        ("hidden", (-1,)),
        ("k_scale", 0.0),
        ("k_scale", -0.5),
        ("k_scale", float("nan")),
        ("k_scale", float("inf")),
        ("tenor_scale", 0.0),
        ("tenor_scale", float("nan")),
    ],
)
def test_network_spec_refuses_a_shape_it_cannot_build(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="must"):
        replace_field(NetworkSpec(), field, value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cold_steps", 0),
        ("cold_steps", -1),
        ("warm_steps", 0),
        ("cold_learning_rate", 0.0),
        ("cold_learning_rate", float("nan")),
        ("cold_learning_rate", float("inf")),
        ("warm_learning_rate", -1e-3),
        ("warm_learning_rate", float("nan")),
        ("butterfly_penalty", -1.0),
        ("butterfly_penalty", float("nan")),
        ("butterfly_penalty", float("inf")),
        ("calendar_penalty", -1.0),
        ("calendar_penalty", float("nan")),
    ],
)
def test_fit_settings_refuse_a_value_that_cannot_train(field: str, value: object) -> None:
    """Every guard, and the NaN case for each float, because ``nan <= 0`` is ``False``."""
    with pytest.raises(ValueError, match="must be"):
        replace_field(TorchFitSettings(), field, value)


def test_a_penalty_weight_of_zero_is_legal() -> None:
    """Zero is how the soft tier is switched off: the experiment, not a misconfiguration."""
    settings = TorchFitSettings(butterfly_penalty=0.0, calendar_penalty=0.0)

    assert settings.butterfly_penalty == 0.0
    assert settings.calendar_penalty == 0.0


def test_relu_is_not_an_activation_this_adapter_offers() -> None:
    """Design 6.2: smooth activations only. The enum is the enforcement, so it must not grow one."""
    assert {member.value for member in Activation} == {"tanh", "softplus"}
    with pytest.raises(ValueError):
        Activation("relu")


def test_every_activation_member_has_the_string_a_configuration_file_would_name() -> None:
    for member in Activation:
        assert isinstance(member.value, str)
        assert member.value == member.name.lower()


# --- the port's promises


def test_the_learner_satisfies_the_port_and_can_be_called_through_it() -> None:
    """The one structural check that exists: an assignment to the port type, and a call through it.

    ``mypy`` checks the assignment; the call is what proves the shape at runtime.
    """
    learner: SurfaceLearner = make_torch_learner()

    surface = learner.update(None, make_snapshot_batch())

    assert learner.producer_id == PRODUCER_ID
    assert isinstance(surface, TorchSurface)


def test_the_producer_names_itself_and_keeps_the_name() -> None:
    assert make_torch_learner().producer_id == "mlp-torch"
    assert make_torch_learner().producer_id == make_torch_learner().producer_id


def test_a_cold_start_is_version_one_and_a_warm_start_counts_up() -> None:
    learner = make_torch_learner()
    batch = make_snapshot_batch()

    first = learner.update(None, batch)
    second = learner.update(first, batch)
    third = learner.update(second, batch)

    assert (first.version, second.version, third.version) == (1, 2, 3)


def test_a_restart_resets_the_version_because_it_has_no_history_to_count_from() -> None:
    """``update(None, ...)`` after a lineage is the scheduled restart (ADR-019), visibly so."""
    learner = make_torch_learner()
    batch = make_snapshot_batch()
    warmed = learner.update(learner.update(None, batch), batch)

    restarted = learner.update(None, batch)

    assert warmed.version == 2
    assert restarted.version == 1


def test_an_update_returns_a_new_object_and_not_the_previous_one() -> None:
    learner = make_torch_learner()
    batch = make_snapshot_batch()
    previous = learner.update(None, batch)

    updated = learner.update(previous, batch)

    assert updated is not previous
    assert all(new is not old for new, old in zip(updated.weights, previous.weights, strict=True))


def test_the_same_cold_start_twice_gives_the_same_surface_bit_for_bit() -> None:
    """ADR-004: a recorded session replays exactly, so a cold start may not depend on anything
    but its arguments and its configured seed."""
    batch = make_snapshot_batch()

    first = make_torch_learner().update(None, batch)
    second = make_torch_learner().update(None, batch)

    assert np.array_equal(grid(first), grid(second))
    assert all(torch.equal(a, b) for a, b in zip(first.weights, second.weights, strict=True))


def test_the_same_previous_and_batch_give_the_same_surface_twice() -> None:
    """The guard on the optimiser's moments.

    ``Optimizer.load_state_dict`` hands back the very tensor it was given when dtype and device
    match, and Adam updates its moments in place -- so a warm start that loaded ``previous``'s
    moments without copying them would rewrite ``previous`` and answer differently the second
    time. This is the test that fails if the deep copy is ever removed.
    """
    learner = make_torch_learner()
    batch = make_snapshot_batch()
    previous = learner.update(None, batch)

    first = learner.update(previous, batch)
    second = learner.update(previous, batch)

    assert np.array_equal(grid(first), grid(second))


def test_a_warm_start_leaves_the_previous_surface_untouched() -> None:
    learner = make_torch_learner()
    batch = make_snapshot_batch()
    previous = learner.update(None, batch)
    weights_before = tuple(w.clone() for w in previous.weights)
    grid_before = grid(previous).copy()
    moments_before = {
        index: {name: value.clone() for name, value in state.items()}
        for index, state in previous.moments["state"].items()
    }

    learner.update(previous, batch)

    assert np.array_equal(grid(previous), grid_before)
    assert all(torch.equal(a, b) for a, b in zip(previous.weights, weights_before, strict=True))
    for index, state in previous.moments["state"].items():
        for name, value in state.items():
            assert torch.equal(value, moments_before[index][name]), (index, name)


def test_a_warm_start_continues_from_the_previous_surface_rather_than_restarting() -> None:
    """A learner that accepted ``previous`` and ignored it would pass every type check.

    Five steps at the warm rate move a surface by a small amount; a cold start lands somewhere
    else entirely. So the warm answer must be close to what it was handed and far from the cold
    answer trained from the same batch.
    """
    learner = make_torch_learner(
        settings=replace(TEST_SETTINGS, init_seed=1),
    )
    batch = make_snapshot_batch()
    previous = make_torch_learner().update(None, batch)

    warmed = learner.update(previous, batch)
    cold = learner.update(None, batch)

    drift_from_previous = np.max(np.abs(grid(warmed) - grid(previous)))
    distance_to_cold = np.max(np.abs(grid(warmed) - grid(cold)))
    assert drift_from_previous < 0.1 * distance_to_cold


def test_a_previous_surface_of_another_kind_is_refused_not_silently_cold_started() -> None:
    learner = make_torch_learner()

    with pytest.raises(NeuralSurfaceError, match="can only fine-tune"):
        learner.update(FlatVolSurface(), make_snapshot_batch())


def test_a_previous_surface_trained_under_another_spec_is_refused() -> None:
    batch = make_snapshot_batch()
    previous = make_torch_learner(spec=NetworkSpec(hidden=(8,))).update(None, batch)

    with pytest.raises(NeuralSurfaceError, match="cannot fine-tune"):
        make_torch_learner().update(previous, batch)


def test_the_surface_answers_tenor_major_on_a_rectangular_mesh() -> None:
    """The transposition an adapter gets wrong on its first day, invisible on a square mesh."""
    surface = make_torch_learner().update(None, make_snapshot_batch())

    w = evaluate_total_variance(surface, K_AXIS, TENOR_AXIS)

    assert w.shape == (len(TENOR_AXIS), len(K_AXIS))
    # Row i belongs to tenors[i] and column j to k[j]: every mesh node must agree with the same
    # surface asked about that one point alone, which is a statement no heuristic about the shape
    # of the fit could make.
    for i, tenor in enumerate(TENOR_AXIS):
        for j, k in enumerate(K_AXIS):
            alone = surface.total_variance(np.asarray([k]), np.asarray([tenor]))
            assert w[i, j] == pytest.approx(alone[0, 0], rel=1e-12, abs=1e-15)


def test_total_variance_is_strictly_positive_even_from_an_untrained_network() -> None:
    """Softplus on the way out: no iterate, not even the first, can hand the gate a zero."""
    barely_trained = make_torch_learner(settings=replace(TEST_SETTINGS, cold_steps=1))

    w = barely_trained.update(None, make_snapshot_batch()).total_variance(
        np.linspace(-3.0, 3.0, 13), np.asarray([1e-3, 1.0, 5.0])
    )

    assert np.all(np.isfinite(w)) and np.all(w > 0.0)


def test_a_cold_start_does_not_consume_the_global_torch_generator() -> None:
    """No global state: the cold-start weights come from a generator local to the call.

    ``torch.nn.Linear`` would have drawn from the global generator in its constructor; building
    the tensors by hand is what keeps the global stream where another library left it.
    """
    before = torch.random.get_rng_state().clone()

    make_torch_learner().update(None, make_snapshot_batch())

    assert torch.equal(torch.random.get_rng_state(), before)


def test_the_seed_is_read() -> None:
    batch = make_snapshot_batch()

    default = make_torch_learner().update(None, batch)
    reseeded = make_torch_learner(settings=replace(TEST_SETTINGS, init_seed=7)).update(None, batch)

    assert not np.array_equal(grid(default), grid(reseeded))


def test_the_cold_budget_is_read() -> None:
    batch = make_snapshot_batch()

    short = make_torch_learner(settings=replace(TEST_SETTINGS, cold_steps=1)).update(None, batch)
    long = make_torch_learner().update(None, batch)

    assert rmse_bp(long, batch) < rmse_bp(short, batch)


def test_the_warm_budget_is_read() -> None:
    batch = make_snapshot_batch()
    previous = make_torch_learner().update(None, batch)

    one = make_torch_learner(settings=replace(TEST_SETTINGS, warm_steps=1)).update(previous, batch)
    many = make_torch_learner(settings=replace(TEST_SETTINGS, warm_steps=50)).update(
        previous, batch
    )

    assert np.max(np.abs(grid(one) - grid(previous))) < np.max(np.abs(grid(many) - grid(previous)))


def test_the_warm_learning_rate_is_read_rather_than_restored_from_the_moments() -> None:
    """``Optimizer.load_state_dict`` restores the saved learning rate along with the moments; the
    configured one has to win, or a change to ``warm_learning_rate`` would silently do nothing."""
    batch = make_snapshot_batch()
    previous = make_torch_learner().update(None, batch)

    gentle = make_torch_learner(settings=replace(TEST_SETTINGS, warm_learning_rate=1e-4)).update(
        previous, batch
    )
    brisk = make_torch_learner(settings=replace(TEST_SETTINGS, warm_learning_rate=1e-1)).update(
        previous, batch
    )

    gentle_move = np.max(np.abs(grid(gentle) - grid(previous)))
    brisk_move = np.max(np.abs(grid(brisk) - grid(previous)))
    assert gentle_move < 0.1 * brisk_move


def test_the_batch_weights_are_consumed_as_given() -> None:
    """A poisoned point at zero weight must steer nothing; the same point at full weight must.

    The vacuity guard: without the second half, a learner that ignored the weights entirely and
    happened to fit the rest well enough would pass the first.
    """
    batch = make_snapshot_batch()
    poison = make_sample(log_moneyness=0.0, tenor_years=1.0 / 12.0, implied_vol=3.0, weight=0.0)
    ignored = replace(batch, samples=(*batch.samples, poison))
    heeded = replace(batch, samples=(*batch.samples, replace(poison, weight=5.0)))

    clean = make_torch_learner().update(None, batch)
    with_ignored = make_torch_learner().update(None, ignored)
    with_heeded = make_torch_learner().update(None, heeded)

    assert np.array_equal(grid(with_ignored), grid(clean))
    assert not np.array_equal(grid(with_heeded), grid(clean))


# --- the soft tier


def test_the_torch_durrleman_agrees_with_the_domain() -> None:
    """Two copies of one formula, held together: a sign slip in ``w'`` still yields a plausible
    ``g``, so the penalty's copy is pinned against the gate's rather than trusted."""
    mesh = make_mesh()
    k = mesh.k_array
    tenors = mesh.tenor_array
    params = make_params()
    w = np.vstack([params.total_variance(k) * (12.0 * t) for t in tenors])
    step = float(k[1] - k[0])
    k_mesh, _ = torch.meshgrid(torch.as_tensor(k), torch.as_tensor(tenors), indexing="xy")

    ours = _durrleman_g(torch.as_tensor(w), k_mesh, step).numpy()
    theirs = np.vstack([durrleman_g(row, k) for row in w])

    assert ours.shape == theirs.shape == (len(tenors), len(k) - 2)
    np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-12)


def crossing_chain() -> TrainingBatch:
    """Eighty percent vol at one month over forty at three: ``w`` runs backwards in the tenor."""
    samples = tuple(
        make_sample(log_moneyness=k, tenor_years=tenor, implied_vol=vol, weight=1.0)
        for tenor, vol in ((1.0 / 12.0, 0.80), (0.25, 0.40))
        for k in DENSE_MONEYNESS
    )
    return TrainingBatch(
        market_id="BTC-DERIBIT",
        snapshot_id="snap-crossing",
        ts_snapshot=NOW,
        samples=samples,
        n_fresh=len(samples),
    )


def spiked_smile() -> TrainingBatch:
    """A flat smile with a narrow bump at the money: Durrleman's ``g`` is deeply negative there."""
    tenor = 1.0 / 12.0

    def total_variance(k: float) -> float:
        return 0.04 + 0.03 * math.exp(-((k / 0.05) ** 2))

    samples = tuple(
        make_sample(
            log_moneyness=k,
            tenor_years=tenor,
            implied_vol=math.sqrt(total_variance(k) / tenor),
            weight=1.0,
        )
        for k in DENSE_MONEYNESS
    )
    return TrainingBatch(
        market_id="BTC-DERIBIT",
        snapshot_id="snap-spike",
        ts_snapshot=NOW,
        samples=samples,
        n_fresh=len(samples),
    )


def test_the_spiked_smile_really_carries_a_butterfly_violation() -> None:
    """The vacuity guard on the test below: the data must be arbitrageable, or the penalty test
    would pass against a smile the network could fit cleanly anyway."""
    k = np.asarray(DENSE_MONEYNESS)
    w = np.asarray([s.total_variance for s in spiked_smile().samples])

    assert np.min(durrleman_g(w, k)) < -1.0


def test_the_calendar_penalty_reduces_the_crossing_the_data_asks_for() -> None:
    settings_off = TorchFitSettings(cold_steps=600, calendar_penalty=0.0)
    settings_on = replace(settings_off, calendar_penalty=1000.0)

    unconstrained = make_torch_learner(settings=settings_off).update(None, crossing_chain())
    constrained = make_torch_learner(settings=settings_on).update(None, crossing_chain())

    off = check_surface(unconstrained, make_mesh()).calendar_violation
    on = check_surface(constrained, make_mesh()).calendar_violation
    assert off > 1e-3, "without the term the network follows the crossing data"
    assert on < 0.25 * off


def test_the_butterfly_penalty_removes_the_violation_the_data_asks_for() -> None:
    settings_off = TorchFitSettings(cold_steps=600, butterfly_penalty=0.0)
    settings_on = replace(settings_off, butterfly_penalty=1000.0)

    unconstrained = make_torch_learner(settings=settings_off).update(None, spiked_smile())
    constrained = make_torch_learner(settings=settings_on).update(None, spiked_smile())

    off = check_surface(unconstrained, make_mesh()).butterfly_violation
    on = check_surface(constrained, make_mesh()).butterfly_violation
    assert off > 1.0, "without the term the network follows the spike"
    assert on < 0.01 * off


def test_an_arbitrageable_surface_is_returned_for_the_gate_to_judge_not_raised() -> None:
    """Soft constraints train, hard constraints govern: the learner never refuses a surface."""
    learner = make_torch_learner(
        settings=TorchFitSettings(cold_steps=600, butterfly_penalty=0.0, calendar_penalty=0.0)
    )

    surface = learner.update(None, crossing_chain())

    report = check_surface(surface, make_mesh())
    assert report.exceeds(GENEROUS_GATE.butterfly, GENEROUS_GATE.calendar)


# --- accuracy and the update regime


def test_the_shipped_defaults_recover_a_known_surface_within_tolerance() -> None:
    """Design 8.2's known-truth row for the neural producer: quotes from a known SVI, fitted from
    scratch, land within a bound a spread would forgive -- and pass the gate at honest
    tolerances. The measured figure is about five basis points; the bound leaves room."""
    batch = make_snapshot_batch()
    learner = TorchLearner(penalty_mesh=make_mesh())

    surface = learner.update(None, batch)

    assert rmse_bp(surface, batch) < 15.0
    report = check_surface(surface, make_mesh())
    assert not report.exceeds(GENEROUS_GATE.butterfly, GENEROUS_GATE.calendar)


def test_warm_steps_on_carried_moments_do_not_wreck_a_converged_fit() -> None:
    """The measurement that put the moments in the surface: fresh moments on a converged network
    took a 42 bp fit to 150 bp in ten steps. Carried, five cycles must stay within a basis point."""
    batch = make_snapshot_batch()
    learner = TorchLearner(penalty_mesh=make_mesh(), settings=TorchFitSettings(warm_steps=10))
    surface = learner.update(None, batch)
    converged = rmse_bp(surface, batch)

    for _ in range(5):
        surface = learner.update(surface, batch)

    assert rmse_bp(surface, batch) < converged + 1.0


def test_fine_tuning_follows_a_market_that_moved() -> None:
    """Continuous learning (Design 6.4): after the market re-prices, the warm steps must close on
    it rather than hold the old surface."""
    learner = make_torch_learner(settings=replace(TEST_SETTINGS, warm_steps=20))
    before = make_snapshot_batch()
    moved = make_snapshot_batch(
        make_market_snapshot(log_moneyness=DENSE_MONEYNESS, params=replace(make_params(), a=0.06))
    )
    surface = learner.update(None, before)
    error_at_the_move = rmse_bp(surface, moved)

    for _ in range(20):
        surface = learner.update(surface, moved)

    assert error_at_the_move > 100.0, "the re-priced market must actually be far from the old fit"
    assert rmse_bp(surface, moved) < 0.5 * error_at_the_move


def test_one_cycle_through_the_real_use_case_publishes_and_the_next_chains_the_weights() -> None:
    """The learner behind ``TrainOnSnapshot``, in place of the stub: the first snapshot publishes
    version one, the second fine-tunes it and publishes version two. The wiring test this stage
    can run without a composition root for the producer.

    On the shipped learner, not the session's small one: this is the cycle as it would run in the
    engine, gate included, and the small network's short budget leaves a calendar crossing in the
    extrapolated wings that the gate rightly refuses.
    """
    metrics = RecordingMetrics()
    clock = ManualClock(NOW)
    use_case = TrainOnSnapshot(
        learner=TorchLearner(penalty_mesh=make_mesh()),
        state=TrainingState(),
        buffer=make_buffer(),
        clock=clock,
        metrics=metrics,
        weighting=make_weighting(),
        grid=make_grid_spec(),
        mesh=make_mesh(),
        thresholds=GENEROUS_GATE,
        schedule=make_schedule(),
        rng=np.random.default_rng(20260920),
    )
    first_snapshot = make_market_snapshot(log_moneyness=DENSE_MONEYNESS)
    second_snapshot = make_market_snapshot(
        snapshot_id="BTC-DERIBIT:00000001",
        ts_exchange=NOW + timedelta(seconds=30),
        log_moneyness=DENSE_MONEYNESS,
    )

    first = use_case.handle(first_snapshot)
    clock.advance(30.0)
    second = use_case.handle(second_snapshot)

    assert [type(event) for event in first] == [SurfaceCalibrated]
    assert [type(event) for event in second] == [SurfaceCalibrated]
    assert isinstance(first[0], SurfaceCalibrated) and isinstance(second[0], SurfaceCalibrated)
    assert first[0].surface.producer_id == PRODUCER_ID
    first_meta, second_meta = first[0].surface.producer_meta, second[0].surface.producer_meta
    assert first_meta is not None and second_meta is not None
    assert first_meta["weights_version"] == 1.0
    assert second_meta["weights_version"] == 2.0
    assert math.isfinite(second[0].surface.fit.rmse_vol_bp)


def test_the_test_spec_and_settings_are_smaller_than_the_shipped_ones() -> None:
    """The module docstring of ``torch_builders`` promises a cheaper session; keep it true."""
    assert sum(TEST_SPEC.hidden) < sum(NetworkSpec().hidden)
    assert TEST_SETTINGS.cold_steps < TorchFitSettings().cold_steps
