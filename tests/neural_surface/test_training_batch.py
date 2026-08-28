from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import timedelta

import pytest

from tests.neural_surface.builders import (
    FAR_TENOR,
    FRESH_SAMPLES,
    NAIVE,
    NEAR_TENOR,
    NOW,
    REPLAYED_SAMPLES,
    make_batch,
    make_sample,
)
from tests.support import replace_field

# --- TrainingSample: the coordinate


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_sample_rejects_a_non_finite_log_moneyness(bad: float) -> None:
    with pytest.raises(ValueError, match="log-moneyness must be finite"):
        replace(make_sample(), log_moneyness=bad)


def test_sample_accepts_at_the_money_forward() -> None:
    """``k = 0`` is the most informative point on the surface, and ``not 0.0`` is ``True``."""
    sample = make_sample(log_moneyness=0.0)

    assert sample.log_moneyness == 0.0


def test_sample_accepts_a_negative_log_moneyness() -> None:
    """Every strike below the forward, which is half the smile."""
    sample = make_sample(log_moneyness=-0.45)

    assert sample.log_moneyness == -0.45


# --- TrainingSample: the positive scalars


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_sample_rejects_a_non_positive_tenor(bad: float) -> None:
    with pytest.raises(ValueError, match="tenor in years must be positive"):
        replace(make_sample(), tenor_years=bad)


@pytest.mark.parametrize("bad", [0.0, -0.65, float("nan"), float("inf")])
def test_sample_rejects_a_non_positive_implied_vol(bad: float) -> None:
    with pytest.raises(ValueError, match="implied volatility must be positive"):
        replace(make_sample(), implied_vol=bad)


@pytest.mark.parametrize("field", ["tenor_years", "implied_vol"])
def test_sample_rejects_a_nan_that_slips_past_a_bare_ordering_guard(field: str) -> None:
    """The trap this codebase keeps rediscovering, asserted rather than trusted.

    ``float("nan") <= 0`` is ``False``, so a guard written as ``if value <= 0`` would admit the
    NaN and hand it to the first backward pass, where it poisons every weight at once.
    """
    nan = float("nan")
    assert not nan <= 0

    with pytest.raises(ValueError, match="positive and finite"):
        replace_field(make_sample(), field, nan)


# --- TrainingSample: the weight


@pytest.mark.parametrize("bad", [-0.30, float("nan"), float("inf")])
def test_sample_rejects_an_unusable_weight(bad: float) -> None:
    with pytest.raises(ValueError, match="weight must be non-negative and finite"):
        replace(make_sample(), weight=bad)


def test_sample_accepts_a_zero_weight() -> None:
    """How a flagged quote stays in the batch, and in the buffer's occupancy, with no influence."""
    sample = make_sample(weight=0.0)

    assert sample.weight == 0.0


# --- TrainingSample: the instant


def test_sample_rejects_a_naive_observation_instant() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(make_sample(), ts_observed=NAIVE)


# --- TrainingSample: the derived target


def test_sample_total_variance_is_the_squared_vol_times_the_tenor() -> None:
    sample = make_sample(implied_vol=0.60, tenor_years=0.25)

    assert sample.total_variance == pytest.approx(0.60 * 0.60 * 0.25)


def test_sample_total_variance_really_depends_on_the_tenor() -> None:
    """Guards the assertion above from passing vacuously.

    At the builder's one-month tenor the multiplication is far from a no-op, so a
    ``total_variance`` that forgot the tenor -- or that returned the volatility squared, or the
    volatility itself -- would be caught rather than agreeing by coincidence.
    """
    sample = make_sample(implied_vol=0.65, tenor_years=NEAR_TENOR)

    assert sample.total_variance != sample.implied_vol**2
    assert sample.total_variance != sample.implied_vol
    assert sample.total_variance == pytest.approx(0.65 * 0.65 / 12.0)


def test_sample_total_variance_scales_linearly_in_the_tenor() -> None:
    """The property Design 6.2 trains against: flat vol means variance proportional to ``T``."""
    near = make_sample(implied_vol=0.65, tenor_years=NEAR_TENOR)
    far = make_sample(implied_vol=0.65, tenor_years=FAR_TENOR)

    assert far.total_variance == pytest.approx(3.0 * near.total_variance)


# --- TrainingBatch: identity


@pytest.mark.parametrize("field", ["market_id", "snapshot_id"])
def test_batch_rejects_an_empty_identifier(field: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        replace_field(make_batch(), field, "")


def test_batch_rejects_a_naive_snapshot_instant() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(make_batch(), ts_snapshot=NAIVE)


def test_batch_accepts_a_sample_observed_after_the_snapshot_instant() -> None:
    """Clock skew between a venue and the engine is the ACL's problem, not a domain exception."""
    ahead = make_sample(ts_observed=NOW + timedelta(seconds=2))
    batch = make_batch(samples=(ahead,), n_fresh=1)

    assert batch.samples[0].ts_observed > batch.ts_snapshot


# --- TrainingBatch: the samples


def test_batch_rejects_no_samples() -> None:
    """A gradient step over nothing is a no-op that would still be counted as an update."""
    with pytest.raises(ValueError, match="at least one sample"):
        make_batch(samples=(), n_fresh=0)


def test_batch_rejects_weights_that_sum_to_zero() -> None:
    """Every weight legal on its own, and no point left with any influence on the step."""
    flagged = tuple(replace(sample, weight=0.0) for sample in FRESH_SAMPLES)

    with pytest.raises(ValueError, match="sum to a positive number"):
        make_batch(samples=flagged, n_fresh=2)


def test_batch_weight_sum_is_never_asked_to_catch_a_negative_weight() -> None:
    """The aggregate guard cannot be fooled, because a negative weight never reaches it.

    A bare ``sum(weights) > 0`` passes happily on the poisoned set below -- the assertion proves
    it -- so the guard that actually stands between the loss and a point pulling the optimiser
    *away* from itself is the per-sample one, and it fires at construction.
    """
    poisoned = (0.30, -0.20, 0.25)
    assert sum(poisoned) > 0

    with pytest.raises(ValueError, match="weight must be non-negative"):
        make_sample(weight=poisoned[1])


def test_batch_accepts_a_single_sample() -> None:
    """One point is a real, if uninformative, step; conditioning is the learner's judgement."""
    batch = make_batch(samples=(make_sample(),), n_fresh=1)

    assert len(batch.samples) == 1


def test_batch_accepts_samples_in_no_particular_order() -> None:
    """There is no ordering invariant: the network sees the surface as a point cloud, and the
    replay buffer interleaves tenors on purpose."""
    scattered = (
        make_sample(log_moneyness=0.30, tenor_years=FAR_TENOR),
        make_sample(log_moneyness=-0.05, tenor_years=NEAR_TENOR),
        make_sample(log_moneyness=0.10, tenor_years=FAR_TENOR),
    )

    batch = make_batch(samples=scattered, n_fresh=1)

    assert batch.samples == scattered


def test_batch_accepts_a_repeated_coordinate() -> None:
    """The same strike seen at two instants is two observations, and the buffer holds both."""
    earlier = make_sample(ts_observed=NOW - timedelta(minutes=4))
    later = make_sample(ts_observed=NOW)

    batch = make_batch(samples=(later, earlier), n_fresh=1)

    assert batch.samples[0].log_moneyness == batch.samples[1].log_moneyness
    assert batch.samples[0].tenor_years == batch.samples[1].tenor_years


# --- TrainingBatch: the fresh/replayed split


@pytest.mark.parametrize("bad", [-1, 5])
def test_batch_rejects_a_fresh_count_outside_the_samples(bad: int) -> None:
    with pytest.raises(ValueError, match="between 0 and 4"):
        make_batch(n_fresh=bad)


def test_batch_accepts_no_fresh_samples() -> None:
    """The scheduled restart of Design 6.4: the whole buffer, no new quotes at all.

    Zero must never be tested for truthiness, which is exactly what this batch would trip.
    """
    batch = make_batch(samples=REPLAYED_SAMPLES, n_fresh=0)

    assert batch.n_fresh == 0
    assert batch.fresh == ()
    assert batch.replayed == REPLAYED_SAMPLES


def test_batch_accepts_every_sample_being_fresh() -> None:
    """A cold start whose buffer has nothing to contribute yet."""
    batch = make_batch(samples=FRESH_SAMPLES, n_fresh=len(FRESH_SAMPLES))

    assert batch.fresh == FRESH_SAMPLES
    assert batch.replayed == ()


def test_batch_splits_the_samples_at_the_fresh_count() -> None:
    batch = make_batch()

    assert batch.fresh == FRESH_SAMPLES
    assert batch.replayed == REPLAYED_SAMPLES


def test_batch_fresh_and_replayed_partition_the_samples() -> None:
    """No gap and no overlap: they are two halves of one slice at one index."""
    batch = make_batch()

    assert batch.fresh + batch.replayed == batch.samples
    assert len(batch.fresh) + len(batch.replayed) == len(batch.samples)
    assert len(batch.fresh) > 0 and len(batch.replayed) > 0


# --- Architecture: both are frozen, slotted value objects


@pytest.mark.parametrize(
    ("value", "field"),
    [
        (make_sample(), "implied_vol"),
        (make_batch(), "market_id"),
    ],
)
def test_training_values_are_frozen_and_slotted(value: object, field: str) -> None:
    assert not hasattr(value, "__dict__")

    with pytest.raises(FrozenInstanceError):
        setattr(value, field, None)


def test_training_batch_is_not_a_contract() -> None:
    """Rule 3: the domain does not know ``contracts/``, so nothing here serialises itself."""
    batch = make_batch()

    assert not hasattr(batch, "to_dict")
    assert not hasattr(batch, "from_dict")
    assert not hasattr(batch, "schema_version")
