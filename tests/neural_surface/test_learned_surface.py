from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.neural_surface.builders import (
    FAR_TENOR,
    MESH_K,
    NEAR_TENOR,
    CallableSurface,
    CountingSurface,
    FlatVolSurface,
    TransposedSurface,
)
from volengine.neural_surface.domain.errors import SurfaceEvaluationError
from volengine.neural_surface.domain.learned_surface import (
    LearnedSurface,
    evaluate_total_variance,
    implied_vol_grid,
)

K = np.array(MESH_K, dtype=np.float64)
TENORS = np.array([NEAR_TENOR, FAR_TENOR], dtype=np.float64)
"""Twenty-five moneyness nodes at two tenors: a mesh whose two axes have different lengths.

Chosen that way on purpose. Every claim below about the layout of the result would be vacuous on
a square mesh, where a transposed answer has the right shape and the wrong meaning, and the
transposition is the mistake a real adapter actually makes.
"""

VOL = 0.65
"""The flat surface's volatility, an ordinary at-the-money crypto number."""


def _nan_grid(k_mesh: NDArray[np.float64], tenor_mesh: NDArray[np.float64]) -> NDArray[np.float64]:
    """A model whose weights all went to NaN on one bad gradient step."""
    return np.full(k_mesh.shape, np.nan)


# --- The port itself


def test_a_double_satisfies_the_protocol_where_it_is_annotated() -> None:
    """The structural conformance check that actually runs: mypy sees this assignment.

    ``tests/`` is type-checked, so a builder double that stopped matching the Protocol -- a
    renamed member, a return type that drifted -- fails the type check here rather than in an
    adapter nobody has written yet. That is the argument for a Protocol over an ABC made
    concrete: nothing in ``builders.py`` inherits from anything.
    """
    surface: LearnedSurface = FlatVolSurface(vol=VOL)

    assert surface.version == 1
    assert evaluate_total_variance(surface, K, TENORS).shape == (len(TENORS), len(K))


def test_the_protocol_refuses_an_isinstance_check() -> None:
    """Not runtime-checkable, deliberately: name matching would pass the one bug that matters.

    ``isinstance`` against a runtime-checkable Protocol compares member names only, so
    ``TransposedSurface`` -- which this module exists to catch -- would be certified a valid
    surface by it. Keeping the decorator off makes the useless check impossible to write instead
    of merely inadvisable.
    """
    with pytest.raises(TypeError, match="runtime_checkable"):
        isinstance(FlatVolSurface(), LearnedSurface)  # type: ignore[misc]


def test_a_diverged_model_is_not_reported_as_a_caller_bug() -> None:
    """The two failure kinds must stay distinguishable by ``except``, not only by message.

    A malformed argument and a model that has stopped describing a market lead to different
    places: one is fixed in code, the other republishes the last good surface and increments a
    counter. A ``SurfaceEvaluationError`` inheriting from ``ValueError`` would let an empty array
    trip the operational path, and a bare ``except ValueError`` around an evaluation would
    swallow the divergence.
    """
    assert not issubclass(SurfaceEvaluationError, ValueError)


# --- evaluate_total_variance: the happy path and the layout


def test_evaluating_a_flat_surface_gives_the_volatility_squared_times_the_tenor() -> None:
    """w = sigma^2 * T, the one case whose arithmetic a reader can do in the margin.

    Pinned twice: exactly against the definition, and approximately against an independently
    computed magnitude, so a rewrite that agrees with its own algebra still has to land on 0.0352
    at one month and 0.1056 at three.
    """
    w = evaluate_total_variance(FlatVolSurface(vol=VOL), K, TENORS)

    assert np.all(w[0] == NEAR_TENOR * VOL * VOL)
    assert np.all(w[1] == FAR_TENOR * VOL * VOL)
    assert float(w[0][0]) == pytest.approx(0.035208, abs=1e-6)
    assert float(w[1][0]) == pytest.approx(0.105625, abs=1e-6)


def test_the_grid_has_one_row_per_tenor_and_one_column_per_moneyness_node() -> None:
    """The shape the whole context is written against, asserted where it is promised."""
    assert evaluate_total_variance(FlatVolSurface(), K, TENORS).shape == (2, 25)


def test_the_grid_is_tenor_major() -> None:
    """Row i is the smile at tenors[i]; column j is the term structure at k[j].

    A surface that varies along both axes is what makes the claim testable at all: on a flat
    surface every row is identical and any indexing convention would pass.
    """
    surface = CallableSurface(fn=lambda k_mesh, tenor_mesh: tenor_mesh * (1.0 + k_mesh))

    # Guards the point: a transposed answer cannot pass by accident, because the two axes have
    # different lengths and the shapes would not even match.
    assert len(K) != len(TENORS)

    w = evaluate_total_variance(surface, K, TENORS)
    assert float(w[0][0]) == pytest.approx(NEAR_TENOR * (1.0 + float(K[0])))
    assert float(w[1][-1]) == pytest.approx(FAR_TENOR * (1.0 + float(K[-1])))


def test_the_surface_is_asked_for_the_mesh_it_was_given_and_asked_once() -> None:
    """No resampling, no second pass: the caller chose the mesh and gets it evaluated.

    A function that quietly evaluated on a denser mesh of its own would make the gate's reported
    counts describe points nobody asked about, and evaluating twice would double the cost of the
    most expensive call in the cycle.
    """
    surface = CountingSurface(vol=VOL)

    evaluate_total_variance(surface, K, TENORS)

    assert surface.calls == [(len(TENORS), len(K))]


# --- evaluate_total_variance: a model that has stopped being a surface


def test_a_transposed_surface_is_refused() -> None:
    """The adapter mistake of the first day: right numbers, wrong axes.

    It would not raise anywhere downstream. It would divide each smile by the wrong expiry in
    ``implied_vol_grid`` and publish a term structure labelled as a smile, which is why the shape
    is checked here rather than trusted to the type annotation.
    """
    with pytest.raises(SurfaceEvaluationError, match="tenor-major"):
        evaluate_total_variance(TransposedSurface(vol=VOL), K, TENORS)


def test_a_nan_surface_is_refused_although_every_ordering_test_calls_it_clean() -> None:
    """The reason this is an exception and not an entry in the arbitrage report.

    The first two assertions are the whole argument: a grid of NaN is not negative anywhere, and
    ``max(0, -min(w))`` -- the shape every violation measure in this engine has -- scores it
    exactly zero. Left to the report, a model whose weights all went to NaN would be published as
    the cleanest surface of the session.
    """
    surface = CallableSurface(fn=_nan_grid)
    raw = surface.total_variance(K, TENORS)

    assert not bool(np.any(raw < 0.0))
    assert max(0.0, -float(np.min(raw))) == 0.0

    with pytest.raises(SurfaceEvaluationError, match="finite and strictly positive"):
        evaluate_total_variance(surface, K, TENORS)


def test_an_infinite_total_variance_is_refused() -> None:
    """An infinity passes ``w <= 0`` too, and then divides its way to a plausible-looking grid."""
    surface = CallableSurface(fn=lambda k_mesh, tenor_mesh: np.full(k_mesh.shape, np.inf))

    with pytest.raises(SurfaceEvaluationError, match="finite and strictly positive"):
        evaluate_total_variance(surface, K, TENORS)


def test_a_vanishing_total_variance_is_refused() -> None:
    """Zero is the degenerate limit, not a cheap market: the distribution has collapsed.

    Nothing negative happens here either -- zero reads clean on every ordering test -- but there
    is no volatility to recover from it, since ``sqrt(0 / T)`` is a surface quoting no
    uncertainty at any strike.
    """
    surface = CallableSurface(fn=lambda k_mesh, tenor_mesh: np.zeros(k_mesh.shape))

    with pytest.raises(SurfaceEvaluationError, match="finite and strictly positive"):
        evaluate_total_variance(surface, K, TENORS)


def test_a_negative_total_variance_anywhere_is_refused() -> None:
    """One node is enough. A negative variance has no reading, so there is nothing to report.

    The surface is healthy everywhere except a single wing node, which is exactly how a diverged
    network fails first, and a check that only looked at the mean or at the money would miss it.
    """
    surface = CallableSurface(
        fn=lambda k_mesh, tenor_mesh: np.where(k_mesh < -0.55, -0.01, tenor_mesh * VOL * VOL)
    )

    with pytest.raises(SurfaceEvaluationError, match="finite and strictly positive"):
        evaluate_total_variance(surface, K, TENORS)


# --- evaluate_total_variance: a malformed question, which is not the model's fault


def test_an_empty_moneyness_axis_is_the_callers_bug() -> None:
    """Nothing was looked at is not nothing was found, and only one of them scores clean."""
    with pytest.raises(ValueError, match="at least one point"):
        evaluate_total_variance(FlatVolSurface(), np.array([]), TENORS)


def test_an_empty_tenor_axis_is_the_callers_bug() -> None:
    with pytest.raises(ValueError, match="at least one point"):
        evaluate_total_variance(FlatVolSurface(), K, np.array([]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_moneyness_node_is_the_callers_bug(bad: float) -> None:
    """It poisons its own column and every reduction over it, and blames the model for it."""
    with pytest.raises(ValueError, match="finite at every point"):
        evaluate_total_variance(FlatVolSurface(), np.array([-0.1, bad, 0.1]), TENORS)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_tenor_is_the_callers_bug(bad: float) -> None:
    """Finiteness is tested before the sign: ``nan <= 0`` is False, so a NaN tenor would
    otherwise be accepted as a perfectly ordinary expiry."""
    with pytest.raises(ValueError, match="finite at every point"):
        evaluate_total_variance(FlatVolSurface(), K, np.array([NEAR_TENOR, bad]))


@pytest.mark.parametrize("bad", [0.0, -NEAR_TENOR])
def test_a_tenor_that_is_not_strictly_positive_is_the_callers_bug(bad: float) -> None:
    """Zero is the expiry itself and negative is the past; neither has a total variance."""
    with pytest.raises(ValueError, match="strictly positive at every point"):
        evaluate_total_variance(FlatVolSurface(), K, np.array([bad, FAR_TENOR]))


def test_a_two_dimensional_axis_is_the_callers_bug() -> None:
    """``len`` of a 2-D array is its first dimension, silently redefining the promised shape."""
    with pytest.raises(ValueError, match="one-dimensional"):
        evaluate_total_variance(FlatVolSurface(), K.reshape(5, 5), TENORS)


def test_a_malformed_axis_never_reaches_the_model() -> None:
    """Validation precedes evaluation, so the expensive call is not made to be thrown away.

    It also keeps the attribution honest: a model that was never asked cannot be the one at
    fault, and this is what stops a caller bug from showing up in the refusal-rate metric that
    Design 6.5 reads as the neural producer's quality signal.
    """
    surface = CountingSurface()

    with pytest.raises(ValueError, match="at least one point"):
        evaluate_total_variance(surface, np.array([]), TENORS)

    assert surface.calls == []


# --- implied_vol_grid


def test_implied_vol_grid_inverts_the_total_variance_of_a_flat_surface() -> None:
    """sqrt(sigma^2 * T / T) is sigma at every node, which is the whole conversion.

    A flat surface is the only one whose answer is known in closed form at every point, so it is
    where the arithmetic gets pinned rather than merely sanity-checked.
    """
    vols = implied_vol_grid(FlatVolSurface(vol=VOL), K, TENORS)

    assert vols.shape == (len(TENORS), len(K))
    assert vols == pytest.approx(np.full((len(TENORS), len(K)), VOL))


def test_implied_vol_grid_divides_each_row_by_its_own_tenor() -> None:
    """The pairing between a row and its expiry, on a surface where the two rows differ.

    A flat surface would hide a swapped tenor: both rows would still come back at the same
    volatility. Here the near month is a 50% market and the far one an 80% market, so dividing a
    row by the other row's tenor produces neither number.
    """
    surface = CallableSurface(
        fn=lambda k_mesh, tenor_mesh: tenor_mesh * np.where(tenor_mesh < 0.1, 0.25, 0.64)
    )

    vols = implied_vol_grid(surface, K, TENORS)

    assert vols[0] == pytest.approx(np.full(len(K), 0.5))
    assert vols[1] == pytest.approx(np.full(len(K), 0.8))


def test_implied_vol_grid_refuses_a_diverged_surface_through_the_same_door() -> None:
    """It delegates to ``evaluate_total_variance``, so the guarantee is stated once.

    A second, looser validation path into the square root is exactly how the two consumers of a
    surface would come to disagree about what "usable" means.
    """
    with pytest.raises(SurfaceEvaluationError, match="finite and strictly positive"):
        implied_vol_grid(CallableSurface(fn=_nan_grid), K, TENORS)


def test_implied_vol_grid_refuses_a_usable_variance_that_overflows_on_division() -> None:
    """The one failure this step adds on its own: a legal tenor small enough to overflow.

    The total variance is finite and strictly positive, so ``evaluate_total_variance`` has
    nothing to object to; the tenor is finite and strictly positive, so neither has the axis
    guard. An infinite volatility is still not publishable, and it would reach a contract whose
    grid promises every vol is finite.
    """
    tiny = np.array([1e-320], dtype=np.float64)
    surface = CallableSurface(fn=lambda k_mesh, tenor_mesh: np.ones(k_mesh.shape))

    # Guards the point: the variance itself is perfectly usable, so the refusal below can only
    # come from the division.
    assert evaluate_total_variance(surface, K, tiny).shape == (1, len(K))

    with (
        np.errstate(over="ignore", divide="ignore"),
        pytest.raises(SurfaceEvaluationError, match="implied volatilities must be finite"),
    ):
        implied_vol_grid(surface, K, tiny)
