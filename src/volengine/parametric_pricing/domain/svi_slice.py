"""SVI in its own coordinates: log-forward-moneyness against total variance.

The whole context works in ``(k, w)`` and never in ``(strike, vol)``.

``k = ln(K / F)`` puts every market on the same axis -- 5% above the forward is the same ``k``
on BTC at 60,000 and on SPX at 5,000 -- and it is the coordinate in which the smile is roughly
symmetric, which is what makes a five-parameter form enough.

``w = sigma^2 * T`` is *total* variance: the variance accumulated between now and expiry,
rather than the annualised rate. Two consequences follow, and both shape the types below.
First, no-arbitrage is stated in total variance (Durrleman's butterfly condition on one slice,
monotonicity in ``T`` for the calendar one), so working in ``w`` means the constraints are
expressible at all. Second, **the tenor is already inside ``w``**, which is exactly why
``SVIParams`` carries no tenor field: the five numbers describe a curve ``w(k)``, and turning
that curve into a volatility needs a ``T`` that the caller supplies. Storing a tenor next to
the parameters would invite two sources of truth for it, and ``SVISlice`` -- which does own a
tenor, because a slice is attached to an expiry -- is the one place that pairing lives.

The raw SVI form is

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

with ``a`` the overall level, ``b`` the wing steepness, ``rho`` the skew (the tilt between the
two wings), ``m`` the horizontal shift of the smile's minimum and ``sigma`` the curvature at
the bottom -- ``sigma`` here is a *shape* parameter of the hyperbola, not a volatility. Far
from ``m`` the square root becomes ``|k - m|`` and the curve is asymptotically linear: slope
``b * (1 + rho)`` on the right, ``b * (rho - 1)`` on the left. Those two slopes are what a
sign error in ``rho`` corrupts, and they are the reason the wings are tested numerically.

**That form itself lives in ``shared_kernel/domain/svi.py``, not here** (ADR-026). What this
module owns is the *type*: which five numbers make a fitted slice, which invariants a fit's
output must satisfy and which it must deliberately not, and how the whole thing maps into the
unconstrained space an optimiser searches. Market Data's generator writes an ``SVIParamsSpec``
with the same five fields and stricter invariants, and it must -- a specification of what to
generate is not the output of a fit -- but the two are not allowed to disagree about the
mathematics, because that context's surface is what this context's calibrator is measured
against, and a sign slip on both sides cancels.

Nothing here imports jax, scipy or ``contracts/``: what lives in this module is what SVI
*means*, and none of it depends on which library computed the numbers (see the package
docstring). numpy is allowed, and is used for the vectorised evaluation only -- which is the
one thing that cannot move to the kernel, since rule 1 confines it to the standard library.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from typing import Final, overload

import numpy as np
from numpy.typing import NDArray

from volengine.shared_kernel.domain import svi
from volengine.shared_kernel.domain.instants import require_aware

SOFTPLUS_LINEAR_ABOVE: Final[float] = 20.0
"""Above this, ``softplus(x)`` and ``x`` agree to within a rounding error of each other.

``softplus(20) - 20`` is about ``2e-9`` in relative terms, and the inverse can be evaluated in
its linear form from here on without ever calling ``expm1`` on a large argument.
"""

RHO_LIMIT: Final[float] = math.nextafter(1.0, 0.0)
"""Largest double strictly below 1, and the clamp applied to ``tanh`` in ``from_free``.

``math.tanh`` returns exactly ``1.0`` for arguments beyond roughly 19.06, so an optimiser that
pushes ``rho_raw`` into that region would otherwise build an ``SVIParams`` with ``|rho| == 1``
and be rejected by its own reparameterisation. Clamping here is not hiding an error: at that
point ``tanh`` has already lost every bit of information distinguishing the value from 1, and
this is the nearest representable number that still satisfies the invariant.
"""

MIN_POSITIVE: Final[float] = sys.float_info.min
"""Smallest positive normal double, the floor ``to_free`` uses for ``b`` and ``sigma``.

The free space is the *open* half-line: ``softplus`` never returns 0, so ``b = 0`` -- a legal,
flat slice -- has no finite preimage and the honest inverse would be ``-inf``. A ``FreeParams``
holding an infinity is useless to any optimiser, which has to be able to step away from it, so
the mapping floors the value instead. The round trip through zero is therefore not exact; every
strictly positive ``b`` round-trips normally.
"""


def _softplus(x: float) -> float:
    """``log(1 + exp(x))``, evaluated so that neither tail overflows.

    The textbook form overflows for ``x`` above ~709 -- ``exp(x)`` is infinite long before
    ``log`` of it would be. The identity ``softplus(x) = x + softplus(-x)`` moves the
    exponential to the safe side, so only ``exp`` of a non-positive number is ever taken, which
    underflows harmlessly to 0 instead of overflowing to infinity.
    """
    if x > 0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def _inverse_softplus(y: float) -> float:
    """``log(expm1(y))``, evaluated so that neither tail breaks.

    Two failure modes of the naive expression, one at each end:

    * Large ``y``. ``expm1(y)`` overflows above ~709.78 and the inverse returns infinity for a
      perfectly ordinary parameter. Above :data:`SOFTPLUS_LINEAR_ABOVE` the equivalent form
      ``y + log1p(-exp(-y))`` is used instead: the exponential is of a *negative* number, so it
      cannot overflow, and the correction term is the exact difference from the linear regime.
    * ``y = 0``. ``expm1(0)`` is 0 and ``log(0)`` raises ``ValueError`` from inside the
      arithmetic. The caller floors the input at :data:`MIN_POSITIVE` before getting here, which
      is where that decision is documented; this function assumes a positive argument.
    """
    if y > SOFTPLUS_LINEAR_ABOVE:
        return y + math.log1p(-math.exp(-y))
    return math.log(math.expm1(y))


@dataclass(frozen=True, slots=True)
class FreeParams:
    """The same slice seen from the unconstrained space the optimiser actually searches in
    (Design 5.4).

    SVI's parameters are constrained -- ``b`` and ``sigma`` positive, ``|rho| < 1`` -- and a
    constrained optimiser is a different, slower and far more fragile animal than an
    unconstrained one. Reparameterising through ``softplus`` and ``tanh`` makes the constraints
    structural: every point of R^5 maps to a set of parameters that already satisfies them, so
    the optimiser can take any step it likes and never propose ``rho = 1.4``. That is what
    ``domain/errors.py`` means when it says a bad ``rho`` is a construction bug rather than a
    market condition -- the mapping exists precisely so that no optimiser can produce one.

    Three of the four constraints are structural this way. The fourth, non-negative minimum
    total variance, involves ``a`` and is *not*, so ``SVIParams.from_free`` can still raise --
    an optimiser exploring a deeply negative ``a`` is proposing a negative variance, and there
    is no reparameterisation that would make that a surface.

    The fields are ``a`` and ``m`` unchanged -- both are genuinely unconstrained, ``a`` is a
    level and ``m`` a horizontal shift -- plus a raw counterpart for each constrained one. The
    ``_raw`` suffix is there so that no call site can confuse ``sigma_raw = -3.0`` with a
    ``sigma`` of ``-3.0``, which would be nonsense.
    """

    a: float
    """Level of the total-variance curve, unconstrained. Finite."""

    b_raw: float
    """Pre-image of ``b = softplus(b_raw) >= 0``. Finite."""

    rho_raw: float
    """Pre-image of ``rho = tanh(rho_raw)``, hence ``|rho| < 1``. Finite."""

    m: float
    """Horizontal shift of the smile's minimum, unconstrained. Finite."""

    sigma_raw: float
    """Pre-image of ``sigma = softplus(sigma_raw) > 0``. Finite."""

    def __post_init__(self) -> None:
        # Only finiteness is checkable here. Anything else would be a constraint on the free
        # space, and the whole point of the free space is that it has none: the optimiser must
        # be allowed to step anywhere, and the mapping is what keeps the result admissible.
        for name, value in (
            ("a", self.a),
            ("b_raw", self.b_raw),
            ("rho_raw", self.rho_raw),
            ("m", self.m),
            ("sigma_raw", self.sigma_raw),
        ):
            if not math.isfinite(value):
                raise ValueError(f"The free parameter {name} must be finite, got {value}")


@dataclass(frozen=True, slots=True)
class SVIParams:
    """The five raw SVI parameters of one slice. No tenor: it is already inside ``w``.

    **The invariants below say "this is a surface", not "this surface is arbitrage-free".**
    That split is the most important decision in this module. A constructor guarantees only
    what makes the object meaningful at all -- finite numbers, a non-negative variance, a
    ``rho`` inside its range -- and deliberately stops there. The butterfly (Durrleman)
    condition is *not* enforced, even though it is the condition an admissible slice must
    ultimately satisfy, because enforcing it here would make the calibrator impossible to
    write: least-squares walks through intermediate iterates, many of them arbitrageable, and a
    constructor that refused them would refuse the optimiser its own search path. Worse, it
    would make the violation unmeasurable -- you cannot report how far a fit is from
    admissibility if you cannot hold the object that violates it.

    So the responsibilities are split exactly as they are elsewhere in this project: this value
    object guarantees *this is a surface*, and ``durrleman.py`` measures whether it is a
    *healthy* one, as a number the use case compares against a configured threshold. It is the
    same division of labour Market Data already applies -- ingestion flags, the calibrator
    weights or excludes -- one layer further down: the value object admits, the metric judges.
    """

    a: float
    """Vertical level of the curve, in total-variance units. Finite.

    Not required to be non-negative on its own: with ``b > 0`` the square-root term is strictly
    positive everywhere, so a slightly negative ``a`` can still leave every ``w(k)`` positive.
    The condition that matters is on the minimum of the curve, checked below.
    """

    b: float
    """Wing steepness -- half the difference of the two asymptotic slopes. Non-negative, finite.

    The right asymptote has slope ``b * (1 + rho)`` and the left one ``b * (rho - 1)``, so half
    their difference is ``b`` exactly, whatever the skew. Equivalently, it is how wide the two
    wings open together while ``rho`` decides how that opening is shared between them.

    Zero is legal and is the degenerate flat slice ``w(k) = a``, a constant total variance
    across strikes. That is what a market with no smile at all looks like, it is a perfectly
    representable state, and it is also the analytic case a calibrator's first iterate often
    starts near.
    """

    rho: float
    """Skew, the tilt between the two wings. Strictly inside ``(-1, 1)``, finite.

    Negative ``rho`` lifts the left wing relative to the right one, which is the usual equity
    and crypto shape: downside strikes trade at a higher implied vol. At ``|rho| = 1`` one wing
    is flat and the curve stops being a proper hyperbola, so the bound is strict.
    """

    m: float
    """Horizontal position of the smile's minimum, in log-forward-moneyness. Finite.

    Usually small and negative for a skewed market. Zero means the minimum sits at the money
    forward -- a legitimate value, so this field must never be tested for truthiness.
    """

    sigma: float
    """Curvature of the smile at its bottom. Strictly positive and finite.

    A shape parameter of the hyperbola, **not** a volatility, despite the name the literature
    settled on. Small ``sigma`` is a sharp V-shaped smile, large ``sigma`` a wide rounded one.
    Zero would collapse the square root to ``|k - m|`` and leave a kink -- a curve with no
    second derivative at the minimum, which the butterfly condition is stated in terms of -- so
    it is excluded.
    """

    def __post_init__(self) -> None:
        for name, value in (
            ("a", self.a),
            ("b", self.b),
            ("rho", self.rho),
            ("m", self.m),
            ("sigma", self.sigma),
        ):
            if not math.isfinite(value):
                raise ValueError(f"The SVI parameter {name} must be finite, got {value}")

        # Every guard below joins the *bad* conditions with `or`, after an explicit finiteness
        # check. A NaN compares False against everything, so `nan <= 0` is False and a NaN
        # would sail through an ordering test that looked correct.
        if self.b < 0:
            raise ValueError(f"The SVI parameter b must be non-negative, got {self.b}")
        if abs(self.rho) >= 1:
            raise ValueError(f"The SVI parameter rho must be inside (-1, 1), got {self.rho}")
        if self.sigma <= 0:
            raise ValueError(f"The SVI parameter sigma must be positive, got {self.sigma}")

        if self.min_total_variance < 0:
            raise ValueError(
                "The minimum total variance of the slice must be non-negative, got "
                f"{self.min_total_variance}"
            )

    @property
    def min_total_variance(self) -> float:
        """Lowest value the curve attains, ``a + b * sigma * sqrt(1 - rho^2)``.

        Closed form rather than a search: differentiating ``w`` and solving gives a single
        interior minimum, and substituting it back leaves this expression, which the shared
        kernel derives in full. It is the quantity the constructor checks, because a curve
        dipping below zero somewhere is claiming a negative variance at that strike -- not an
        arbitrage to be measured and reported, but an object that is not a volatility surface
        at all.

        Called during ``__post_init__``, **after** the guard on ``rho``, so the kernel's own
        domain check on ``1 - rho^2`` can never be the one that fires: a bad ``rho`` is reported
        by this class, in this class's words.
        """
        return svi.min_total_variance(a=self.a, b=self.b, rho=self.rho, sigma=self.sigma)

    @overload
    def total_variance(self, k: float) -> float: ...

    @overload
    def total_variance(self, k: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def total_variance(self, k: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        """``w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))``.

        Accepts one log-moneyness or a whole grid of them. The two overloads exist so that the
        return type follows the argument type instead of being a union every caller has to
        narrow: a loss function feeds an array and wants an array back, while a test or a single
        quote feeds a float and wants a float.

        **The closed form itself lives in the shared kernel** (ADR-026), and the scalar branch
        is nothing but a call into it. The array branch cannot be: rule 1 keeps numpy out of the
        kernel, and evaluating a dense grid one Python call at a time is not an option inside an
        optimiser's loss. So what is written here is the *elementwise image* of
        ``shared_kernel.domain.svi.total_variance`` and nothing else -- same operations, same
        order, therefore the same double to the last bit --  and
        ``test_total_variance_agrees_between_scalar_and_array_input`` is what keeps it that way,
        since the scalar it is compared against is now the kernel's own answer.

        The result is guaranteed non-negative by the constructor's minimum-variance invariant,
        which is what lets ``implied_vol`` take a square root without a guard.
        """
        if not isinstance(k, np.ndarray):
            # ``float`` because "scalar" reaches this branch as a ``np.float64`` often enough --
            # an element pulled out of a grid -- and a numpy scalar escaping into code that
            # asked for a builtin is only ever noticed by whatever serialises it.
            return float(
                svi.total_variance(k, a=self.a, b=self.b, rho=self.rho, m=self.m, sigma=self.sigma)
            )
        centred: NDArray[np.float64] = k - self.m
        root: NDArray[np.float64] = np.sqrt(centred * centred + self.sigma * self.sigma)
        w: NDArray[np.float64] = self.a + self.b * (self.rho * centred + root)
        return w

    @overload
    def implied_vol(self, k: float, tenor_years: float) -> float: ...

    @overload
    def implied_vol(self, k: NDArray[np.float64], tenor_years: float) -> NDArray[np.float64]: ...

    def implied_vol(
        self, k: float | NDArray[np.float64], tenor_years: float
    ) -> float | NDArray[np.float64]:
        """Annualised Black implied volatility, ``sqrt(w(k) / T)``.

        The tenor is an argument rather than a field for the reason the module docstring gives:
        these five numbers describe a total-variance curve, and only a caller who knows which
        expiry it belongs to can annualise it. ``SVISlice`` is the type that owns that pairing.

        No guard on the square root is needed -- the constructor already rules out a negative
        ``w`` -- but ``tenor_years`` comes from outside and does need one: dividing by zero
        would return an infinite vol, and dividing by a negative one a NaN, both of which
        propagate silently into a fit.

        Raises:
            ValueError: If ``tenor_years`` is not positive and finite.
        """
        if not math.isfinite(tenor_years) or tenor_years <= 0:
            raise ValueError(f"The tenor in years must be positive and finite, got {tenor_years}")
        if isinstance(k, np.ndarray):
            return np.sqrt(self.total_variance(k) / tenor_years)
        return math.sqrt(self.total_variance(k) / tenor_years)

    def to_free(self) -> FreeParams:
        """Map into the unconstrained space, inverting ``softplus`` and ``tanh``.

        Used to seed the optimiser from a previous calibration -- the warm start of Design 5.6
        stores ``SVIParams`` and hands the optimiser its free image.

        Both inverses are the numerically careful forms rather than the textbook ones; see
        :func:`_inverse_softplus` for why ``log(expm1(x))`` alone is not good enough. The one
        place the round trip is not exact is ``b = 0``, floored at :data:`MIN_POSITIVE` because
        ``softplus`` has no zero in its image; that floor is documented on the constant.

        ``math.atanh`` is exact enough here without special casing: ``|rho| < 1`` is already an
        invariant, so its argument is never at the singularity. The saturation risk lives in the
        other direction -- ``tanh`` of a large raw value collapsing onto exactly 1 -- and is
        handled by the clamp in ``from_free``, which is also what makes an extreme ``rho`` such
        as ``0.999...`` survive the round trip instead of coming back as an invalid ``1.0``.
        """
        return FreeParams(
            a=self.a,
            b_raw=_inverse_softplus(max(self.b, MIN_POSITIVE)),
            rho_raw=math.atanh(self.rho),
            m=self.m,
            sigma_raw=_inverse_softplus(max(self.sigma, MIN_POSITIVE)),
        )

    @classmethod
    def from_free(cls, free: FreeParams) -> SVIParams:
        """Map back from the unconstrained space, applying ``softplus`` and ``tanh``.

        This is where an optimiser's iterate becomes a slice again, so it runs on every step and
        every constraint it can enforce is a constraint the optimiser never has to know about.

        ``tanh`` is clamped to :data:`RHO_LIMIT` and ``softplus`` floored at
        :data:`MIN_POSITIVE` for ``sigma``: both functions saturate at the closed end of their
        range in floating point, and an unclamped result would hand the constructor a ``rho`` of
        exactly 1 or a ``sigma`` of exactly 0 for an input that is merely large. ``b`` needs no
        floor, since ``b = 0`` is a legal flat slice.

        Raises:
            ValueError: If the resulting parameters have a negative minimum total variance. That
                is the one SVI constraint the reparameterisation cannot make structural, because
                it couples ``a`` -- which is unconstrained by construction -- with the other
                four. An optimiser proposing it is proposing a negative variance.
        """
        return cls(
            a=free.a,
            b=_softplus(free.b_raw),
            rho=max(-RHO_LIMIT, min(RHO_LIMIT, math.tanh(free.rho_raw))),
            m=free.m,
            sigma=max(_softplus(free.sigma_raw), MIN_POSITIVE),
        )


@dataclass(frozen=True, slots=True)
class SVISlice:
    """One fitted expiry: the parameters, the tenor that annualises them, and where they hold.

    The parameters alone are a curve on the whole real line, which is a lie about what was
    calibrated. A chain quotes a finite band of strikes, and outside it the fit is extrapolation
    of a functional form, not evidence. Carrying ``k_min`` and ``k_max`` alongside the
    parameters keeps that distinction available to whoever consumes the slice -- the grid
    builder, the risk engine asking for a far-wing vol -- instead of losing it at the moment the
    fit ends, which is the only moment it is known.

    The band is *not* enforced by ``total_variance``: SVI is defined everywhere and clamping
    evaluation to the band would replace an honest extrapolation with a silent one. It is
    published so the decision can be made where it belongs.
    """

    expiry: datetime
    """Exact expiry instant, timezone-aware.

    Aware because it is subtracted from other timestamps -- ordering slices, matching an expiry
    across snapshots -- and mixing a naive datetime into that raises ``TypeError`` far from
    here. Validated with the shared kernel's ``require_aware`` rather than a private copy of the
    same two lines.
    """

    tenor_years: float
    """Year fraction to expiry under the market's day count. Positive and finite.

    Already convention-dependent work done upstream: the calibrator receives a homogeneous
    number and never learns which market it came from. Kept next to the parameters because
    annualising ``w`` needs it and ``SVIParams`` deliberately does not carry it.
    """

    params: SVIParams
    """The five raw SVI parameters for this expiry, in total-variance space."""

    k_min: float
    """Lowest log-forward-moneyness backed by a real quote. Finite, strictly below ``k_max``."""

    k_max: float
    """Highest log-forward-moneyness backed by a real quote. Finite, strictly above ``k_min``.

    Strictly above, not merely different: a degenerate band of a single point would describe a
    slice fitted on one strike, which is not a slice at all -- five parameters through one point
    is not a fit, and the surrounding code would happily interpolate inside an empty interval.
    """

    def __post_init__(self) -> None:
        require_aware(self.expiry, "expiry")
        if not math.isfinite(self.tenor_years) or self.tenor_years <= 0:
            raise ValueError(
                f"The tenor in years must be positive and finite, got {self.tenor_years}"
            )
        if not math.isfinite(self.k_min) or not math.isfinite(self.k_max):
            raise ValueError(
                f"The validity domain must be finite, got ({self.k_min}, {self.k_max})"
            )
        if self.k_min >= self.k_max:
            raise ValueError(
                f"The validity domain must be non-empty, got k_min={self.k_min} "
                f"not below k_max={self.k_max}"
            )


@dataclass(frozen=True, slots=True)
class SVISurface:
    """Every fitted slice of one market at one instant, ordered by tenor.

    **No calendar-arbitrage check here, on purpose (ADR-008).** Total variance at a fixed ``k``
    must be non-decreasing in ``T`` -- accumulated uncertainty cannot shrink -- and raw SVI
    fitted one slice at a time gives no guarantee of it, because nothing couples the expiries
    during the fit. That risk is *accepted*: the alternative, SSVI, ties every tenor into one
    global parameterisation and changes the shape of the whole calibration cycle, which ADR-008
    defers to its own milestone.

    Accepted does not mean ignored. ``durrleman.calendar_violation`` measures the crossing
    between adjacent slices and it is reported as a metric, exactly as Market Data's
    ``flag_surface`` docstring says the check would eventually land here. Refusing to construct
    a surface that violates it would also destroy the measurement, for the same reason the
    butterfly condition is not an ``SVIParams`` invariant: you cannot report how far an object
    is from admissible if you are not allowed to hold it.

    What *is* enforced is structural: the slices are ordered, no expiry appears twice, and there
    is at least one of them. Those are not modelling judgements, they are what makes a
    collection of slices addressable as a surface -- an interpolator walking the tenor axis
    assumes the order, and a duplicated expiry means two different answers to the same question.
    """

    slices: tuple[SVISlice, ...]
    """Fitted slices, strictly increasing in ``tenor_years``, expiries unique. Non-empty.

    A tuple rather than a list because this object is frozen and hashable-by-value in spirit; a
    mutable field would let a caller reorder the surface behind the invariant that was checked
    at construction.
    """

    def __post_init__(self) -> None:
        if not self.slices:
            raise ValueError("A surface must hold at least one slice")

        # `pairwise`, not `zip(xs, xs[1:])`: consecutive pairs differ in length by one by
        # design, so the strict= flag that this codebase requires everywhere else would be
        # wrong here and its absence would be an unexplained exception.
        for near, far in pairwise(self.slices):
            if near.tenor_years >= far.tenor_years:
                raise ValueError(
                    "The slices must be strictly increasing in tenor, got "
                    f"{near.tenor_years} before {far.tenor_years}"
                )

        # Checked separately from the tenor ordering rather than deduced from it. Two expiries
        # can round to the same year fraction under a coarse day count without being the same
        # instant, and the reverse -- one expiry appearing twice with tenors that differ by a
        # rounding error -- would slip past a purely ordinal test.
        expiries = [one.expiry for one in self.slices]
        if len(set(expiries)) != len(expiries):
            raise ValueError(f"The slices must have unique expiries, got {expiries}")
