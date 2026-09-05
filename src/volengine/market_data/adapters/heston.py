"""A Heston surface, priced semi-analytically: the second generator the synthetic feed can quote.

``SyntheticProvider`` was born quoting SVI slices, and it still does. This module adds the other
generator Design 4.6 asks for, and adds it **beside** the first rather than in place of it, because
the two answer different questions. An SVI slice is a shape someone drew: five numbers per expiry,
independent of each other, and a fit measured against them recovers exactly what was written down.
Heston is a *model of how the market got there* -- one variance process, five parameters for the
whole surface -- so the smile it produces is not in the calibrator's own family and its term
structure is a consequence rather than a choice. A calibrator that scored perfectly against SVI and
badly here would be telling you it had learned the generator instead of the market.

**The two are connected by nothing but a shape.** ``synthetic.VolatilitySpec`` asks for one method,
``volatility(k, tenor_years)``, and both spellings satisfy it structurally -- so this module does
not import ``synthetic`` and ``synthetic`` does not import this one. A third generator arrives the
same way.

**How the price is computed.** The Heston characteristic function of ``ln(F_T / F_0)`` is known in
closed form, and Lewis's single-integral representation turns it into a forward-normalised European
call::

    C(k) / F = 1 - exp(k/2) / pi * integral_0^inf Re[exp(-i u k) phi(u - i/2)] / (u^2 + 1/4) du

evaluated numerically, then inverted through the shared kernel's Black-76 to the volatility the
feed quotes. That inversion is the whole point: everything downstream of this module speaks
volatilities and premiums, so a second generator has to arrive as a volatility surface or nothing
in the engine would be able to consume it.

**The integral is the only hard part, and the hard part is the truncation.** The Heston
characteristic function does *not* decay like a Gaussian: for large ``u`` it decays like
``exp(-C_inf * u)`` with ``C_inf = sqrt(1 - rho^2) (v0 + kappa theta T) / sigma``, a tail that gets
fatter the larger the vol of vol is. A range chosen from the total standard deviation alone -- what
an implementation written for a Black world would pick -- is then far too short, and the price it
returns is wrong in the sixth decimal: large enough to matter to a generator the engine is measured
against, small enough to look perfectly plausible. :func:`_upper_limit` takes the larger of the two
requirements, and the panels of the composite Gauss-Legendre rule are counted from the oscillation
of ``exp(-i u k)`` rather than fixed, because a fixed-node rule loses digits over a panel spanning
several waves. Measured against a 400001-point Simpson rule over a deliberately excessive range,
tenors from one week to two years, vol of vol up to 1.5 and ``|k| <= 1.0``, the worst error is
under 1e-9 of the forward.

**That accuracy is absolute, not relative, and the deep wing is where the difference bites.** A
one-week option a quarter of the way out of the money is worth less than 1e-11 of the forward, and
this rule cannot resolve it: the price it returns there is the quadrature's own noise and can even
come out slightly negative. :data:`PRICE_RESOLUTION` is where that is refused rather than quoted --
a volatility inverted from noise is exactly the silent wrong answer this module is written to
avoid, and it is the same trap the shared kernel's ``norm_cdf`` was written around. A generator
asked for a chain it cannot price says so; it does not invent one.

**What this module is not.** It is not a calibrator and there is no Heston fit anywhere in the
engine: the parameters here are handwritten configuration, the same as an SVI slice's. It is also
not a path generator -- Design 10.3 keeps that as an extension, and the closed form below is the
piece that would stay unchanged if paths were ever added beside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt

from volengine.shared_kernel.domain.black76 import implied_vol

QUADRATURE_ORDER = 32
"""Gauss-Legendre nodes per panel. Enough that a panel holding half an oscillation is exact to
machine precision, so the accuracy of the whole rule is decided by the panel count and the
truncation rather than by this number."""

MIN_PANELS = 32
"""Panels used even at the money, where ``exp(-i u k)`` does not oscillate at all.

At ``k = 0`` the integrand is a smooth decaying hump and the oscillation count below is zero, which
would leave the rule with a single panel spanning the entire truncated range. The floor is what
stops the at-the-money quote -- the one every downstream test leans on -- from being the least
accurate point on the slice.
"""

PANELS_PER_OSCILLATION = 2.0
"""Panels per full period of ``exp(-i u k)``. Two, so no panel spans more than half a wave.

Gauss-Legendre is exact for polynomials, and a panel covering several periods of a sinusoid is
where a fixed-node rule quietly loses digits. Counting panels from the oscillation rather than
fixing them is what keeps the wings as accurate as the money.
"""

TOTAL_STDEV_REACH = 14.0
"""How many total standard deviations the integration range must cover.

The Gaussian half of the truncation: at ``u * sqrt(vbar * T) = 14`` the near-Gaussian body of the
characteristic function has fallen below ``exp(-98)``, so nothing beyond it can matter.
"""

EXPONENTIAL_REACH = 45.0
"""How many e-foldings of the *exponential* tail the range must cover.

The half of the truncation that is easy to get wrong. ``C_inf = sqrt(1 - rho^2) (v0 + kappa theta
T) / sigma`` is the decay rate, so this requirement scales like ``1 / T`` where the Gaussian one
scales like ``1 / sqrt(T)``: the exponential tail is what decides the range at long tenors and at
high vol of vol, and it is invisible at the parameters a first test happens to pick. The guard in
``tests/market_data/test_heston.py`` is a market with a vol of vol of 1.5, where a Gaussian-only
range is wrong by 8.6e-6 at one year -- four orders of magnitude worse than the rule below.
"""

PRICE_RESOLUTION = 1e-10
"""Smallest forward-normalised premium this module will hand back as a volatility.

Not a threshold in ADR-012's sense -- an operator retuning it would only be choosing to publish
noise -- but a property of double-precision quadrature. The integral is computed to an absolute
accuracy near 1e-11 of the forward, so an out-of-the-money option worth less than that is *below
the arithmetic*, and the number that comes out is the rule's own rounding error rather than a
price. Inverting it would produce a volatility with no digits in it at all, and a generated surface
with one nonsense point on it is worse than one that refused: the refusal names the strike.
"""

_CACHED_TENORS = 32
"""How many (parameters, tenor) quadratures are kept.

The nodes and the characteristic function depend on the tenor and the model, never on the strike,
so a whole strike ladder costs one evaluation and one dot product each. The cache is what makes
that true across the calls the provider makes one strike at a time. Purely an optimisation: the
function it decorates is deterministic, so a hit and a miss return the identical arrays.
"""


@dataclass(frozen=True, slots=True)
class HestonParamsSpec:
    """The five Heston parameters, as a *specification* of the market to generate.

    A specification, never the output of a fit -- nothing in this engine calibrates Heston. The
    invariants are therefore the model's own domain and nothing looser: an optimiser never walks
    through these, so there is no iterate to admit.

    The Feller condition ``2 kappa theta >= sigma^2`` is deliberately **not** enforced. It governs
    whether the variance process can touch zero along a path, and this module simulates no paths:
    the closed form below is an expectation, and it remains perfectly well defined for a violated
    Feller condition -- which is the regime real equity and crypto surfaces are usually fitted in.
    Refusing it here would rule out the parameter sets most worth generating.
    """

    v0: float
    """Variance now, in annualised variance units: ``0.36`` is a 60% spot volatility. Positive."""

    kappa: float
    """Speed the variance reverts at, per year. Positive. Larger means a flatter term structure."""

    theta: float
    """Variance it reverts to. Positive. With ``v0`` it fixes both ends of the term structure."""

    vol_of_vol: float
    """Volatility of the variance process, ``sigma`` in the literature. Positive.

    Spelled out rather than called ``sigma`` because SVI already owns that name one class over and
    means something entirely different by it -- a smile-width parameter, not a volatility. Zero is
    refused: the closed form divides by its square, so the deterministic-variance limit is
    approached numerically and cannot be evaluated at.
    """

    rho: float
    """Correlation between the price and its variance. Strictly inside ``(-1, 1)``.

    The skew. Negative is the sign every equity and crypto book prints -- variance rises as the
    price falls -- and it is what tilts the generated smile's downside wing above its upside one.
    """

    def __post_init__(self) -> None:
        # Finiteness first and the bad cases joined with `or`, because `float("nan") <= 0` is
        # `False` and an ordering guard alone lets a NaN through into the quadrature, where it
        # becomes a NaN price and then an unhelpful inversion error a long way from here.
        for name, value in (
            ("v0", self.v0),
            ("kappa", self.kappa),
            ("theta", self.theta),
            ("vol_of_vol", self.vol_of_vol),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"The Heston parameter {name} must be positive, got {value}")
        if not math.isfinite(self.rho) or abs(self.rho) >= 1:
            raise ValueError(f"The Heston parameter rho must be inside (-1, 1), got {self.rho}")

    def expected_variance(self, tenor_years: float) -> float:
        """Mean variance over the life of the option, ``E[(1/T) int_0^T v_t dt]``.

        ``theta + (v0 - theta) * (1 - exp(-kappa T)) / (kappa T)``, in closed form: the whole term
        structure of the model's *level*, with none of its smile. It is the implied variance this
        surface would have if ``vol_of_vol`` were zero, which makes it both the natural scale for
        the quadrature's truncation and the statement a test can check the level against.

        Raises:
            ValueError: If the tenor is not positive and finite. There is no average over an
                interval of no length.
        """
        _require_positive_finite(tenor_years, "tenor")
        decay = kappa_decay(self.kappa, tenor_years)
        return self.theta + (self.v0 - self.theta) * decay

    def call_price(self, k: float, tenor_years: float) -> float:
        """Forward-normalised European call at log-forward-moneyness ``k``: ``C / F``.

        Normalised by the forward on purpose, and it is the same reason the calibrators receive
        homogeneous numbers: a model surface has no currency, and multiplying by ``F`` is the
        caller's business. ``k = ln(K / F)``, so ``k = 0`` is at the money forward and the
        normalised strike is ``exp(k)``.

        Raises:
            ValueError: If ``k`` is not finite or the tenor is not positive and finite.
        """
        if not math.isfinite(k):
            raise ValueError(f"The log-moneyness must be finite, got {k}")
        _require_positive_finite(tenor_years, "tenor")
        nodes, weights, characteristic = _quadrature(self, tenor_years)
        integrand = np.real(np.exp(-1j * nodes * k) * characteristic) / (nodes**2 + 0.25)
        integral = float(np.dot(weights, integrand))
        return 1.0 - math.exp(k / 2.0) / math.pi * integral

    def volatility(self, k: float, tenor_years: float) -> float:
        """The Black-76 volatility this model implies at that moneyness and tenor.

        The method ``synthetic.VolatilitySpec`` asks for, which is the whole of what the feed needs
        from a generator. The model price is turned into a volatility here rather than downstream
        because a volatility is the only thing the rest of the engine can consume: the feed adds
        its noise in volatility, prices the result through the shared kernel, and the calibrator
        inverts its way back.

        **The out-of-the-money leg is the one inverted**, with the in-the-money price obtained by
        put-call parity. Both legs carry identical time value, but an in-the-money call is mostly
        intrinsic, so inverting it asks a root-finder to recover a small residue from the
        difference of two large numbers -- the same reason ADR-017 has the pricing ACL keep the
        out-of-the-money twin.

        Raises:
            ValueError: If ``k`` is not finite, the tenor is not positive and finite, or the
                out-of-the-money premium falls below :data:`PRICE_RESOLUTION` -- a wing this
                quadrature cannot resolve, which is a chain to narrow rather than a market state.
        """
        premium = self.call_price(k, tenor_years)
        strike = math.exp(k)
        is_call = k >= 0.0
        if not is_call:
            # Put-call parity on a forward-normalised, undiscounted contract: P = C - (1 - K).
            premium = premium - 1.0 + strike
        if premium < PRICE_RESOLUTION:
            raise ValueError(
                f"The option at log-moneyness {k} and tenor {tenor_years} is worth {premium} of "
                f"the forward, which is below the {PRICE_RESOLUTION} this quadrature resolves: "
                "no volatility can be inverted from it"
            )
        return implied_vol(
            target_price=premium,
            forward=1.0,
            strike=strike,
            tenor_years=tenor_years,
            is_call=is_call,
        )


def kappa_decay(kappa: float, tenor_years: float) -> float:
    """``(1 - exp(-kappa T)) / (kappa T)``: the weight the *current* variance keeps over a tenor.

    One at a zero tenor and falling to zero over a long one, which is the shape of every Heston
    term structure. Written out rather than inlined because the small-``kappa T`` limit needs care:
    the expression is ``0 / 0`` there, and ``math.expm1`` is what keeps it accurate instead of
    cancelling to a few significant figures.
    """
    exponent = kappa * tenor_years
    if exponent < 1e-8:
        return 1.0
    return -math.expm1(-exponent) / exponent


@lru_cache(maxsize=_CACHED_TENORS)
def _quadrature(
    params: HestonParamsSpec, tenor_years: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    """Nodes, weights and the characteristic function at those nodes, for one model and tenor.

    Cached because none of the three depends on the strike: a ladder of forty strikes at one expiry
    costs one characteristic function and forty dot products. The arrays are returned rather than
    copied, so **no caller may write into them** -- they are shared with every other caller holding
    the same tenor.
    """
    upper = _upper_limit(params, tenor_years)
    panels = _panel_count(upper)
    nodes, weights = _composite_nodes(panels, upper)
    argument = np.asarray(nodes - 0.5j, dtype=np.complex128)
    return nodes, weights, _characteristic(argument, params, tenor_years)


def _upper_limit(params: HestonParamsSpec, tenor_years: float) -> float:
    """Where the integral is cut off: the larger of the Gaussian and the exponential reach.

    The two tails are genuinely different mechanisms and the larger one wins at different ends of
    the curve -- the Gaussian dominates a two-year tenor, the exponential a one-week one -- so
    taking the maximum is what makes one rule cover the whole surface.
    """
    total_stdev = math.sqrt(params.expected_variance(tenor_years) * tenor_years)
    exponential_rate = (
        math.sqrt(1.0 - params.rho**2)
        * (params.v0 + params.kappa * params.theta * tenor_years)
        / params.vol_of_vol
    )
    return max(TOTAL_STDEV_REACH / total_stdev, EXPONENTIAL_REACH / exponential_rate)


def _panel_count(upper: float) -> int:
    """How many panels the range is split into, counted from how fast the integrand oscillates.

    The oscillation is ``exp(-i u k)`` and the widest ``|k|`` this module is documented for is 1.0,
    which is what the count is taken at: sizing the panels per call would make the *quadrature*
    depend on the strike, and with it the cache, turning one characteristic function per expiry
    into one per quote.
    """
    oscillations = upper / (2.0 * math.pi)
    return max(MIN_PANELS, math.ceil(oscillations * PANELS_PER_OSCILLATION))


def _composite_nodes(
    panels: int, upper: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """A composite Gauss-Legendre rule over ``[0, upper]``, flattened into one node array."""
    abscissae, weights = np.polynomial.legendre.leggauss(QUADRATURE_ORDER)
    edges = np.linspace(0.0, upper, panels + 1)
    low, high = edges[:-1, None], edges[1:, None]
    middle, half = 0.5 * (low + high), 0.5 * (high - low)
    return (
        np.asarray((middle + half * abscissae[None, :]).ravel(), dtype=np.float64),
        np.asarray((half * weights[None, :]).ravel(), dtype=np.float64),
    )


def _characteristic(
    u: npt.NDArray[np.complex128], params: HestonParamsSpec, tenor_years: float
) -> npt.NDArray[np.complex128]:
    """``E[exp(i u ln(F_T / F_0))]`` under the Heston dynamics, in the branch-stable form.

    The textbook expression has two algebraically equal spellings, differing in whether the ratio
    inside the logarithm is built from ``beta - d`` or ``beta + d``. They are **not**
    numerically equal: the second sends the argument of the complex logarithm across its branch cut
    as ``u`` grows, and the integral above then integrates a function with jumps in it. That is the
    "little Heston trap", and it produces prices that look plausible and are wrong. The form below
    is the stable one, and the sign convention is checked by the martingale identity
    ``phi(-i) == 1``, which a test pins.
    """
    beta = params.kappa - params.rho * params.vol_of_vol * 1j * u
    discriminant = np.sqrt(beta**2 + params.vol_of_vol**2 * (1j * u + u**2))
    ratio = (beta - discriminant) / (beta + discriminant)
    decay = np.exp(-discriminant * tenor_years)
    variance_term = (
        (beta - discriminant) / params.vol_of_vol**2 * (1.0 - decay) / (1.0 - ratio * decay)
    )
    drift_term = (params.kappa * params.theta / params.vol_of_vol**2) * (
        (beta - discriminant) * tenor_years - 2.0 * np.log((1.0 - ratio * decay) / (1.0 - ratio))
    )
    return np.asarray(np.exp(drift_term + variance_term * params.v0), dtype=np.complex128)


def _require_positive_finite(value: float, what: str) -> None:
    """Reject anything that is not a usable positive number, NaN included."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"The {what} must be positive and finite, got {value}")
