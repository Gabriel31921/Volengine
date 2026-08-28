from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from datetime import timedelta

import pytest

from tests.risk.builders import EXPIRIES, NAIVE, NOW, make_portfolio, make_position
from tests.support import replace_field
from volengine.risk.domain.pricing import OptionKindR

# --- Position: the fields it keeps


def test_position_keeps_every_field_as_given() -> None:
    """Nothing is normalised, rounded or reordered on the way in."""
    position = make_position(
        underlying="ETH", expiry=EXPIRIES[0], strike=3_200.0, kind=OptionKindR.PUT, quantity=-2.5
    )

    assert position.underlying == "ETH"
    assert position.expiry == EXPIRIES[0]
    assert position.strike == 3_200.0
    assert position.kind is OptionKindR.PUT
    assert position.quantity == -2.5


@pytest.mark.parametrize(
    ("value", "field"),
    [(make_position(), "quantity"), (make_portfolio(), "positions")],
)
def test_the_book_is_frozen_and_slotted(value: object, field: str) -> None:
    """Both are values: rewriting a quantity in place would leave a report unauditable."""
    assert not hasattr(value, "__dict__")

    with pytest.raises(FrozenInstanceError):
        setattr(value, field, None)


# --- Position: the underlying


def test_position_rejects_an_empty_underlying() -> None:
    with pytest.raises(ValueError, match="underlying must not be empty"):
        replace(make_position(), underlying="")


def test_position_rejects_a_whitespace_only_underlying() -> None:
    """A line of padding read out of a portfolio file names no underlying at all."""
    with pytest.raises(ValueError, match="underlying must not be empty"):
        replace(make_position(), underlying="   ")


def test_position_does_not_cross_check_the_underlying_against_any_market() -> None:
    """The documented open seam: nothing in the domain can tell this string is wrong.

    ``CalibratedSurface`` publishes a ``market_id`` and no underlying, so a position naming an
    instrument no surface describes is still perfectly constructible. Pinned here so that the day
    someone adds a check, they have to delete a test that says why there is none.
    """
    assert make_position(underlying="DOGE").underlying == "DOGE"


# --- Position: the expiry


def test_position_rejects_a_naive_expiry() -> None:
    """The tenor is this instant minus the snapshot's; mixing the two kinds raises TypeError."""
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(make_position(), expiry=NAIVE)


def test_position_accepts_an_expiry_already_in_the_past() -> None:
    """Representable on purpose: the report has to be able to say a line has expired.

    That verdict is ``ExpiredPositionError`` from ``SurfaceView.tenor_of``, which needs a surface
    to compare against, so it cannot be reached from a constructor that has never seen one.
    """
    stale = make_position(expiry=NOW - timedelta(days=1))

    assert stale.expiry < NOW


# --- Position: the strike


@pytest.mark.parametrize("bad", [0.0, -1.0, -60_400.0])
def test_position_rejects_a_non_positive_strike(bad: float) -> None:
    with pytest.raises(ValueError, match="strike must be positive"):
        replace_field(make_position(), "strike", bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_position_rejects_a_non_finite_strike(bad: float) -> None:
    """NaN is in the parametrisation on purpose: it is the case an ordering guard misses."""
    with pytest.raises(ValueError, match="strike must be positive and finite"):
        replace_field(make_position(), "strike", bad)


def test_a_nan_would_pass_a_strike_guard_written_without_isfinite() -> None:
    """Vacuity guard for the two tests above: prove the poisoned value really is poison.

    ``float("nan") <= 0`` is ``False``, so a guard reading ``if self.strike <= 0: raise`` would
    admit a NaN strike, and the rejection test would then be asserting that ``isfinite`` does work
    it never actually did. This is the repo's recurring trap, and it is why the guard tests
    finiteness first and joins the bad cases with ``or``.
    """
    assert not (float("nan") <= 0)
    assert not (float("nan") > 0)
    assert not math.isfinite(float("nan"))


# --- Position: the quantity, and the three sizes that are all legal


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_position_rejects_a_non_finite_quantity(bad: float) -> None:
    """The only size this context cannot use. Sign and zero are both meaningful."""
    with pytest.raises(ValueError, match="quantity must be finite"):
        replace_field(make_position(), "quantity", bad)


def test_position_accepts_a_zero_quantity() -> None:
    """A leg flattened intraday stays in the file so that its risk line still shows.

    The guard is therefore ``isfinite`` only: ``not 0.0`` is ``True``, so a truthiness test would
    reject exactly the case the field was designed to carry.
    """
    assert make_position(quantity=0.0).quantity == 0.0


def test_position_accepts_a_negative_quantity() -> None:
    """Short is a sign on the size, never a fourth member of the kind enum."""
    assert make_position(quantity=-4.0).quantity == -4.0


def test_a_zero_quantity_would_be_rejected_by_a_truthiness_guard() -> None:
    """Vacuity guard for the test above: ``not 0.0`` really is ``True``.

    Without this, ``test_position_accepts_a_zero_quantity`` reads as though it were guarding
    against nothing in particular. The mistake it guards against is one line of plausible Python.
    """
    assert not 0.0
    assert math.isfinite(0.0)


# --- Position: the option side


def test_position_kind_survives_as_a_str_enum_member() -> None:
    """A ``StrEnum`` *is* a ``str``, so equality alone cannot tell a member from a bare string."""
    position = make_position(kind=OptionKindR.PUT)

    assert isinstance(position.kind, OptionKindR)
    assert position.kind is OptionKindR.PUT


def test_position_kind_carries_no_direction_of_its_own() -> None:
    """Only two sides exist: a short call is ``CALL`` with a negative quantity."""
    assert {member.value for member in OptionKindR} == {"CALL", "PUT"}


# --- Position: equality is structural


def test_two_identically_built_positions_are_equal() -> None:
    """Compared by value, which is what makes duplicate positions a decision rather than a bug."""
    assert make_position() == make_position()


def test_positions_differing_only_in_kind_are_not_equal() -> None:
    """Same strike and same expiry on the two sides are two different contracts."""
    assert make_position(kind=OptionKindR.CALL) != make_position(kind=OptionKindR.PUT)


# --- Portfolio


def test_portfolio_rejects_an_empty_book() -> None:
    """The honest report over an empty book is a total of ``0.0`` and no lines.

    That is indistinguishable, on the page, from a book whose every option expired worthless, so
    a portfolio file that failed to load must fail loudly here instead of reporting a plausible
    number.
    """
    with pytest.raises(ValueError, match="at least one position"):
        make_portfolio(positions=())


def test_portfolio_accepts_a_single_position() -> None:
    assert len(make_portfolio(positions=(make_position(),)).positions) == 1


def test_portfolio_accepts_duplicate_positions() -> None:
    """Two lots of the same option are two positions; netting them is the reader's decision."""
    duplicated = make_portfolio(positions=(make_position(), make_position()))

    assert len(duplicated.positions) == 2
    assert duplicated.positions[0] == duplicated.positions[1]


def test_duplicate_positions_would_collapse_in_a_set() -> None:
    """Vacuity guard for the test above: the two lots really are indistinguishable by value.

    A ``Position`` is a frozen dataclass and therefore hashable, so storing the book in a ``set``
    or a dict keyed by position would silently halve it. Keeping a ``tuple`` is what makes the
    duplicate survive, and this is the assertion that proves the tuple is doing the work.
    """
    assert len({make_position(), make_position()}) == 1


def test_portfolio_preserves_the_order_it_was_given() -> None:
    """No sorting and no grouping: the order the file was written in is the order it is read."""
    first = make_position(strike=70_000.0)
    second = make_position(strike=50_000.0)

    assert make_portfolio(positions=(first, second)).positions == (first, second)


def test_portfolio_carries_no_market_id() -> None:
    """Which market prices a book is routing, answered by the composition root.

    An architecture test rather than a behavioural one: it keeps the decision standing against
    the next reader who finds it convenient to staple a market onto the aggregate.
    """
    assert not hasattr(make_portfolio(), "market_id")
