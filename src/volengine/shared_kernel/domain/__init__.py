"""Value objects shared by every context.

Frozen, slotted, validated in ``__post_init__`` and compared by value: two ``Strike`` built
from the same number are the same strike, which is what makes them safe as dictionary keys
and safe to pass around without defensive copies.

They wrap a single float on purpose. The point is not the arithmetic, it is that a function
taking a ``Strike`` cannot silently be handed a tenor, and that "positive and finite" is
enforced once at construction instead of re-checked at every call site.
"""

from __future__ import annotations
