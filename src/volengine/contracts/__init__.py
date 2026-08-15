"""The published language between bounded contexts.

Every DTO that crosses a boundary lives here, and nothing else does. These types are data:
no pricing, no greeks, no ``implied_vol(K, T)`` method (ADR-001). A contract you can only use
by holding a reference to the producer's memory is not a contract -- it cannot be serialized,
cannot cross a process, cannot be replayed from a recording tomorrow.

Rules that hold for everything in this package:

- Imports are stdlib plus ``shared_kernel/`` only. **Never numpy, jax or torch** (ADR-011):
  the moment a contract carries an array type, the consumer inherits the producer's numerical
  stack and the boundary stops being a boundary.
- ``to_dict()`` emits primitives only -- no datetime, no enum object, no dataclass -- because
  the bus is not allowed to assume shared memory (ADR-003). Everything here survives a
  ``json.dumps`` round trip, and the tests assert the real hop rather than a round-trip
  equality that shared references would satisfy trivially.
- ``schema_version`` is present from day one and checked on read.
- ``StrEnum`` members carry explicit string values. That string is the wire format, and
  ``auto()`` would let it change silently when someone renames a member.
- Convention-dependent quantities are resolved *before* they get here (ADR-002).

Only ``*/application/acl.py`` in each context may build or consume these types. The domains
do not import this package at all.
"""

from __future__ import annotations
