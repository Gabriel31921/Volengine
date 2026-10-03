"""Platform implementations that touch a file, rather than only memory or the event loop.

The rest of ``platform/`` -- the bus, the clocks, the executors, the null and logging sinks -- is
machinery that lives and dies with the process. What lives here writes something that outlives
it, which is the one property worth a subpackage: it is where an ``OSError`` can come from, and
where a reader of the composition root should look when a run leaves a file behind.

The import rule is unchanged (rule 7): ``contracts/`` and stdlib, plus the rest of ``platform/``.
"""

from __future__ import annotations
