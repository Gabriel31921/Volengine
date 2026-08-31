"""Builders for the JAX side of this context, kept apart from ``builders.py`` on purpose.

``builders.py`` is imported by every test module here, including the ones that must run with no
optional extra installed, so it may not touch ``jax``. This module may, and only the modules that
already require the extra import it.

**One shape and one settings object for the whole session**, which is not tidiness either. The
calibrator compiles in its constructor, and :func:`~volengine...jax_calibrator._compiled` caches
that compilation against exactly the ``(settings, shape)`` pair it was built from. Every test that
takes the defaults below therefore shares one compilation; a test that bends a knob pays for
another, which is a good reason to bend one only when the test is about it.
"""

from __future__ import annotations

from volengine.parametric_pricing.adapters.jax_calibrator import JaxCalibrator, JaxFitSettings
from volengine.parametric_pricing.adapters.padding import PadShape

TEST_SHAPE = PadShape(max_slices=4, max_quotes=32, mesh_nodes=21)
"""The rectangle the tests fit on: four expiries, thirty-two strikes, twenty-one mesh nodes.

Smaller than the production reservation of ADR-009 and identical in kind. The padding is exercised
by every test that uses it -- a two-slice task leaves half the rows empty -- and the arithmetic on
the reserved cells is what the suite would otherwise spend its time on.
"""

TEST_SETTINGS = JaxFitSettings()
"""The shipped defaults, named so that every test asks for the same object and shares its
compilation."""


def make_jax_calibrator(
    settings: JaxFitSettings | None = None, shape: PadShape | None = None
) -> JaxCalibrator:
    """A calibrator on the shared test shape, with one knob if a test needs it."""
    return JaxCalibrator(
        settings=TEST_SETTINGS if settings is None else settings,
        shape=TEST_SHAPE if shape is None else shape,
    )
