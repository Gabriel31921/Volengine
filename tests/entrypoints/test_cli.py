"""The command line: the four verbs, and what each says when it cannot do its job.

Driven through typer's own runner, so what is asserted is what a person would see -- an exit code
and a line on stderr -- rather than the return value of a function nobody calls that way.

``run`` and ``report`` are not exercised to completion here, and cannot be until F1-08 registers
an adapter: ``default_adapters()`` is empty, so every path through them stops at the registry.
That is the seam, and the tests below pin the *message* it fails with, because "no provider
adapter is registered under 'constant'" is the difference between a five-second fix and an
afternoon. The graph those commands build is covered against fakes in ``test_pipeline.py``.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tests.entrypoints.builders import CONFIG_TOML, replacing, write_config
from volengine.entrypoints.cli import app

runner = CliRunner()


def invoke(*arguments: str) -> tuple[int, str]:
    """Run the CLI and return what a shell would see."""
    result = runner.invoke(app, list(arguments))
    return result.exit_code, result.output


# --- the two operative commands


def test_run_stops_at_the_registry_until_the_walking_skeleton(tmp_path: Path) -> None:
    """F1-07 composes the graph; F1-08 supplies the adapters it names. See docs/SEAMS.md.

    The assertion names the registry rather than one of its three mappings: which lookup fails
    first is an ordering detail of ``build_pipeline``, and a test pinned to it would break the day
    F1-08 registers a provider and leaves the writer for last.
    """
    code, output = invoke("run", "--config", str(write_config(tmp_path)))

    assert code == 2
    assert "adapter is registered under" in output


def test_report_refuses_a_count_of_zero(tmp_path: Path) -> None:
    code, output = invoke("report", "--config", str(write_config(tmp_path)), "--count", "0")

    assert code == 2
    assert "--count must be positive" in output


def test_an_unreadable_file_is_reported_rather_than_raised(tmp_path: Path) -> None:
    code, output = invoke("run", "--config", str(tmp_path / "absent.toml"))

    assert code == 2
    assert "cannot be read" in output


def test_a_bad_threshold_names_its_table(tmp_path: Path) -> None:
    path = write_config(tmp_path, replacing(CONFIG_TOML, "warn_seconds", "warn_seconds = -1.0"))

    code, output = invoke("run", "--config", str(path))

    assert code == 2
    assert "risk.freshness" in output


# --- narrowing what runs


def test_an_unknown_market_lists_the_ones_configured(tmp_path: Path) -> None:
    code, output = invoke("run", "--config", str(write_config(tmp_path)), "--market", "ETH-DERIBIT")

    assert code == 2
    assert "BTC-DERIBIT" in output


def test_a_configured_market_gets_past_the_selection(tmp_path: Path) -> None:
    """The vacuous-pass guard on the test above: the right name fails later, at the registry."""
    code, output = invoke("run", "--config", str(write_config(tmp_path)), "--market", "BTC-DERIBIT")

    assert code == 2
    assert "adapter is registered under" in output


def test_an_unknown_calibrator_lists_the_ones_configured(tmp_path: Path) -> None:
    code, output = invoke(
        "run", "--config", str(write_config(tmp_path)), "--calibrators", "svi-jax"
    )

    assert code == 2
    assert "svi-scipy" in output


def test_an_empty_calibrator_flag_is_refused(tmp_path: Path) -> None:
    code, output = invoke("run", "--config", str(write_config(tmp_path)), "--calibrators", ",")

    assert code == 2
    assert "names nothing" in output


# --- the two declared ones


def test_record_says_which_task_implements_it(tmp_path: Path) -> None:
    code, output = invoke("record", "--config", str(write_config(tmp_path)))

    assert code == 1
    assert "F3-B" in output


def test_replay_says_which_task_implements_it(tmp_path: Path) -> None:
    code, output = invoke("replay", "--config", str(write_config(tmp_path)))

    assert code == 1
    assert "F3-B" in output


def test_every_verb_the_design_names_is_declared() -> None:
    """``--help`` is the engine's shape as a person meets it, including the modes not built yet."""
    _, output = invoke("--help")

    assert all(verb in output for verb in ("run", "report", "record", "replay"))
