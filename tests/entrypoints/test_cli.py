"""The command line: the four verbs, and what each says when it cannot do its job.

Driven through typer's own runner, so what is asserted is what a person would see -- an exit code
and a line on stderr -- rather than the return value of a function nobody calls that way.

``run`` and ``report`` are not exercised to completion here: the file every test below starts
from names ``svi-jax``, which is F3-A's calibrator and which no adapter is registered for, so
those two paths stop at the registry. That is deliberate -- the tests pin the *message* it fails
with, because "no calibrator adapter is registered under 'svi-jax'" is the difference between a
five-second fix and an afternoon. The graph those commands build is covered against fakes in
``test_pipeline.py``, the run that reaches a report in ``test_walking_skeleton.py``, and the one
that reaches it through a fit in ``test_synthetic_vertical.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.entrypoints.builders import CONFIG_TOML, replacing, write_config
from volengine.entrypoints.cli import app
from volengine.market_data.adapters.recorded import open_recording
from volengine.parametric_pricing.adapters.flat_vol import PRODUCER_ID

runner = CliRunner()


def invoke(*arguments: str) -> tuple[int, str]:
    """Run the CLI and return what a shell would see."""
    result = runner.invoke(app, list(arguments))
    return result.exit_code, result.output


# --- the two operative commands


def test_run_stops_at_the_registry_when_the_file_names_an_unregistered_adapter(
    tmp_path: Path,
) -> None:
    """A start-up failure by design: a name nobody registered is refused before anything runs.

    The assertion names the registry rather than one of its three mappings: which lookup fails
    first is an ordering detail of ``build_pipeline``, and a test pinned to it broke the day
    F1-08 registered a provider and a writer but not this file's calibrator.
    """
    code, output = invoke("run", "--config", str(write_config(tmp_path)))

    assert code == 2
    assert "adapter is registered under" in output


def test_run_refuses_a_duration_of_zero(tmp_path: Path) -> None:
    """Caught before the file is even read: "stop before starting" is a typo, not a session."""
    code, output = invoke("run", "--config", str(write_config(tmp_path)), "--duration", "0")

    assert code == 2
    assert "--duration must be positive" in output


@pytest.mark.parametrize("duration", ["nan", "inf"])
def test_run_refuses_a_duration_that_is_not_a_number_of_seconds(
    tmp_path: Path, duration: str
) -> None:
    """The NaN-ordering trap, at the boundary that owns flag validation.

    ``nan <= 0`` is ``False``, so a guard written as an ordering test alone hands both of these to
    ``Pipeline.run``, whose ``ValueError`` nothing in this module catches -- the operator gets a
    traceback and exit 1 where the whole point of this layer is a message and exit 2. ``inf`` is
    the same hole seen from the other side: it passes the ordering test outright and would sleep
    for the rest of the session.
    """
    code, output = invoke("run", "--config", str(write_config(tmp_path)), "--duration", duration)

    assert code == 2
    assert "--duration must be positive and finite" in output


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
    assert "svi-jax" in output


def test_an_empty_calibrator_flag_is_refused(tmp_path: Path) -> None:
    code, output = invoke("run", "--config", str(write_config(tmp_path)), "--calibrators", ",")

    assert code == 2
    assert "names nothing" in output


# --- recording and replay


def runnable(directory: Path) -> Path:
    """The same file, naming adapters this build registers and a position it can value.

    ``CONFIG_TOML`` names ``svi-jax`` on purpose, which is what makes the tests above stop at the
    registry. The two verbs below have to reach the end of a session instead, so they get the
    walking skeleton's three adapters and an expiry inside the tenors the constant feed quotes.
    """
    text = replacing(CONFIG_TOML, "calibrators", f'calibrators = ["{PRODUCER_ID}"]')
    text = replacing(text, "material_move_threshold", "material_move_threshold = 0.0")
    expiry = datetime.now(UTC) + timedelta(days=45)
    text = replacing(text, "expiry = ", f"expiry = {expiry:%Y-%m-%d}T08:00:00Z")
    return write_config(directory, text)


@pytest.fixture(scope="module")
def session(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, str]:
    """One recorded session, shared by every test below that needs one.

    Module-scoped because the constant feed republishes its chain once a second and a session is
    therefore four seconds long: recording it once per test would spend most of this file's runtime
    watching a fixture. Nothing below writes to the two paths, and the one test that has to change
    the configuration copies it into its own directory first.
    """
    directory = tmp_path_factory.mktemp("session")
    config, recording = runnable(directory), directory / "session.jsonl"
    code, output = invoke("record", "--config", str(config), "--recording", str(recording))

    assert code == 0
    return config, recording, output


def test_record_writes_a_recording_naming_the_market_it_ran(
    session: tuple[Path, Path, str],
) -> None:
    """The recording half of ADR-004 from the command line: a run, and a file of what it saw."""
    _, recording, _ = session

    assert open_recording(recording).market_id == "BTC-DERIBIT"


def test_record_still_writes_the_reports_of_the_session_it_records(
    session: tuple[Path, Path, str],
) -> None:
    """A tap, not a mode: what is recorded is an ordinary run, and it still values the book.

    Without this the recorder could be running a reduced pipeline, and the first thing a replay
    would be asked to reproduce is a session nobody will ever have.
    """
    _, _, output = session

    assert f"[BTC-DERIBIT / {PRODUCER_ID}]" in output


def test_record_refuses_to_choose_between_two_markets(tmp_path: Path) -> None:
    """One file holds one market, so an ambiguous configuration is refused rather than guessed."""
    config = runnable(tmp_path)
    head, _, tail = config.read_text(encoding="utf-8").partition("[calibration]")
    second = head.replace('id = "BTC-DERIBIT"', 'id = "ETH-DERIBIT"').replace(
        'underlying = "BTC"', 'underlying = "ETH"'
    )
    config.write_text(f"{head}{second}[calibration]{tail}", encoding="utf-8")

    code, output = invoke(
        "record", "--config", str(config), "--recording", str(tmp_path / "session.jsonl")
    )

    assert code == 2
    assert "--market" in output


def test_replay_runs_the_recorded_session_and_exits_cleanly(
    session: tuple[Path, Path, str],
) -> None:
    """The pair, as a person uses them: record once, then replay from the file alone."""
    config, recording, _ = session

    code, output = invoke("replay", "--config", str(config), "--recording", str(recording))

    assert code == 0
    assert f"[BTC-DERIBIT / {PRODUCER_ID}]" in output


def test_replay_says_so_when_the_file_is_not_there(
    session: tuple[Path, Path, str], tmp_path: Path
) -> None:
    config, _, _ = session

    code, output = invoke(
        "replay", "--config", str(config), "--recording", str(tmp_path / "absent.jsonl")
    )

    assert code == 2
    assert "cannot read the recording" in output


def test_replay_says_so_when_the_file_is_not_a_recording(
    session: tuple[Path, Path, str], tmp_path: Path
) -> None:
    """A CSV renamed, which is the mistake that actually happens."""
    config, _, _ = session
    impostor = tmp_path / "report.jsonl"
    impostor.write_text("underlying,strike,vol\n", encoding="utf-8")

    code, output = invoke("replay", "--config", str(config), "--recording", str(impostor))

    assert code == 2
    assert "not readable" in output


def test_replay_refuses_a_recording_of_another_underlying(
    session: tuple[Path, Path, str], tmp_path: Path
) -> None:
    """Caught here rather than by the chain, which raises on the first quote inside a task."""
    _, recording, _ = session
    config = runnable(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            'underlying = "BTC"\nprovider', 'underlying = "ETH"\nprovider'
        ),
        encoding="utf-8",
    )

    code, output = invoke("replay", "--config", str(config), "--recording", str(recording))

    assert code == 2
    assert "holds BTC quotes" in output


def test_replay_refuses_a_count_of_zero(session: tuple[Path, Path, str]) -> None:
    config, recording, _ = session

    code, output = invoke(
        "replay", "--config", str(config), "--recording", str(recording), "--count", "0"
    )

    assert code == 2
    assert "--count must be positive" in output


def test_every_verb_the_design_names_is_declared() -> None:
    """``--help`` is the engine's shape as a person meets it."""
    _, output = invoke("--help")

    assert all(verb in output for verb in ("run", "report", "record", "replay"))
