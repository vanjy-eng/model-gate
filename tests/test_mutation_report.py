"""The mutation reporter, checked against the run it failed to catch.

`scripts/mutation_report.py` exists to stop the mutation job reporting a
number it cannot support. It has been wrong twice, in the same direction both
times — reporting green over a run that had not done the work:

1. It counted statuses from `mutmut results`, which lists only survivors, and
   so computed a 0% kill rate whatever the truth.
2. It compared the verdict count against a floor and nothing else, so a run
   killed by `timeout` at 1221 of 1651 mutants reported "32.2% kill rate" with
   no indication that a quarter of the surface was never touched.

The second one shipped and sat green in CI. These tests pin the arithmetic
against that exact log so it cannot come back.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "mutation_report.py"

#: The real progress line from CI run 34435174771, the truncated one. 1221 of
#: 1651 processed: 382 killed + 805 survived + 34 not covered.
TRUNCATED = "⠦ 1221/1651  🎉 382 🫥 34  ⏰ 0  🤔 0  🙁 805  🔇 0  🧙 0"

#: The same run had it been allowed to finish: every mutant accounted for.
COMPLETE = "⠦ 1651/1651  🎉 600 🫥 34  ⏰ 0  🤔 0  🙁 1017  🔇 0  🧙 0"


@pytest.fixture
def report(tmp_path):
    """Run the reporter over a log holding `log_text`, and hand back the
    finished process. Invoked as a subprocess rather than by importing `main`,
    because the exit status is the thing CI acts on."""

    def run(log_text: str, *args: str) -> subprocess.CompletedProcess:
        log = tmp_path / "mutation.log"
        log.write_text(log_text, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(log), *args],
            capture_output=True,
            text=True,
            cwd=REPO,
        )

    return run


def test_a_truncated_run_fails_even_though_it_did_plenty_of_work(report):
    """The regression. 1187 mutants got a verdict, which is far above any
    sensible `--min-tested` floor, so every count-based check passes — and the
    run was still cut off at 74%."""
    done = report(TRUNCATED, "--min-tested", "200", "--run-exit-status", "124")
    assert done.returncode == 1, done.stdout
    assert "never reached      430" in done.stdout.replace("   ", "  ")
    assert "killed by its timeout" in done.stderr
    assert "1221 of 1651" in done.stderr


def test_the_kill_rate_of_a_truncated_run_is_labelled_where_it_is_read(report):
    """A partial rate quoted without the word "partial" is how the number got
    into a release summary in the first place. The label goes on the same line
    as the number, not three lines below it."""
    done = report(TRUNCATED, "--min-tested", "200", "--run-exit-status", "124")
    rate_line = next(line for line in done.stdout.splitlines() if "kill rate" in line)
    assert "32.2%" in rate_line
    assert "PARTIAL" in rate_line


def test_truncation_is_caught_from_the_tally_alone(report):
    """Belt and braces: even with no exit status to go on — an older log, or a
    runner that lost it — 1221 processed against 1651 generated is arithmetic
    the script can do by itself."""
    done = report(TRUNCATED, "--min-tested", "200")
    assert done.returncode == 1
    assert "only 1221 of 1651" in done.stderr


def test_a_complete_run_passes(report):
    """The guard has to let the good case through, or it just gets disabled."""
    done = report(COMPLETE, "--min-tested", "200", "--run-exit-status", "0")
    assert done.returncode == 0, done.stderr
    assert "1651/1651 processed" in done.stdout
    assert "PARTIAL" not in done.stdout


def test_a_nonzero_exit_fails_even_when_every_mutant_was_processed(report):
    """`mutmut run` crashing after the last mutant still means the run is not
    trustworthy, and the tally alone cannot see that."""
    done = report(COMPLETE, "--min-tested", "200", "--run-exit-status", "1")
    assert done.returncode == 1
    assert "exited 1" in done.stderr


def test_a_partial_run_can_be_accepted_deliberately(report):
    """An explicit opt-out exists so the answer to a red job is never to delete
    the check."""
    done = report(
        TRUNCATED, "--min-tested", "200", "--run-exit-status", "124", "--allow-incomplete"
    )
    assert done.returncode == 0, done.stderr
    assert "PARTIAL" in done.stdout


def test_an_empty_log_still_fails(report):
    """The original bug: `mutmut run` importing nothing and doing nothing."""
    done = report("no progress line here", "--min-tested", "200")
    assert done.returncode == 1
    assert "never ran a mutant" in done.stderr
