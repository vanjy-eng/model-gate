#!/usr/bin/env python3
"""Summarise a mutmut run, and refuse to report a score it cannot support.

Two traps this exists to avoid, both of which caught the first version of the
CI job:

1. `mutmut run` can fail to import the package and do nothing at all. With
   `|| true` in the workflow, that reported green.
2. The run can be **cut off part-way** and still look like a finished run.
   `timeout 25m ... || true` in the workflow discarded exit 124, and this
   script compared the verdict count against a floor of 200 — so a run killed
   at 1221 of 1651 mutants reported a confident "32.2% kill rate" with no hint
   that a quarter of the surface was never touched. A partial score presented
   as a whole one is worse than no score, because it gets quoted.
3. `mutmut results` lists **only survivors**. Counting statuses from it and
   dividing yields a 0% kill rate whatever the truth, because a killed mutant
   never appears in that output.

So the counts come from the progress line mutmut prints during the run, which
is the only place the full tally appears:

    410/3430  🎉 179  🫥 100  ⏰ 0  🤔 0  🙁 131  🔇 0  🧙 0

Usage:
    mutmut run 2>&1 | tee mutation.log
    python scripts/mutation_report.py mutation.log [--min-tested N]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# mutmut 3 renders each status as an emoji in its progress line.
STATUS_ICONS = {
    "🎉": "killed",
    "🙁": "survived",
    "⏰": "timeout",
    "🤔": "suspicious",
    "🫥": "not covered",
    "🔇": "skipped",
    "🧙": "check was skipped",
}
# Statuses that mean the mutant genuinely ran and the suite gave a verdict.
VERDICTS = {"killed", "survived", "timeout", "suspicious"}

PAIR = re.compile(r"(" + "|".join(map(re.escape, STATUS_ICONS)) + r")\s*(\d+)")
TOTAL = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")


def parse(text: str) -> tuple[dict[str, int], int] | None:
    """Returns (counts, total_generated) from a mutmut run log, or None.

    mutmut redraws its progress line with carriage returns, so the whole run
    can arrive as a single line containing every intermediate tally. Matching
    "the last one" mixes groups across redraws — an earlier version of this
    script reported 410 processed alongside 3139 verdicts.

    The counts only ever increase during a run, so taking the maximum seen for
    each status is both correct and immune to how the terminal output is
    chunked.
    """
    counts: dict[str, int] = {}
    for icon, value in PAIR.findall(text):
        status = STATUS_ICONS[icon]
        counts[status] = max(counts.get(status, 0), int(value))
    if not counts:
        return None
    total = max((int(b) for _, b in TOTAL.findall(text)), default=0)
    return counts, total


def survivors() -> list[str]:
    """`mutmut results` lists survivors — useful for *where* to add assertions,
    just not for counting."""
    for command in ([sys.executable, "-m", "mutmut", "results"], ["mutmut", "results"]):
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if done.returncode == 0:
            return [line.strip() for line in done.stdout.splitlines() if "bdp_model_gate" in line]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path, help="output captured from `mutmut run`")
    parser.add_argument(
        "--min-tested",
        type=int,
        default=50,
        help="fail if fewer than this many mutants got a verdict (default 50)",
    )
    parser.add_argument(
        "--run-exit-status",
        type=int,
        default=None,
        help=(
            "exit status of the `mutmut run` that produced this log. 124 is "
            "`timeout` killing it; any non-zero value means the tally below is "
            "partial."
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="report a truncated run without failing (default: fail)",
    )
    parser.add_argument(
        "--min-kill-rate",
        type=float,
        default=None,
        help="optionally fail below this kill rate, as a fraction",
    )
    args = parser.parse_args()

    if not args.log.is_file():
        print(f"no such log: {args.log}", file=sys.stderr)
        return 1

    parsed = parse(args.log.read_text(errors="replace"))
    if parsed is None:
        print(
            "could not find a progress line in the log — `mutmut run` produced no\n"
            "tally, which usually means it never ran a mutant.",
            file=sys.stderr,
        )
        return 1

    counts, total = parsed
    tested = sum(n for status, n in counts.items() if status in VERDICTS)
    killed = counts.get("killed", 0)

    # Every status is one processed mutant, so this is exactly the left-hand
    # side of mutmut's own `1221/1651` progress counter. Anything missing was
    # never reached.
    processed = sum(counts.values())
    unreached = max(total - processed, 0)
    timed_out = args.run_exit_status == 124
    incomplete = bool(unreached) or bool(args.run_exit_status)

    print("Mutation testing")
    print("=" * 52)
    print(f"  mutants generated  {total}")
    for status, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        marker = " " if status in VERDICTS else "*"
        print(f"  {marker} {status:20} {n:6}")
    print("  * did not run, so not evidence either way")

    if unreached:
        print(f"  ! never reached     {unreached:6}")

    if tested:
        rate = killed / tested
        # Label the number at the point it is read. A partial kill rate quoted
        # without that word is the whole failure this guard exists for.
        partial = " (PARTIAL — see below)" if incomplete else ""
        print(f"\n  kill rate        {killed}/{tested} = {rate:.1%}{partial}")
    else:
        rate = 0.0

    by_module: dict[str, int] = {}
    for line in survivors():
        module = line.split(".x")[0].strip().rstrip(":")
        by_module[module] = by_module.get(module, 0) + 1
    if by_module:
        print("\n  survivors by module — where assertions are missing:")
        for module, n in sorted(by_module.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {n:5}  {module}")

    if incomplete and not args.allow_incomplete:
        if timed_out:
            why = (
                "the run was killed by its timeout — mutmut had processed "
                f"{processed} of {total} mutant(s)"
            )
        elif args.run_exit_status:
            why = f"`mutmut run` exited {args.run_exit_status}"
        else:
            why = f"only {processed} of {total} mutant(s) were processed"
        print(
            f"\nFAIL: {why}.\n"
            f"The kill rate above covers {processed / total:.0%} of the surface and is not "
            "a result for the whole of it.\nRaise the timeout, cut the surface, or pass "
            "--allow-incomplete to accept a partial run.",
            file=sys.stderr,
        )
        return 1

    if tested < args.min_tested:
        print(
            f"\nFAIL: only {tested} mutant(s) got a verdict, expected at least "
            f"{args.min_tested}. The run did no useful work.",
            file=sys.stderr,
        )
        return 1

    if args.min_kill_rate is not None and rate < args.min_kill_rate:
        print(
            f"\nFAIL: kill rate {rate:.1%} is below the floor of {args.min_kill_rate:.1%}.",
            file=sys.stderr,
        )
        return 1

    print(f"\nOK: {tested} mutant(s) got a real verdict, {processed}/{total} processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
