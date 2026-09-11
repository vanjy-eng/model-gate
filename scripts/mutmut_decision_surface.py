#!/usr/bin/env python3
"""Run mutmut over the **decision surface** only, and say what that means.

## Why not the whole codebase

Counted with mutmut's own operator table at 0.5.4, across the modules
`pyproject.toml` asks it to mutate:

    operator_arg_removal      6,034   48%
    operator_string           3,426   28%
    operator_assignment       1,114    9%
    operator_swap_op            706    6%   <-- decision surface
    operator_number             660    5%   <-- decision surface
    operator_keywords           253    2%   <-- decision surface
    operator_remove_unary_ops    81    1%   <-- decision surface
    everything else             162    1%
                             ------
                             12,436

**The 1,700 mutants that can produce a wrong verdict are 14% of the
population**, and the other 86% pushes the score in both directions at once:

- `arg_removal` drops each call argument, and 2,837 of the 3,593 argument
  positions in those modules are *required positionals*. Dropping one raises
  `TypeError`, so any test touching the line kills it. Free kills that prove
  nothing.
- `string` mutates prose detail strings. Killed or not depending on whether
  some test happens to assert on that substring, which says nothing about
  whether the verdict is right.

Which is why the trend was uninterpretable: 0.5.2 killed 286 *more* mutants
than 0.5.1 and scored 0.7 points *lower*. It was a ratio over a population
whose composition changed every release.

And the timed run already reached only a fraction of it — 4,185 of 7,004 at
0.5.2, so roughly a third of today's total, with *which* third decided by
where the clock stopped. **Focusing is not less coverage. It replaces an
arbitrary subset with a chosen one**, which is what makes a rate comparable
release to release and a floor safe to turn on.

Focusing did not, on its own, make it finish. The first version of this claim
said the pruned surface was "small enough to finish ... in roughly eleven
minutes"; at 0.6.0 it took ~34 minutes in CI against a 25-minute box, so every
run was cut off at about 74% and `|| true` hid it. The budget now comes from
the measured rate, and scripts/mutation_report.py fails rather than reporting a
partial tally as a whole one.

The four operators kept are the project's stated failure mode written as
mutations: a flipped comparison, a shifted threshold, an inverted boolean.
That *is* the confident-wrong-green-number.

## What this gives up

Questions like "would anything notice if `metadata=` were dropped?". A few of
those are real findings. They are a minority of a majority, and the timed run
was not answering them either — it ran out of clock first.

## What a survivor here does and does not mean

It means no test in `pytest_add_cli_args_test_selection` would notice. That
qualifier is load-bearing: when the selection was five hand-picked files, 305
of the 1,058 reported survivors were killed as soon as the rest of the suite
was allowed to run, and the kill rate moved 34.6% -> 54.4% over the same 1,651
mutants without a single new assertion being written. See the comment on that
setting in pyproject.toml.

## Usage

    python scripts/mutmut_decision_surface.py run --max-children 4
    python scripts/mutmut_decision_surface.py results

Any mutmut subcommand works; the operator table is pruned before mutmut
generates anything.
"""

from __future__ import annotations

import sys

#: The operators that can change a verdict.
DECISION_SURFACE = frozenset(
    {
        "operator_swap_op",  # > -> <, >= -> >, and -> or
        "operator_number",  # 0.5 -> 1.5, 30 -> 31
        "operator_keywords",  # True/False/None, break/continue
        "operator_remove_unary_ops",  # not x -> x
    }
)


def prune() -> tuple[int, int]:
    """Restrict mutmut's operator table. Returns (kept, dropped).

    Both module bindings have to be rewritten. `mutation/file_mutation.py`
    does `from mutmut.mutation.mutators import mutation_operators`, which
    binds the *list object's name* in its own namespace at import time — so
    patching only the source module leaves the consumer pointing at the
    original and the pruning silently does nothing. That is the kind of no-op
    this repository has been bitten by before, so the count is asserted below
    rather than assumed.
    """
    from mutmut.mutation import file_mutation, mutators

    original = list(mutators.mutation_operators)
    kept = [pair for pair in original if pair[1].__name__ in DECISION_SURFACE]

    unknown = DECISION_SURFACE - {op.__name__ for _, op in original}
    if unknown:
        raise SystemExit(
            f"mutmut {getattr(__import__('mutmut'), '__version__', '?')} has no operator(s) "
            f"named {sorted(unknown)} — the pin in pyproject.toml moved, and this script "
            "needs re-checking against the new table"
        )

    mutators.mutation_operators = kept
    file_mutation.mutation_operators = kept
    if len(file_mutation.mutation_operators) != len(kept):
        raise SystemExit("pruning the operator table did not take effect")
    return len(kept), len(original) - len(kept)


def main(argv: list[str] | None = None) -> int:
    kept, dropped = prune()
    print(
        f"mutating the decision surface only: kept {kept} operator(s) "
        f"({', '.join(sorted(DECISION_SURFACE))}), dropped {dropped}.",
        file=sys.stderr,
    )
    print(
        "A survivor here means a comparison, threshold or boolean could be "
        "wrong and no test in `pytest_add_cli_args_test_selection` would "
        "notice.",
        file=sys.stderr,
    )

    from mutmut.__main__ import cli

    try:
        cli(list(argv if argv is not None else sys.argv[1:]), standalone_mode=False)
    except SystemExit as exit_code:  # click raises this for --help and friends
        return int(exit_code.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
