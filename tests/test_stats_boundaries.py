"""The guard clauses in `stats.py`, exercised at their boundaries.

Every test here was written against a specific mutant that survived
`scripts/mutmut_decision_surface.py`. These are small functions, and the
suite tested them where they are well behaved: several groups, both classes
present, a healthy spread of values. The guard clauses -- `n_pos == 0`,
`counts > 0`, `ss_total <= 0`, `std() == 0` -- were only ever reached from
one side.

That matters more here than the size of the functions suggests. `stats.py`
exists so that leakage detection and proxy correlation work on a **core
install**, without scikit-learn. A guard that returns the wrong thing at its
own edge does not raise; it returns a number, and the number goes in the
report.

Mutants deliberately not covered here, because they are equivalent rather
than unkilled:

- `average_ranks`: `stop - start > 1` to `>= 1` or to `stop + start > 1`. A
  singleton run spanning `[s, s+1)` averages to `(s + s+1 + 1) / 2 == s + 1`,
  which is the ordinal rank it already had, so applying the averaging to
  singletons changes nothing.
- `benjamini_hochberg`: the upper clip bound, `1.0` to `2.0`. The largest
  p-value scales by `n / n`, and the adjustment is a cumulative minimum from
  there down, so no adjusted value can exceed 1 before the clip.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from bdp_model_gate.stats import (
    correlation_ratio,
    pearson_r,
    rank_auc,
    selection_rate_difference,
)

# --------------------------------------------------------------------------
# rank_auc: AUC is undefined with *no* positives, not with *one*
# --------------------------------------------------------------------------


def test_auc_is_defined_with_a_single_positive():
    """A rare-event model is the normal case in this domain, not an edge one.

    The guard exists for "one class is absent". Moving it to `n_pos == 1`
    makes the AUC vanish exactly when the positive class is rarest, which is
    when the leakage check matters most -- and it returns NaN rather than
    raising, so the check would report "not applicable" and the gate would
    stay green.
    """
    y = np.array([0, 0, 0, 0, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
    assert rank_auc(y, scores) == pytest.approx(1.0)


def test_auc_is_defined_with_a_single_negative():
    """The mirror of the above; a separate branch of the same `or`."""
    y = np.array([1, 1, 1, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    assert rank_auc(y, scores) == pytest.approx(1.0)


def test_a_single_class_yields_nan_without_dividing_by_zero():
    """`or`, not `and`: either class being absent makes AUC undefined.

    Under `and` the guard only fires when *both* are absent, so a
    single-class input reaches `n_pos * n_neg == 0` and divides by zero. The
    answer is still NaN, which is why no assertion on the return value
    catches it -- the difference is the RuntimeWarning on the way there, and
    a numpy warning inside a bootstrap draw is exactly what
    `bdp_model_gate.uncertainty` treats as "this draw is not evidence".
    """
    y = np.zeros(8, dtype=int)
    scores = np.linspace(0, 1, 8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(rank_auc(y, scores))


# --------------------------------------------------------------------------
# Group counting: a group of one is still a group
# --------------------------------------------------------------------------


def test_a_group_of_one_counts_towards_the_selection_rate_spread():
    """`counts > 0`, not `counts > 1`.

    Small groups are the ones disparity analysis is about. Dropping them
    silently narrows the measured spread -- here from 0.5 to 0.0, turning a
    finding into a clean bill of health.
    """
    y_pred = np.array([1, 0, 1, 0, 1])
    groups = np.array(["a", "a", "b", "b", "c"])  # c has one member, rate 1.0
    assert selection_rate_difference(y_pred, groups) == pytest.approx(0.5)


def test_a_group_of_one_counts_towards_the_correlation_ratio():
    """The same rule in the proxy screen. A one-member group carries variance
    between groups like any other, and excluding it changes eta-squared."""
    values = pd.Series([1.0, 1.0, 1.0, 9.0])
    groups = pd.Series(["a", "a", "a", "b"])
    assert correlation_ratio(values, groups) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# correlation_ratio: the degenerate inputs
# --------------------------------------------------------------------------


def test_an_empty_feature_is_zero_not_one():
    """0.0 means "this feature tells you nothing about the group", which is
    the right answer for no data. 1.0 would mean the opposite -- a perfect
    proxy -- and would flag every empty slice as a finding."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert correlation_ratio(pd.Series([], dtype=float), pd.Series([], dtype=object)) == 0.0


def test_a_constant_feature_is_zero_not_one():
    """`ss_total <= 0`, not `< 0`: a constant feature has exactly zero total
    variance, so `<` lets it through to a 0/0 division. And the value
    returned there must be 0.0 -- a column that never varies cannot encode
    anything about group membership."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ratio = correlation_ratio(pd.Series([3.0, 3.0, 3.0, 3.0]), pd.Series(list("aabb")))
    assert ratio == 0.0


def test_a_feature_with_small_variance_is_still_measured():
    """The guard is against *zero* variance, not small variance.

    Mutating `ss_total <= 0` to `<= 1` silently returns 0.0 for any feature
    whose total sum of squares is under 1 -- which is a unit choice, not a
    property of the data. This feature is a perfect proxy (eta-squared 1.0)
    and has `ss_total` of 0.04.
    """
    values = pd.Series([0.0, 0.2, 0.0, 0.2])
    groups = pd.Series(["a", "b", "a", "b"])
    assert correlation_ratio(values, groups) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# pearson_r: "constant" means zero variance, not unit variance
# --------------------------------------------------------------------------


def test_a_standardised_vector_is_not_mistaken_for_a_constant_one():
    """`std() == 0` is the constant-column guard. Mutated to `std() == 1` it
    returns 0.0 for any already-standardised column -- and standardising
    before a correlation is a normal thing for a caller to have done, so this
    would read as "no association" for precisely the inputs most likely to be
    passed in."""
    a = np.array([1.0, 2.0])
    b = np.array([-1.0, 1.0])  # mean 0, population std exactly 1.0
    assert np.std(b) == 1.0, "fixture no longer has unit variance"
    assert pearson_r(a, b) == pytest.approx(1.0)
