"""Metamorphic tests: properties that must hold for *any* input.

Example-based tests only check the cases someone thought of. These check
relationships between two runs — permute the rows, rescale a feature, rename
the groups — where the answer must stay the same, or move in a known way.

That is how the perturbation-scale bug would have been caught: no example
test was wrong, but scale-invariance was violated, and no test asked.
"""

import numpy as np
import pandas as pd
import pytest

from bdp_model_gate import GateConfig, ModelGate, PerformanceConfig, StructuredGateContext
from bdp_model_gate.metrics import resolve_metric
from bdp_model_gate.structured import default_structured_checks
from bdp_model_gate.structured.fairness import DisparateImpactCheck, ProxyCorrelationCheck
from bdp_model_gate.structured.regression_fairness import GroupMeanGapCheck
from bdp_model_gate.structured.security import AdversarialRobustnessCheck


class Linear:
    """A deterministic scorer with a known dependence on each column."""

    def __init__(self, weights, cut=0.0):
        self.weights, self.cut = dict(weights), cut

    def _score(self, X):
        return sum(w * X[c].to_numpy() for c, w in self.weights.items())

    def predict(self, X):
        return (self._score(X) >= self.cut).astype(int)


@pytest.fixture
def frame():
    rng = np.random.default_rng(17)
    n = 400
    X = pd.DataFrame(
        {
            "income": rng.normal(50_000, 12_000, n),
            "age": rng.integers(21, 65, n).astype(float),
            "tenure": rng.exponential(40, n),
        }
    )
    protected = pd.DataFrame({"region": rng.choice(["Lagos", "Kano"], n)})
    y = (X["income"] > X["income"].median()).astype(int).to_numpy()
    return X, y, protected


def _context(X, y, protected, **kw):
    base = dict(
        model=Linear({"income": 1.0}, cut=50_000),
        X=X,
        y_true=y,
        y_pred=y.astype(float),
        protected_df=protected,
        task="binary",
    )
    base.update(kw)
    return StructuredGateContext(**base)


def _flags(report):
    """A comparable fingerprint of a run: which check produced which flag."""
    return sorted((r.check_name, r.flag) for r in report.results)


# --- permutation -------------------------------------------------------------


def test_row_order_does_not_change_the_verdict(frame):
    """Rows are observations, not a sequence. Shuffling them consistently must
    not change any conclusion."""
    X, y, protected = frame
    config = GateConfig(performance=PerformanceConfig(metric="accuracy", min_score=0.0))
    checks = lambda: default_structured_checks(config, include_plugins=False)  # noqa: E731

    original = ModelGate(checks=checks()).run(_context(X, y, protected))

    order = np.random.default_rng(0).permutation(len(X))
    shuffled = ModelGate(checks=checks()).run(
        _context(
            X.iloc[order].reset_index(drop=True),
            y[order],
            protected.iloc[order].reset_index(drop=True),
        )
    )

    assert _flags(original) == _flags(shuffled)
    assert original.gate_status == shuffled.gate_status
    assert original.model_score == pytest.approx(shuffled.model_score)


def test_group_relabelling_does_not_change_disparity_magnitude(frame):
    """ "Lagos"/"Kano" carry no meaning to the metric. Renaming them must move
    nothing but the labels in the output."""
    pytest.importorskip("fairlearn")
    X, y, protected = frame
    renamed = protected.replace({"Lagos": "north", "Kano": "south"})

    a = DisparateImpactCheck().run(_context(X, y, protected))[0]
    b = DisparateImpactCheck().run(_context(X, y, renamed))[0]

    assert a.metadata["demographic_parity_diff"] == pytest.approx(
        b.metadata["demographic_parity_diff"]
    )
    assert a.flag == b.flag


# --- scale invariance --------------------------------------------------------


@pytest.mark.parametrize("factor", [1e-3, 1e3, 1e6])
def test_feature_rescaling_does_not_change_the_flip_rate(frame, factor):
    """Multiplying a feature by a constant is a change of units, not of the
    model's behaviour — the decision boundary moves with it.

    This is the invariant the adversarial check violated: a perturbation
    scaled off the mean across *all* columns made the flip rate depend on
    whether a column was measured in naira or millions of naira.
    """
    X, y, protected = frame
    model = Linear({"income": 1.0}, cut=50_000)
    scaled_model = Linear({"income": 1.0}, cut=50_000 * factor)
    X_scaled = X.assign(income=X["income"] * factor)

    # Score every row: the subsample is content-addressed, so two frames that
    # differ by a scale factor would otherwise draw different rows and the
    # comparison would measure sampling variance rather than the perturbation.
    check = AdversarialRobustnessCheck(n_samples=len(X))
    base = check.run(_context(X, y, protected, model=model))[0]
    scaled = check.run(_context(X_scaled, y, protected, model=scaled_model))[0]

    assert base.metadata["flip_rate"] == pytest.approx(scaled.metadata["flip_rate"], abs=0.02)


@pytest.mark.parametrize("factor", [1e-3, 1e3])
def test_proxy_correlation_is_scale_free(frame, factor):
    """eta^2 is a variance ratio, so it cannot depend on units."""
    X, y, protected = frame
    base = ProxyCorrelationCheck(GateConfig().fairness).run(_context(X, y, protected))
    scaled = ProxyCorrelationCheck(GateConfig().fairness).run(
        _context(X.assign(income=X["income"] * factor), y, protected)
    )
    assert [r.flag for r in base] == [r.flag for r in scaled]


def test_regression_mean_gap_is_scale_free(frame):
    """The gap is relative to the overall mean, so scaling the target cancels."""
    X, y, protected = frame
    y_pred = X["income"].to_numpy()

    def gap_for(scale):
        return (
            GroupMeanGapCheck()
            .run(
                StructuredGateContext(
                    model=Linear({"income": 1.0}),
                    X=X,
                    y_true=y_pred * scale,
                    y_pred=y_pred * scale,
                    protected_df=protected,
                    task="regression",
                )
            )[0]
            .metadata["relative_gap"]
        )

    assert gap_for(1.0) == pytest.approx(gap_for(1000.0), abs=1e-9)


def test_shap_gap_is_scale_free(frame):
    """The reason shap_gap_threshold became relative in 0.4.2: an absolute
    threshold in target units flags nothing on one scale and everything on
    another."""
    pytest.importorskip("shap")
    from sklearn.ensemble import GradientBoostingRegressor

    from bdp_model_gate.structured.fairness import ShapSubgroupCheck

    X, y, protected = frame
    target = X["income"].to_numpy() + 3_000 * (protected["region"] == "Kano").to_numpy()

    def gaps_for(scale):
        model = GradientBoostingRegressor(random_state=0, n_estimators=30).fit(X, target * scale)
        results = ShapSubgroupCheck().run(
            StructuredGateContext(
                model=model,
                X=X,
                y_true=target * scale,
                y_pred=model.predict(X),
                protected_df=protected,
                task="regression",
            )
        )
        return sorted(r.metadata.get("relative_gap", 0.0) for r in results if not r.is_ok)

    small, large = gaps_for(1.0), gaps_for(1e5)
    assert len(small) == len(large)
    for a, b in zip(small, large):
        assert a == pytest.approx(b, rel=0.02)


# --- metric behaviour under transformation -----------------------------------


def test_scaling_the_target_scales_error_metrics_and_leaves_r2_alone():
    """rmse and mae are in target units, so they scale by k. r2 is a variance
    ratio, so it does not. Getting this backwards would make every regression
    threshold meaningless."""
    rng = np.random.default_rng(4)
    y_true = rng.normal(100, 20, 300)
    y_pred = y_true + rng.normal(0, 5, 300)
    k = 37.0

    for name in ("rmse", "mae"):
        fn = resolve_metric(name, "regression").fn
        assert fn(y_true * k, y_pred * k) == pytest.approx(fn(y_true, y_pred) * k)

    r2 = resolve_metric("r2", "regression").fn
    assert r2(y_true * k, y_pred * k) == pytest.approx(r2(y_true, y_pred))

    # MAPE is a ratio of errors to actuals, so it is scale-free too.
    mape = resolve_metric("mape", "regression").fn
    assert mape(y_true * k, y_pred * k) == pytest.approx(mape(y_true, y_pred))


def test_class_relabelling_does_not_change_ordinal_metrics():
    """Ordinal metrics depend on rank, not on what the ranks are called."""
    from bdp_model_gate.metrics import ordinal_mae, quadratic_kappa

    order = ["low", "mid", "high"]
    renamed = ["D", "R", "A"]
    truth = ["high", "mid", "low", "high", "mid"]
    pred = ["mid", "mid", "low", "high", "low"]
    swap = dict(zip(order, renamed))

    assert ordinal_mae(truth, pred, order) == pytest.approx(
        ordinal_mae([swap[v] for v in truth], [swap[v] for v in pred], renamed)
    )
    assert quadratic_kappa(truth, pred, order) == pytest.approx(
        quadratic_kappa([swap[v] for v in truth], [swap[v] for v in pred], renamed)
    )


# --- monotonicity ------------------------------------------------------------


def test_making_a_model_strictly_more_unfair_never_lowers_the_disparity(frame):
    """A disparity metric that can fall as the model gets worse is not
    measuring disparity."""
    pytest.importorskip("fairlearn")
    X, y, protected = frame
    is_kano = (protected["region"] == "Kano").to_numpy()
    rng = np.random.default_rng(5)
    base = rng.random(len(X)) < 0.5

    previous = -1.0
    for skew in (0.0, 0.25, 0.5, 0.75, 1.0):
        # Progressively deny more of one group while leaving the other alone.
        denied = is_kano & (rng.random(len(X)) < skew)
        y_pred = np.where(denied, 0, base.astype(int))
        gap = (
            DisparateImpactCheck()
            .run(
                StructuredGateContext(
                    model=Linear({"income": 1.0}),
                    X=X,
                    y_true=y,
                    y_pred=y_pred,
                    protected_df=protected,
                    task="binary",
                )
            )[0]
            .metadata["demographic_parity_diff"]
        )
        assert gap >= previous - 0.05, f"disparity fell as the model got more unfair at skew={skew}"
        previous = gap


def test_adding_noise_never_lowers_the_error(frame):
    """A monotone relationship the error metrics must obey."""
    rng = np.random.default_rng(6)
    y_true = rng.normal(100, 20, 400)
    rmse = resolve_metric("rmse", "regression").fn

    previous = -1.0
    for sigma in (0.0, 1.0, 5.0, 20.0):
        value = rmse(y_true, y_true + rng.normal(0, sigma, 400))
        assert value >= previous
        previous = value


# --- the sampling primitive itself -------------------------------------------


def test_stable_sample_is_order_independent(frame):
    """The property the whole permutation-invariance result rests on."""
    from bdp_model_gate._sampling import stable_sample

    X, _, _ = frame
    order = np.random.default_rng(11).permutation(len(X))
    shuffled = X.iloc[order].reset_index(drop=True)

    a = stable_sample(X, 50).reset_index(drop=True).sort_values(list(X.columns))
    b = stable_sample(shuffled, 50).reset_index(drop=True).sort_values(list(X.columns))

    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_stable_sample_respects_the_seed_and_the_size(frame):
    from bdp_model_gate._sampling import stable_sample

    X, _, _ = frame
    assert len(stable_sample(X, 50)) == 50
    assert len(stable_sample(X, len(X) + 10)) == len(X)  # never over-draws

    a = stable_sample(X, 50, random_state=1)
    b = stable_sample(X, 50, random_state=2)
    assert not a.equals(b), "random_state must still vary the selection"

    # And it stays reproducible for a fixed seed.
    pd.testing.assert_frame_equal(a, stable_sample(X, 50, random_state=1))


# --- validation methodology (0.5.2) ------------------------------------------


def test_leakage_power_is_invariant_to_a_monotone_rescaling(frame):
    """A leak measured in naira is the same leak measured in thousands.

    Rank AUC depends only on ordering, so any strictly increasing transform of
    a feature must leave its power untouched. A power that moved with the
    units would make `leakage_min_power` mean something different per column.
    """
    from bdp_model_gate.structured.validation_checks import LeakageCheck

    X, y, _ = frame
    leaky = X.assign(settled=y * 1000.0 + 7.0)
    baseline = LeakageCheck().run(_context(leaky, y, None))

    rescaled = leaky.assign(settled=leaky["settled"] / 1000.0 + 12.0)
    after = LeakageCheck().run(_context(rescaled, y, None))

    assert [r.metadata.get("feature_power") for r in baseline] == [
        r.metadata.get("feature_power") for r in after
    ]


def test_leakage_verdict_does_not_depend_on_row_order(frame):
    from bdp_model_gate.structured.validation_checks import LeakageCheck

    X, y, protected = frame
    leaky = X.assign(settled=y * 1000.0)
    order = np.random.default_rng(4).permutation(len(X))

    before = LeakageCheck().run(_context(leaky, y, protected))
    after = LeakageCheck().run(
        _context(
            leaky.iloc[order].reset_index(drop=True),
            y[order],
            protected.iloc[order].reset_index(drop=True),
        )
    )
    assert [(r.flag, r.metadata.get("feature")) for r in before] == [
        (r.flag, r.metadata.get("feature")) for r in after
    ]


def test_split_overlap_is_measured_by_content_not_position(frame):
    """Shuffling either frame must not change how many rows they share —
    the same property `stable_sample` exists to guarantee elsewhere."""
    from bdp_model_gate.structured.validation_checks import SplitOverlapCheck

    X, y, protected = frame
    train, live = X.iloc[:200], X.iloc[100:300].reset_index(drop=True)  # 100 shared
    rng = np.random.default_rng(8)

    def overlap(train_frame, live_frame, labels, groups):
        results = SplitOverlapCheck().run(_context(live_frame, labels, groups, X_train=train_frame))
        return next(
            r.metadata["n_overlapping"]
            for r in results
            if r.metadata["check"] == "overlap_with_training"
        )

    straight = overlap(train, live, y[100:300], protected.iloc[100:300].reset_index(drop=True))
    order = rng.permutation(len(live))
    shuffled = overlap(
        train.sample(frac=1.0, random_state=2),
        live.iloc[order].reset_index(drop=True),
        y[100:300][order],
        protected.iloc[100:300].reset_index(drop=True).iloc[order].reset_index(drop=True),
    )
    assert straight == shuffled == 100


def test_drift_is_symmetric_in_magnitude(frame):
    """Swapping the two frames must not change *whether* a feature drifted.

    The numeric measure is standardised by the training spread, so the two
    directions can differ slightly in size — but a shift that clears the
    threshold one way must not vanish the other, or the verdict would depend
    on which frame you happened to call training.
    """
    from bdp_model_gate.structured.validation_checks import FeatureDriftCheck

    X, y, protected = frame
    shifted = X.assign(income=X["income"] + 3 * X["income"].std())

    forward = FeatureDriftCheck().run(_context(shifted, y, protected, X_train=X))
    backward = FeatureDriftCheck().run(_context(X, y, protected, X_train=shifted))

    assert (
        {r.metadata.get("feature") for r in forward if r.flag == "DRIFT_RISK"}
        == {r.metadata.get("feature") for r in backward if r.flag == "DRIFT_RISK"}
        == {"income"}
    )


def test_renaming_a_feature_does_not_change_whether_it_leaks(frame):
    """The check reads distributions, not names. A column called `target_copy`
    and one called `x7` must be judged identically."""
    from bdp_model_gate.structured.validation_checks import LeakageCheck

    X, y, _ = frame
    leaky = X.assign(obviously_the_answer=y * 1.0)
    renamed = leaky.rename(columns={"obviously_the_answer": "x7"})

    before = LeakageCheck().run(_context(leaky, y, None))
    after = LeakageCheck().run(_context(renamed, y, None))
    assert [r.flag for r in before] == [r.flag for r in after]
    assert before[0].metadata["feature_power"] == after[0].metadata["feature_power"]


# --- exposure and the actuarial measures (0.5.3) -----------------------------


def _pricing(X, protected, y_pred, y_true, **kw):
    return StructuredGateContext(
        model=Linear({"income": 1.0}),
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        protected_df=protected,
        task="regression",
        **kw,
    )


def test_a_uniform_exposure_column_is_a_no_op(frame):
    """The single most important property of the exposure work: a book where
    every policy ran the full year must get byte-identical numbers to a book
    with no exposure column at all.

    `weights_or_ones` is what guarantees it — the weighted path is the only
    path — and this is what would catch a second, unweighted code path being
    introduced later.
    """
    from bdp_model_gate.structured.actuarial_checks import (
        ActualVsExpectedCheck,
        RiskDiscriminationCheck,
    )
    from bdp_model_gate.structured.regression_fairness import (
        CalibrationParityCheck,
        ErrorParityCheck,
        GroupMeanGapCheck,
    )

    X, y, protected = frame
    y_pred = X["income"].to_numpy()
    y_true = y_pred * 0.95 + 500.0
    without = _pricing(X, protected, y_pred, y_true)
    uniform = _pricing(X, protected, y_pred, y_true, exposure=np.ones(len(X)) * 7.0)

    for check in (
        ActualVsExpectedCheck(),
        RiskDiscriminationCheck(),
        GroupMeanGapCheck(),
        ErrorParityCheck(),
        CalibrationParityCheck(),
    ):
        plain = check.run(without)
        weighted = check.run(uniform)
        assert [r.flag for r in plain] == [r.flag for r in weighted], check.name
        for left, right in zip(plain, weighted):
            for key, value in left.metadata.items():
                if key in ("exposure_weighted", "bands"):
                    continue  # the report says it was weighted; that is the point
                assert right.metadata[key] == value, f"{check.name}.{key}"


def test_the_exposure_weighted_metrics_reduce_to_the_unweighted_ones():
    """Same property, one level down: `sample_weight` of a constant must not
    move a metric. An off-by-one in the weighted denominator would show here
    and nowhere else."""
    from bdp_model_gate.metrics import resolve_metric

    rng = np.random.default_rng(21)
    y_true = rng.gamma(2.0, 300.0, 250)
    y_pred = y_true * rng.uniform(0.7, 1.3, 250)

    for name in ("rmse", "mae", "mape", "r2", "lorenz_gini"):
        plain = resolve_metric(name, "regression").fn(y_true, y_pred)
        weighted = resolve_metric(name, "regression", exposure=np.full(250, 4.0)).fn(y_true, y_pred)
        assert weighted == pytest.approx(plain), name


def test_rescaling_the_exposure_unit_does_not_change_a_verdict(frame):
    """Exposure in months and the same exposure in years are the same book.
    The weights are relative, so a verdict that moved on the unit is a bug."""
    from bdp_model_gate.structured.actuarial_checks import ActualVsExpectedCheck

    X, _, protected = frame
    y_pred = X["income"].to_numpy()
    y_true = y_pred * 1.08
    exposure = np.clip(X["tenure"].to_numpy() / 40.0, 0.05, 1.0)

    years = ActualVsExpectedCheck().run(_pricing(X, protected, y_pred, y_true, exposure=exposure))
    months = ActualVsExpectedCheck().run(
        _pricing(X, protected, y_pred, y_true, exposure=exposure * 12.0)
    )
    assert [r.flag for r in years] == [r.flag for r in months]
    assert years[0].metadata["ae"] == pytest.approx(months[0].metadata["ae"])


def test_the_actual_over_expected_verdict_does_not_depend_on_row_order(frame):
    """A/E is a ratio of totals and the bands are content-derived, so sorting
    the validation CSV must not move the finding."""
    from bdp_model_gate.structured.actuarial_checks import ActualVsExpectedCheck

    X, _, protected = frame
    rng = np.random.default_rng(31)
    order = rng.permutation(len(X))
    y_pred = X["income"].to_numpy()
    y_true = y_pred * rng.uniform(0.8, 1.3, len(X))
    exposure = np.clip(X["tenure"].to_numpy() / 40.0, 0.05, 1.0)

    straight = ActualVsExpectedCheck().run(
        _pricing(X, protected, y_pred, y_true, exposure=exposure)
    )
    shuffled = ActualVsExpectedCheck().run(
        _pricing(
            X.iloc[order].reset_index(drop=True),
            protected.iloc[order].reset_index(drop=True),
            y_pred[order],
            y_true[order],
            exposure=exposure[order],
        )
    )
    assert [r.flag for r in straight] == [r.flag for r in shuffled]
    assert straight[0].metadata["ae"] == pytest.approx(shuffled[0].metadata["ae"])
    assert [b["ae"] for b in straight[1].metadata["bands"]] == pytest.approx(
        [b["ae"] for b in shuffled[1].metadata["bands"]]
    )
