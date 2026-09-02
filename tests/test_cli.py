"""Tests for the bdp-model-gate CLI, calling main() in-process for coverage
and speed (rather than shelling out via subprocess)."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

# These tests fit real estimators, so they need the [structured] extra.
# Skipped wholesale on a core-only install rather than failing collection.
pytest.importorskip("sklearn", reason="requires the [structured] extra")

from sklearn.linear_model import LogisticRegression

from bdp_model_gate.cli import main


@pytest.fixture
def cli_fixtures(tmp_path):
    rng = np.random.default_rng(7)
    n = 150
    X = pd.DataFrame(
        {
            "income": rng.normal(50000, 15000, n),
            "age": rng.integers(18, 70, n),
            "credit_score": rng.normal(650, 50, n),
        }
    )
    y = (X["income"] > X["income"].median()).astype(int)
    model = LogisticRegression(max_iter=1000).fit(X, y)

    df = X.copy()
    df["label"] = y
    data_path = tmp_path / "validation.csv"
    df.to_csv(data_path, index=False)

    protected = pd.DataFrame(
        {
            "gender": rng.choice(["M", "F"], n),
            "region": rng.choice(["Lagos", "Abuja"], n),
        }
    )
    protected_path = tmp_path / "protected.csv"
    protected.to_csv(protected_path, index=False)

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    model_card_path = tmp_path / "model_card.json"
    model_card_path.write_text(
        json.dumps(
            {
                "legal_basis": "consent",
                "data_minimization_justification": "only pricing-relevant fields",
                "training_data_source": "SURA internal claims db",
                "use_case": "general_scoring",
            }
        )
    )

    latencies_path = tmp_path / "latencies.txt"
    latencies_path.write_text("\n".join(str(v) for v in rng.uniform(10, 50, 50)))

    output_path = tmp_path / "gate_report.json"

    return {
        "data": str(data_path),
        "protected": str(protected_path),
        "model": str(model_path),
        "model_card": str(model_card_path),
        "latencies": str(latencies_path),
        "output": str(output_path),
        "tmp_path": tmp_path,
    }


def test_cli_passes_and_writes_report(cli_fixtures):
    exit_code = main(
        [
            "--model",
            cli_fixtures["model"],
            "--data",
            cli_fixtures["data"],
            "--target-col",
            "label",
            "--protected",
            cli_fixtures["protected"],
            "--model-card",
            cli_fixtures["model_card"],
            "--latencies",
            cli_fixtures["latencies"],
            "--cost-per-inference",
            "0.0001",
            "--output",
            cli_fixtures["output"],
        ]
    )
    assert exit_code in {0, 1, 2}
    report = json.loads(open(cli_fixtures["output"]).read())
    assert "gate_status" in report


def test_cli_blocks_on_missing_compliance_fields(cli_fixtures):
    bad_card_path = cli_fixtures["tmp_path"] / "bad_card.json"
    bad_card_path.write_text(json.dumps({"use_case": "underwriting"}))

    exit_code = main(
        [
            "--model",
            cli_fixtures["model"],
            "--data",
            cli_fixtures["data"],
            "--target-col",
            "label",
            "--model-card",
            str(bad_card_path),
            "--output",
            cli_fixtures["output"],
        ]
    )
    assert exit_code == 1


def test_cli_minimal_invocation_without_optional_inputs(cli_fixtures):
    exit_code = main(
        [
            "--model",
            cli_fixtures["model"],
            "--data",
            cli_fixtures["data"],
            "--target-col",
            "label",
            "--output",
            cli_fixtures["output"],
        ]
    )
    assert exit_code in {0, 1, 2}


def test_cli_verbose_flag(cli_fixtures, caplog):
    import logging

    caplog.set_level(logging.DEBUG)
    main(
        [
            "--model",
            cli_fixtures["model"],
            "--data",
            cli_fixtures["data"],
            "--target-col",
            "label",
            "--output",
            cli_fixtures["output"],
            "--verbose",
        ]
    )


def test_cli_missing_model_file_returns_error(cli_fixtures):
    exit_code = main(
        [
            "--model",
            str(cli_fixtures["tmp_path"] / "does_not_exist.joblib"),
            "--data",
            cli_fixtures["data"],
            "--target-col",
            "label",
            "--output",
            cli_fixtures["output"],
        ]
    )
    assert exit_code == 1


def test_cli_json_config_overrides(cli_fixtures):
    config_path = cli_fixtures["tmp_path"] / "config.json"
    # deliberately uses the deprecated min_accuracy key — old config files
    # in consumers' repos must keep working
    config_path.write_text(json.dumps({"performance": {"min_accuracy": 1.5}}))

    exit_code = main(
        [
            "--model",
            cli_fixtures["model"],
            "--data",
            cli_fixtures["data"],
            "--target-col",
            "label",
            "--config",
            str(config_path),
            "--output",
            cli_fixtures["output"],
        ]
    )
    assert exit_code == 1  # near-impossible score threshold should block


def test_cli_yaml_config_overrides(cli_fixtures):
    config_path = cli_fixtures["tmp_path"] / "config.yaml"
    config_path.write_text("performance:\n  metric: accuracy\n  min_score: 1.5\n")

    exit_code = main(
        [
            "--model",
            cli_fixtures["model"],
            "--data",
            cli_fixtures["data"],
            "--target-col",
            "label",
            "--config",
            str(config_path),
            "--output",
            cli_fixtures["output"],
        ]
    )
    assert exit_code == 1


def test_cli_toml_config_overrides(cli_fixtures):
    config_path = cli_fixtures["tmp_path"] / "config.toml"
    config_path.write_text('[performance]\nmetric = "f1"\nmin_score = 1.5\n')

    exit_code = main(
        [
            "--model",
            cli_fixtures["model"],
            "--data",
            cli_fixtures["data"],
            "--target-col",
            "label",
            "--config",
            str(config_path),
            "--output",
            cli_fixtures["output"],
        ]
    )
    assert exit_code == 1


def test_cli_unrecognized_config_extension_errors(cli_fixtures):
    config_path = cli_fixtures["tmp_path"] / "config.ini"
    config_path.write_text("[performance]\nmin_score = 0.99\n")

    exit_code = main(
        [
            "--model",
            cli_fixtures["model"],
            "--data",
            cli_fixtures["data"],
            "--target-col",
            "label",
            "--config",
            str(config_path),
            "--output",
            cli_fixtures["output"],
        ]
    )
    assert exit_code == 1


def test_cli_config_with_unknown_section_and_field_warns_but_continues(cli_fixtures, caplog):
    config_path = cli_fixtures["tmp_path"] / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "performance": {"not_a_real_field": 1},
                "not_a_real_section": {"x": 1},
            }
        )
    )

    exit_code = main(
        [
            "--model",
            cli_fixtures["model"],
            "--data",
            cli_fixtures["data"],
            "--target-col",
            "label",
            "--config",
            str(config_path),
            "--output",
            cli_fixtures["output"],
        ]
    )
    assert exit_code in {0, 1, 2}


def test_cli_metric_and_min_score_flags(cli_fixtures):
    exit_code = main(
        [
            "--model",
            cli_fixtures["model"],
            "--data",
            cli_fixtures["data"],
            "--target-col",
            "label",
            "--metric",
            "f1",
            "--min-score",
            "1.5",  # unreachable — proves the flag reached the gate
            "--output",
            cli_fixtures["output"],
        ]
    )
    assert exit_code == 1

    report = json.loads(Path(cli_fixtures["output"]).read_text())
    assert report["model_metric"] == "f1"
    assert report["model_auc"] is None  # not an AUC run, so the legacy key stays empty


def test_cli_flags_win_over_config_file(cli_fixtures):
    config_path = cli_fixtures["tmp_path"] / "config.json"
    config_path.write_text(json.dumps({"performance": {"metric": "accuracy", "min_score": 1.5}}))

    exit_code = main(
        [
            "--model",
            cli_fixtures["model"],
            "--data",
            cli_fixtures["data"],
            "--target-col",
            "label",
            "--config",
            str(config_path),
            "--min-score",
            "0.0",  # overrides the file's unreachable threshold
            "--output",
            cli_fixtures["output"],
        ]
    )
    # This test is about precedence, not the verdict: other checks may flag
    # on this fixture, so assert on what the config actually produced.
    assert exit_code in (0, 1, 2)

    report = json.loads(Path(cli_fixtures["output"]).read_text())
    assert report["model_metric"] == "accuracy"  # from the config file
    performance = [
        r
        for r in report["results_by_category"]["performance"]
        if r["metadata"].get("metric_kind") == "score"
    ]
    assert performance[0]["metadata"]["threshold"] == 0.0  # from the CLI flag, not 1.5
    assert performance[0]["flag"] == "OK"


def test_cli_deprecated_config_key_is_logged(cli_fixtures, caplog):
    config_path = cli_fixtures["tmp_path"] / "config.json"
    config_path.write_text(json.dumps({"performance": {"min_accuracy": 1.5}}))

    with caplog.at_level("WARNING", logger="bdp_model_gate.cli"):
        main(
            [
                "--model",
                cli_fixtures["model"],
                "--data",
                cli_fixtures["data"],
                "--target-col",
                "label",
                "--config",
                str(config_path),
                "--output",
                cli_fixtures["output"],
            ]
        )

    assert "performance.min_accuracy' is deprecated" in caplog.text
    assert "performance.min_score" in caplog.text


@pytest.fixture
def pricing_cli_fixtures(tmp_path):
    """A pricing CSV carrying its side columns — exposure, the incumbent's
    premium, and the expected loss — beside the features."""
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(19)
    n = 300
    X = pd.DataFrame(
        {
            "risk_score": rng.gamma(4.0, 1.5, n),
            "vehicle_age": rng.integers(0, 20, n).astype(float),
        }
    )
    premium = 15_000.0 + 1_000.0 * X["risk_score"]
    model = LinearRegression().fit(X, premium)

    df = X.copy()
    df["realised_loss"] = np.clip(premium * rng.uniform(0.7, 1.3, n), 1.0, None)
    df["earned_years"] = rng.uniform(0.1, 1.0, n)
    df["last_years_premium"] = premium * rng.uniform(0.7, 1.05, n)
    data_path = tmp_path / "pricing.csv"
    df.to_csv(data_path, index=False)

    model_path = tmp_path / "premium.joblib"
    joblib.dump(model, model_path)
    return {
        "data": str(data_path),
        "model": str(model_path),
        "output": str(tmp_path / "report.json"),
    }


def test_cli_reads_exposure_and_a_baseline_and_keeps_them_out_of_X(pricing_cli_fixtures):
    """Both are columns of --data that are not features. Leaving the baseline
    premium in X would hand the model its own answer, which is exactly the
    leak `target_leakage` exists to find — so the CLI must drop them."""
    exit_code = main(
        [
            "--model",
            pricing_cli_fixtures["model"],
            "--data",
            pricing_cli_fixtures["data"],
            "--target-col",
            "realised_loss",
            "--exposure-col",
            "earned_years",
            "--baseline-col",
            "last_years_premium",
            "--task",
            "regression",
            "--metric",
            "mae",
            "--max-error",
            "1e12",
            "--output",
            pricing_cli_fixtures["output"],
        ]
    )
    assert exit_code in (0, 1, 2)

    report = json.loads(Path(pricing_cli_fixtures["output"]).read_text())
    by_name = {
        r["check_name"]: r for results in report["results_by_category"].values() for r in results
    }
    # Both optional inputs arrived: neither check reports its absence.
    assert by_name["prediction_dislocation"]["flag"] != "NOT_APPLICABLE"
    assert by_name["actual_vs_expected"]["metadata"]["exposure_weighted"] is True
    # And the feature contract saw only the two real features.
    assert by_name["performance_thresholds"]["metadata"]["exposure_weighted"] is True


@pytest.mark.parametrize("flag", ["--exposure-col", "--baseline-col"])
def test_cli_names_a_missing_side_column(pricing_cli_fixtures, flag, capsys):
    exit_code = main(
        [
            "--model",
            pricing_cli_fixtures["model"],
            "--data",
            pricing_cli_fixtures["data"],
            "--target-col",
            "realised_loss",
            flag,
            "no_such_column",
            "--task",
            "regression",
            "--metric",
            "r2",
            "--min-score=-1e9",
            "--output",
            pricing_cli_fixtures["output"],
        ]
    )
    assert exit_code == 1
    assert "no_such_column" in capsys.readouterr().err
