"""Post-hoc reporting is isolated from model selection; only synthetic fixtures."""
import json
from pathlib import Path
import sys
import stat

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components"))
import functional_analysis as analysis  # noqa: E402


def test_score_diagnostic_and_ci_annotation():
    values = analysis.score_diagnostics(np.array([0., 0., 0.5, 1.]), .5)
    assert values == {"distinct_scores": 3, "at_cutoff": 1, "probability_boundary_fraction": .75}
    assert analysis.score_diagnostics(np.array([0., .1]), .5, False)["probability_boundary_fraction"] is None
    assert "†" in analysis.format_metric({"estimate": .1, "ci_low": .2, "ci_high": .3})


@pytest.mark.parametrize("uplift", [False, True])
def test_frozen_synthetic_report_does_not_change_results(tmp_path, uplift):
    (tmp_path / "alice/results").mkdir(parents=True)
    (tmp_path / "alice/code").mkdir()
    (tmp_path / "analysis").mkdir()
    (tmp_path / "status.json").write_text('{"status":"passed"}')
    (tmp_path / "declaration.json").write_text('{"dataset":"synthetic", "prepared_experiment":"synthetic"}')
    cfg = {"shared": {"model_seeds": [11], "split_seed": 20260927, "top_k_fraction": .1},
           "datasets": {"synthetic": {"treatment": uplift, "treatment_arms": [1, 2]}}}
    (tmp_path / "alice/code/local_functional.yaml").write_text(yaml.safe_dump(cfg))
    names = ["arm1_centered_qini", "arm2_centered_qini"] if uplift else ["roc_auc"]
    metrics = {name: {"estimate": .4, "ci_low": .3, "ci_high": .5} for name in names}
    report = {"metrics": [], "paired_differences": [], "output_attack_diagnostics": []}
    for route in analysis.ROUTES:
        folder = tmp_path / "alice/trainings" / route / "results"
        folder.mkdir(parents=True)
        np.save(folder / "test_predictions.npy", np.tile([.1, .2, .3] if uplift else [.2], (10, 1)))
        report["metrics"].append({"seed": 11, "route": route, "selected_training": route, "metrics": metrics})
        report["output_attack_diagnostics"].append({"seed": 11, "route": route, "loss_attack_auc": .5, "ci_low": .4, "ci_high": .6})
    for route in analysis.ROUTES[:-1]:
        report["paired_differences"].append({"seed": 11, "comparison": f"L3_secure_minus_{route}", "metrics": metrics})
    evaluation = tmp_path / "alice/results/private_evaluation.json"
    evaluation.write_text(json.dumps(report))
    evaluation.chmod(0o640)
    before = analysis.fingerprint(evaluation)
    result = analysis.write_analysis({"round_path": str(tmp_path), "analysis_root": str(tmp_path / "analysis")})
    assert analysis.fingerprint(evaluation) == before
    assert stat.S_IMODE(evaluation.stat().st_mode) == 0o640
    output = Path(result["report"]).parent
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(f.stat().st_mode) == 0o600 for f in output.iterdir() if f.is_file())
    assert Path(result["figure"]).exists() and Path(result["report"]).exists()
    assert "原始顺序" in Path(result["report"]).read_text()
    assert analysis.write_analysis({"round_path": str(tmp_path), "analysis_root": str(tmp_path / "analysis")})["report"] != result["report"]
    (tmp_path / "status.json").write_text('{"status":"failed"}')
    with pytest.raises(ValueError, match="completed"):
        analysis.write_analysis({"round_path": str(tmp_path), "analysis_root": str(tmp_path / "analysis")})
    with pytest.raises(ValueError, match="Incomplete"):
        analysis.create_analysis_notebook(tmp_path)


def test_confirmation_requires_five_seeds_and_separated_intervals():
    seeds = [11, 22, 33, 44, 55]
    evaluation = {'metrics': [], 'paired_differences': []}
    for seed in seeds:
        for route, low, high in [('L0', .5, .6), ('L3_secure', .65, .75)]:
            evaluation['metrics'].append({'seed': seed, 'route': route,
                'metrics': {'auc': {'ci_low': low, 'ci_high': high}}})
        evaluation['paired_differences'].append({'seed': seed, 'comparison': 'L3_secure_minus_L0',
            'metrics': {'auc': {'ci_low': .02}}})
    assert analysis.conditional_comparison(evaluation, seeds, 'auc', 'L0').startswith('在此构造')
    assert '未达到项目' in analysis.conditional_comparison(evaluation, seeds[:-1], 'auc', 'L0')
    evaluation['metrics'][-1]['metrics']['auc']['ci_low'] = .55
    assert '未达到项目' in analysis.conditional_comparison(evaluation, seeds, 'auc', 'L0')
    evaluation['metrics'][-1]['metrics']['auc']['ci_low'] = .65
    evaluation['paired_differences'][-1]['metrics']['auc']['ci_low'] = -.01
    assert '未达到项目' in analysis.conditional_comparison(evaluation, seeds, 'auc', 'L0')
