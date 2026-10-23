#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/skillcorner-basketball-mpl}"

pytest -q
python scripts/run_study.py
python scripts/run_robustness.py
python scripts/run_negative_control.py
python scripts/run_inference_checks.py
python scripts/run_nested_models.py
python scripts/run_subgroup_analysis.py
python scripts/run_smoothing_sensitivity.py
python scripts/run_spatial_geometry.py
python scripts/run_official_benchmark.py
python scripts/run_pred_error_monte_carlo.py
python scripts/analyze_defender_policy_consequences.py
python scripts/render_results.py
python scripts/render_robustness.py
python scripts/render_disagreement_cases.py
python scripts/render_final_summary.py
python scripts/render_revision_summary.py
python scripts/animate_disagreement.py
for rank in 1 2 3; do
  python scripts/animate_defender_policy_case.py --rank "$rank"
done
python scripts/validate_release.py
