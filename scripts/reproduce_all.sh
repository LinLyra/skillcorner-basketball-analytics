#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/skillcorner-basketball-mpl}"

pytest -q
python scripts/run_study.py
python scripts/run_robustness.py
python scripts/run_negative_control.py
python scripts/run_subgroup_analysis.py
python scripts/run_smoothing_sensitivity.py
python scripts/run_spatial_geometry.py
python scripts/run_official_benchmark.py
python scripts/render_results.py
python scripts/render_robustness.py
python scripts/render_disagreement_cases.py
python scripts/render_final_summary.py
python scripts/animate_disagreement.py
python scripts/validate_release.py
