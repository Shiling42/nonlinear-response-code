#!/usr/bin/env bash
# Regenerate all computed figure panels into figures/.
set -euo pipefail
cd "$(dirname "$0")"

for nb in notebooks/*.ipynb; do
  echo "== $nb"
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$nb"
done

echo "== Fig. 4";  python3 scripts/Fig4_push_pull_hidden_network.py
echo "== Fig. S1"; python3 scripts/FigS1_mfpt_traffic_saturation.py
echo "== Fig. S2"; python3 scripts/FigS2_triangle_inequality_saturation.py
echo "All panels written to figures/"
