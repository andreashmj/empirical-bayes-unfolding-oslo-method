#!/usr/bin/env bash
set -euo pipefail

mkdir -p figures/examples

echo "[quickstart] running a small posterior unfolding example"
python -m src.run_unfolding \
  -c examples/configs/quickstart_posterior.yaml

POST_NC="results/examples/quickstart/posterior_demo/posterior/draws.nc"

echo "[quickstart] plotting posterior spectrum"
python -m src.figures.results.cli posterior-spectrum \
  "$POST_NC" \
  --ex 2500 \
  --var eta \
  --mass 0.95 \
  --zoom "150-350,850-1250" \
  --layout column \
  --out figures/examples/quickstart_posterior_Ex2500.pdf \
  --no-show

echo "[quickstart] done"
echo "  result: $POST_NC"
echo "  figure: figures/examples/quickstart_posterior_Ex2500.pdf"
