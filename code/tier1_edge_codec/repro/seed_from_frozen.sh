#!/usr/bin/env bash
# Copy the frozen JSONs into results/ so the JSON-only consumers
# (RESULTS_md.sh, figures.sh, tier1_tap_forecast.sh) can be exercised without
# first paying for the expensive upstream stages.
#
# THIS IS NOT A REPRODUCTION. It verifies the JSON -> artifact half of the
# pipeline only. Anything produced after seeding is a reproduction of the
# tables and figures, NOT of the numbers inside tier1_results.json.
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

mkdir -p "$RESULTS_DIR"
n=0
for f in tier1_results.json tier1_sensitivity.json tier1_kmu_sweep.json \
         tier1_tap_ablation.json tier1_loss_sensitivity.json; do
  if [[ -f "$FROZEN_DIR/$f" ]]; then cp "$FROZEN_DIR/$f" "$RESULTS_DIR/$f"; n=$((n+1)); fi
done
echo "seeded $n frozen JSON(s) into $RESULTS_DIR"
echo "NOTE: outputs derived from these are NOT a reproduction of the upstream"
echo "      numbers. See docs/status/reproduction.md."
