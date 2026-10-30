#!/usr/bin/env bash
# Produces: results/tier1_tap_forecast.json  (Table VI)
# Reads a results JSON only -- no data, no model, seconds to run.
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
NAME=tier1_tap_forecast

RES=$(cfg $NAME results_json); RATES=$(cfg $NAME rates); OUT=$(cfg $NAME out)

repro_begin "$NAME" "$RESULTS_DIR/$OUT"
cd "$REPO_ROOT"
require_file "$RES" "run repro/tier1_results.sh first, or seed_from_frozen.sh"

"$PY" -m eval.tap_forecast_table \
    --results "$REPO_ROOT/$RES" --rates "$RATES" --out "$RESULTS_DIR/$OUT"
# `rates` is where the FIRST config-drift defect lived: the config said
# "15,10" and the script defaults to 15,10,6.
"$PY" "$REPRO_DIR/lib/assert_params.py" "$RESULTS_DIR/$OUT" \
    --frozen "$FROZEN_DIR/$OUT" rates="$RATES"

repro_end "$NAME"
