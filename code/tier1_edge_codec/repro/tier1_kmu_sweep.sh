#!/usr/bin/env bash
# Produces: results/tier1_kmu_sweep.json
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
NAME=tier1_kmu_sweep

DATA=$(cfg $NAME data_dir); PRE=$(cfg $NAME pretrained)
AL=$(cfg $NAME alpha);      KMU=$(cfg $NAME k_mu); OUT=$(cfg $NAME out)

repro_begin "$NAME" "$RESULTS_DIR/$OUT"
cd "$PKG_DIR"
require_file "$DATA" "preprocessed tier-2 arrays; see code/DATA.md"

"$PY" -m eval.kmu_sweep \
    --data_dir "$DATA" --pretrained "$PRE" --alpha "$AL" --k_mu "$KMU" \
    --out "$RESULTS_DIR/$OUT"
"$PY" "$REPRO_DIR/lib/assert_params.py" "$RESULTS_DIR/$OUT" \
    --frozen "$FROZEN_DIR/$OUT" alpha="$AL"

repro_end "$NAME"
