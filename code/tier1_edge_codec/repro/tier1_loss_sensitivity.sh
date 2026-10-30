#!/usr/bin/env bash
# Produces: results/tier1_loss_sensitivity.json  (Table I)
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
NAME=tier1_loss_sensitivity

DATA=$(cfg $NAME data_dir); AL=$(cfg $NAME alpha)
NC=$(cfg $NAME n_clients);  CFG=$(cfg $NAME configs)
PRE=$(cfg $NAME pretrained); KMU=$(cfg $NAME k_mu); OUT=$(cfg $NAME out)

repro_begin "$NAME" "$RESULTS_DIR/$OUT"
cd "$PKG_DIR"
require_file "$DATA" "preprocessed tier-2 arrays; see code/DATA.md"

"$PY" -m eval.loss_sweep \
    --data_dir "$DATA" --alpha "$AL" --n_clients "$NC" --configs "$CFG" \
    --pretrained "$PRE" --k_mu "$KMU" --out "$RESULTS_DIR/$OUT"
"$PY" "$REPRO_DIR/lib/assert_params.py" "$RESULTS_DIR/$OUT" \
    --frozen "$FROZEN_DIR/$OUT" \
    alpha="$AL" n_clients="$NC" k_mu="$KMU"

repro_end "$NAME"
