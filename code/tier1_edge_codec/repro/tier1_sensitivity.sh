#!/usr/bin/env bash
# Produces: results/tier1_sensitivity.json  (read by figures.sh)
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
NAME=tier1_sensitivity

DATA=$(cfg $NAME data_dir); CKPT=$(cfg $NAME checkpoint)
SPLIT=$(cfg $NAME split);   NORI=$(cfg $NAME n_origins); OUT=$(cfg $NAME out)

repro_begin "$NAME" "$RESULTS_DIR/$OUT"
cd "$PKG_DIR"
require_file "$DATA" "preprocessed tier-2 arrays; see code/DATA.md"
require_file "$CKPT" "the frozen tier-2 forecaster"

"$PY" -m eval.sensitivity \
    --data_dir "$DATA" --checkpoint "$CKPT" --split "$SPLIT" \
    --n_origins "$NORI" --out "$RESULTS_DIR/$OUT"
"$PY" "$REPRO_DIR/lib/assert_params.py" "$RESULTS_DIR/$OUT" \
    --frozen "$FROZEN_DIR/$OUT" n_origins="$NORI"

repro_end "$NAME"
