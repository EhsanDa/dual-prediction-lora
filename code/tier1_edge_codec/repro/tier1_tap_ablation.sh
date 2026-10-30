#!/usr/bin/env bash
# Produces: results/tier1_tap_ablation.json
# Also writes _pretrain_cache/<id>.json, which tier1_results.sh requires,
# and stamps each entry with a manifest so staleness cannot pass silently.
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
NAME=tier1_tap_ablation

DATA=$(cfg $NAME data_dir); TAPS=$(cfg $NAME tap_sets)
KMU=$(cfg $NAME k_mu);      ALPHAS=$(cfg $NAME alphas)
PRE=$(cfg $NAME pretrained); OUT=$(cfg $NAME out)

repro_begin "$NAME" "$RESULTS_DIR/$OUT"
cd "$PKG_DIR"
require_file "$DATA" "preprocessed tier-2 arrays; see code/DATA.md"

"$PY" -m eval.tap_ablation \
    --data_dir "$DATA" --tap_sets "$TAPS" --k_mu "$KMU" \
    --alphas "$ALPHAS" --pretrained "$PRE" \
    --out "$RESULTS_DIR/$OUT"

"$PY" "$REPRO_DIR/lib/assert_params.py" "$RESULTS_DIR/$OUT" \
    --frozen "$FROZEN_DIR/$OUT" \
    alphas="$ALPHAS" k_mu="$KMU"

REPRO_K_MU="$KMU" "$PY" "$REPRO_DIR/lib/manifest.py" write \
    _pretrain_cache "$DATA" eval/tap_ablation.py
repro_end "$NAME"
