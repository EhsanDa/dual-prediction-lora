#!/usr/bin/env bash
# Produces: results/tier1_results.json  -- the artifact everything else reads.
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
NAME=tier1_results

DATA=$(cfg $NAME data_dir);   CKPT=$(cfg $NAME checkpoint)
TAPS=$(cfg $NAME extra_taps); PRE=$(cfg $NAME pretrained)
PC=$(cfg $NAME per_client);   KMU=$(cfg $NAME k_mu)
SF=$(cfg $NAME budget_sf);    RM=$(cfg $NAME retry_margin)
TOL=$(cfg $NAME tolerance);   MB=$(cfg $NAME margin_basis)
SS=$(cfg $NAME select_samples); BS=$(cfg $NAME batch_size)
DEV=$(cfg $NAME device);      OUT=$(cfg $NAME out)

repro_begin "$NAME" "$RESULTS_DIR/$OUT"
cd "$PKG_DIR"
require_file "$DATA" "preprocessed tier-2 arrays; see code/DATA.md"
require_file "$CKPT" "the frozen tier-2 forecaster"
# run_eval SystemExits without these; say so before it does.
IFS=',' read -ra _t <<< "$TAPS"
for t in "${_t[@]}"; do
  require_file "_pretrain_cache/${t}.json" \
    "run repro/tier1_tap_ablation.sh first -- every tap set needs its own ridge fit"
done

"$PY" -m eval.run_eval \
    --data_dir "$DATA" --checkpoint "$CKPT" --extra_taps "$TAPS" \
    --pretrained "$PRE" --per_client "$PC" --k_mu "$KMU" \
    --budget_sf "$SF" --retry_margin "$RM" --tolerance "$TOL" \
    --margin_basis "$MB" --select_samples "$SS" --batch_size "$BS" \
    --device "$DEV" --out "$RESULTS_DIR/$OUT"
"$PY" "$REPRO_DIR/lib/assert_params.py" "$RESULTS_DIR/$OUT" \
    --frozen "$FROZEN_DIR/$OUT" checkpoint="$CKPT"

repro_end "$NAME"
