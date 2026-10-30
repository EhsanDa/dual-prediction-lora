#!/usr/bin/env bash
# Produces: results/edge_pretrained.json + results/edge_pretrained_per_client.npy
#
# The provenance of the SHIPPED predictor initialisation (card P1-02). This is
# the invocation that was missing from the record; running it reproduces the
# checked-in inputs/ files. Seconds to run.
#
# It deliberately writes to results/, never to inputs/. To check provenance,
# run this and diff against inputs/ -- see the tail of this script.
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
NAME=edge_pretrained

DATA=$(cfg $NAME data_dir); OUT=$(cfg $NAME out)
PCO=$(cfg $NAME per_client_out)

repro_begin "$NAME" "$RESULTS_DIR/$OUT"
cd "$PKG_DIR"
require_file "$DATA" "preprocessed tier-2 arrays; see code/DATA.md"

"$PY" -m sim.pretrain --data_dir "$DATA" \
    --out "$RESULTS_DIR/$OUT" --per_client_out "$RESULTS_DIR/$PCO"

echo
echo "--- provenance check against the checked-in inputs/ ---"
"$PY" - "$RESULTS_DIR/$OUT" "$RESULTS_DIR/$PCO" <<'PY'
import json, sys, numpy as np
fresh, fresh_pc = sys.argv[1], sys.argv[2]
a = json.load(open("inputs/edge_pretrained.json"))
b = json.load(open(fresh))
q = a["w_global_q"] == b["w_global_q"]
pa = np.load("inputs/edge_pretrained_per_client.npy")
pb = np.load(fresh_pc)
pc = np.array_equal(pa, pb)
w = max(abs(x - y) / max(abs(x), 1e-300)
        for x, y in zip(a["w_global_real"], b["w_global_real"]))
print(f"  w_global_q identical          : {q}")
print(f"  per_client.npy bitwise equal  : {pc}")
print(f"  w_global_real max rel deviation: {w:.3e}")
print("  VERDICT:", "provenance holds" if (q and pc) else "** DIVERGED -- investigate **")
sys.exit(0 if (q and pc) else 1)
PY
repro_end "$NAME"
