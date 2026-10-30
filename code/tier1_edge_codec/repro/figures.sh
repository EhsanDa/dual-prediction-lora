#!/usr/bin/env bash
# Produces: results/figures/  (5 figures x PDF+PNG, 3 CSV tables, captions)
# Reads results JSONs only.
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
NAME=figures

RES=$(cfg $NAME results_json); SENS=$(cfg $NAME sensitivity_json)
OUTDIR=$(cfg $NAME outdir)

repro_begin "$NAME" "$RESULTS_DIR/$OUTDIR"
cd "$REPO_ROOT"
require_file "$RES" "run repro/tier1_results.sh first, or seed_from_frozen.sh"

MPLBACKEND=Agg "$PY" -m eval.figures \
    --results "$REPO_ROOT/$RES" --sensitivity "$REPO_ROOT/$SENS" \
    --outdir "$RESULTS_DIR/$OUTDIR"
repro_end "$NAME"
