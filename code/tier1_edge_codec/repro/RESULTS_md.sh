#!/usr/bin/env bash
# Produces: results/RESULTS.md
# Reads results JSONs only -- seconds to run.
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
NAME=RESULTS_md

RES=$(cfg $NAME results_json); SENS=$(cfg $NAME sensitivity_json)
OUT=$(cfg $NAME out)

repro_begin "$NAME" "$RESULTS_DIR/$OUT"
cd "$REPO_ROOT"
require_file "$RES" "run repro/tier1_results.sh first, or seed_from_frozen.sh"
# sensitivity is optional: report_tables degrades gracefully without it.

"$PY" -m eval.report_tables \
    --results "$REPO_ROOT/$RES" --sensitivity "$REPO_ROOT/$SENS" \
    --out "$RESULTS_DIR/$OUT"
repro_end "$NAME"
