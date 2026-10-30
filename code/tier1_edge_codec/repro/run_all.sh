#!/usr/bin/env bash
# run_all.sh -- every repro script, in dependency order.
#
#   bash code/tier1_edge_codec/repro/run_all.sh [--dry-run] [--cheap-only]
#
# --dry-run     resolve configs, verify the cache, print the order, run nothing
# --cheap-only  only the JSON-only consumers (seconds, not hours)
#
# Order is not cosmetic:
#   1. tap_ablation    writes _pretrain_cache/<id>.json
#   2. run_eval        SystemExits if _pretrain_cache/{T7,T12}.json are absent
#   3. sensitivity     independent of run_eval, but figures reads its output
#   4. kmu_sweep       independent
#   5. loss_sweep      independent
#   6. RESULTS_md / figures / tap_forecast   all read results/tier1_results.json
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

DRY=0; CHEAP=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --cheap-only) CHEAP=1 ;;
    -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "run_all: unknown option $a" >&2; exit 2 ;;
  esac
done

EXPENSIVE=(tier1_tap_ablation tier1_results tier1_sensitivity
           tier1_kmu_sweep tier1_loss_sensitivity)
CHEAP_ONES=(RESULTS_md figures tier1_tap_forecast)

if [[ $CHEAP -eq 1 ]]; then ORDER=("${CHEAP_ONES[@]}")
else ORDER=("${EXPENSIVE[@]}" "${CHEAP_ONES[@]}"); fi

# ------------------------------------------------- the cache staleness gate
# PLAYBOOK TASK 02 step 4. _pretrain_cache/ is a memoise-to-disk that
# tap_ablation.py reloads on mere file existence. If it was fitted under other
# data, other lags, another k_mu or another fitter, every downstream row that
# uses that tap id silently changes. Refuse to start. Never auto-delete.
echo "run_all: verifying _pretrain_cache against current state"
_data=$(cfg tier1_tap_ablation data_dir)
_kmu=$(cfg tier1_tap_ablation k_mu)
if ! ( cd "$PKG_DIR" && REPRO_K_MU="$_kmu" "$PY" "$REPRO_DIR/lib/manifest.py" \
         verify _pretrain_cache "$_data" eval/tap_ablation.py ); then
  if [[ $DRY -eq 1 ]]; then
    echo "run_all: --dry-run, so continuing to print the order anyway." >&2
    echo "run_all: a real run would have stopped here." >&2
  else
    echo "run_all: aborted before running anything." >&2; exit 1
  fi
fi

echo
echo "run_all: execution order"
for i in "${!ORDER[@]}"; do
  printf '  %d. %s\n' "$((i + 1))" "${ORDER[$i]}"
done
echo

if [[ $DRY -eq 1 ]]; then
  echo "run_all: --dry-run, resolving configs only"
  for n in "${ORDER[@]}"; do
    printf '  %-22s out=%s\n' "$n" "$(cfg "$n" out "$(cfg "$n" outdir '?')")"
  done
  echo "run_all: nothing was executed."
  exit 0
fi

t0=$(date +%s)
for n in "${ORDER[@]}"; do
  bash "$REPRO_DIR/$n.sh"
  echo
done
t1=$(date +%s); dt=$(( t1 - t0 ))
printf 'run_all: %d script(s) in %dm %ds\n' "${#ORDER[@]}" $(( dt / 60 )) $(( dt % 60 ))
echo "run_all: outputs in $RESULTS_DIR -- diff them against $FROZEN_DIR"
