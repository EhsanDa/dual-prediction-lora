#!/usr/bin/env bash
# Produces: results/robustness/*
#
# The robustness study behind docs/status/robustness.md. Answers the reviewers'
# "single seed" objection -- see docs/reference/stochasticity.md for the audit
# of what is actually stochastic (three things) and what is not (everything
# else, which has no seed to vary).
#
#   bash repro/robustness.sh [--stage bootstrap|loss|alpha|all]
#
# Nothing here overwrites results/ or results_frozen/. Each stage writes its
# own file under results/robustness/, and a seed whose file already exists is
# SKIPPED -- so an interrupted run resumes where it stopped rather than
# repeating hours of work. Delete a file to force that seed to re-run.
#
# COST, measured 2026-09-14: bootstrap ~12 min, alpha ~12 min per seed,
# loss ~58 min per seed at 321 clients (projected from 23 min at 64).
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
NAME=robustness

STAGE=all
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    *) echo "robustness: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

DATA=$(cfg $NAME data_dir);    CKPT=$(cfg $NAME checkpoint)
LOSS_SEEDS=$(cfg $NAME loss_seeds);  SEL_SEEDS=$(cfg $NAME select_seeds)
NCLI=$(cfg $NAME n_clients);   CONFIGS=$(cfg $NAME configs)
ALPHA=$(cfg $NAME alpha);      LOSSES=$(cfg $NAME losses)
KMU=$(cfg $NAME k_mu)

OUT="$RESULTS_DIR/robustness"
mkdir -p "$OUT"
cd "$PKG_DIR"
require_file "$DATA" "preprocessed tier-2 arrays; see code/DATA.md"

# ---------------------------------------------------------------- stage 1 ---
# Per-client vectors. run_eval already computes them and throws them away;
# --per_client_out persists them. --out goes to a THROWAWAY path so the
# reproduced results/tier1_results.json is not touched.
if [[ $STAGE == all || $STAGE == bootstrap ]]; then
  echo "=== robustness 1/3: per-client vectors for the client bootstrap ==="
  "$PY" -m eval.run_eval \
      --data_dir "$DATA" --checkpoint "$CKPT" --extra_taps T7,T12 \
      --pretrained inputs/edge_pretrained.json \
      --per_client inputs/edge_pretrained_per_client.npy \
      --k_mu "$KMU" --budget_sf 12 --retry_margin 0.20 --tolerance 0.10 \
      --margin_basis min --select_samples 150000 --batch_size 512 \
      --device cpu \
      --out "$OUT/_run_eval_bootstrap.json" \
      --per_client_out "$OUT/per_client.json"
fi

# ---------------------------------------------------------------- stage 2 ---
# Table I over several packet-loss draws, at every client.
if [[ $STAGE == all || $STAGE == loss ]]; then
  echo "=== robustness 2/3: Table I over seeds ${LOSS_SEEDS} at ${NCLI} clients ==="
  IFS=',' read -ra SEEDS <<< "$LOSS_SEEDS"
  for s in "${SEEDS[@]}"; do
    if [[ -s "$OUT/loss_sweep_seed${s}.json" ]]; then
      echo "--- loss seed $s: already present, skipping ---"; continue
    fi
    echo "--- loss seed $s ---"
    "$PY" -m eval.loss_sweep \
        --data_dir "$DATA" --alpha "$ALPHA" --n_clients "$NCLI" \
        --configs "$CONFIGS" --losses "$LOSSES" --k_mu "$KMU" \
        --pretrained inputs/edge_pretrained.json \
        --seed "$s" \
        --out "$OUT/loss_sweep_seed${s}.json"
  done
fi

# ---------------------------------------------------------------- stage 3 ---
# Does the selected alpha depend on which validation windows were sampled?
if [[ $STAGE == all || $STAGE == alpha ]]; then
  echo "=== robustness 3/3: alpha selection over seeds ${SEL_SEEDS} ==="
  IFS=',' read -ra SEEDS <<< "$SEL_SEEDS"
  for s in "${SEEDS[@]}"; do
    if [[ -s "$OUT/run_eval_selectseed${s}.json" ]]; then
      echo "--- select seed $s: already present, skipping ---"; continue
    fi
    echo "--- select seed $s ---"
    "$PY" -m eval.run_eval \
        --data_dir "$DATA" --checkpoint "$CKPT" --extra_taps T7,T12 \
        --pretrained inputs/edge_pretrained.json \
        --per_client inputs/edge_pretrained_per_client.npy \
        --k_mu "$KMU" --budget_sf 12 --retry_margin 0.20 --tolerance 0.10 \
        --margin_basis min --select_samples 150000 --batch_size 512 \
        --device cpu --select_seed "$s" \
        --out "$OUT/run_eval_selectseed${s}.json"
  done
fi

echo
echo "robustness: artifacts under $OUT"
echo "Summarise with: python -m eval.robustness_report"
