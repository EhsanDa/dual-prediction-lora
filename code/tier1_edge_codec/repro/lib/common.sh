#!/usr/bin/env bash
# common.sh -- shared plumbing for every repro script.
#
# Sourced, never executed. Provides: repo/package paths, the interpreter,
# flat-YAML config reading, and wall-clock timing.

set -euo pipefail

# ---------------------------------------------------------------- paths
# This file is at <repo>/code/tier1_edge_codec/repro/lib/common.sh
REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_DIR="$(cd "$REPRO_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PKG_DIR/../.." && pwd)"
RESULTS_DIR="$REPO_ROOT/results"
FROZEN_DIR="$REPO_ROOT/results_frozen"

# The interpreter. The project venv if it exists, else whatever python3 is on
# PATH. Override with PY=... for a different one.
if [[ -z "${PY:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PY="$REPO_ROOT/.venv/bin/python"
  else
    PY="python3"
  fi
fi
export PY

# eval.* / sim.* / edge.* resolve from the package root, and the scripts are
# all invoked as `python -m`, exactly as runner.ipynb did.
export PYTHONPATH="$PKG_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ---------------------------------------------------------------- config
# cfg <name> <key> [default]
#
# Reads `key: value` from repro/configs/<name>.yaml.
#
# FLAT SCALARS ONLY, and that is deliberate. The interpreter this project
# pins has no yaml module (see requirements-lock.txt), and adding PyYAML would
# widen the dependency set the paper's numbers rest on for the sake of parsing
# eight files of key/value pairs. If a config ever genuinely needs nesting,
# add the dependency then -- and update repro/README.md when you do.
#
# Understands: comments (# ...), surrounding whitespace, and single or double
# quoted values. Does not understand: nesting, lists, multi-line strings,
# anchors. It will silently ignore an indented line rather than misread it.
cfg() {
  local name="$1" key="$2" default="${3-}" file val
  file="$REPRO_DIR/configs/$name.yaml"
  [[ -f "$file" ]] || { echo "repro: no config at $file" >&2; exit 2; }

  val="$(awk -v k="$key" '
    /^[[:space:]]*#/ { next }                 # comment
    /^[[:space:]]/   { next }                 # indented => nested, not ours
    {
      i = index($0, ":")
      if (i == 0) next
      key = substr($0, 1, i - 1)
      gsub(/^[ \t]+|[ \t]+$/, "", key)
      if (key != k) next
      v = substr($0, i + 1)
      sub(/[ \t]+#.*$/, "", v)                # trailing comment
      gsub(/^[ \t]+|[ \t]+$/, "", v)
      gsub(/^"|"$/, "", v)
      gsub(/^'\''|'\''$/, "", v)
      print v
      exit
    }' "$file")"

  if [[ -z "$val" ]]; then
    if [[ $# -ge 3 ]]; then printf '%s' "$default"; return 0; fi
    echo "repro: key '$key' not found in $file and no default given" >&2
    exit 2
  fi
  printf '%s' "$val"
}

# ---------------------------------------------------------------- timing
_repro_t0=""
repro_begin() {
  _repro_t0=$(date +%s)
  mkdir -p "$RESULTS_DIR"
  echo "=============================================================="
  echo "repro: $1"
  echo "  started   $(date -Is)"
  echo "  python    $PY"
  echo "  writing   ${2:-$RESULTS_DIR}"
  echo "=============================================================="
}

repro_end() {
  local t1 dt
  t1=$(date +%s); dt=$(( t1 - _repro_t0 ))
  printf 'repro: %s finished in %dm %ds (%d s wall clock)\n' \
         "$1" $(( dt / 60 )) $(( dt % 60 )) "$dt"
}

# Fail early and legibly rather than deep inside numpy.
require_file() {
  [[ -e "$1" ]] || { echo "repro: required input missing: $1" >&2
                     echo "       $2" >&2; exit 3; }
}
