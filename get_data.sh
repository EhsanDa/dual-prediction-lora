#!/usr/bin/env bash
# get_data.sh -- fetch the public ECL benchmark and build code/tier2_data/.
# The data is not redistributed here. Source: the UCI "ElectricityLoadDiagrams
# 20112014" collection, in the hourly 321-client form of LSTNet (Lai et al.,
# SIGIR 2018), from the LSTNet authors' repository. Check the UCI page for the
# dataset's licence terms before redistributing anything derived from it.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p raw
[ -f raw/electricity.txt ] || {
  wget -O raw/electricity.txt.gz \
    https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz
  gunzip raw/electricity.txt.gz
}
python code/tier1_edge_codec/vendor/10_tier2_preprocess.py \
  --data raw/electricity.txt --out code/tier2_data
echo "built code/tier2_data -- compare with code/DATA.md"
