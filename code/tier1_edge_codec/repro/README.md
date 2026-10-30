# repro/ — the pipeline that replaced the notebook

Every result artifact backing the conference paper, as a script you can run
from the repository root with no arguments.

This replaces `code/runner.ipynb`, now at `_attic/runner.ipynb`. The notebook
mounted Google Drive and `cd`'d into one person's `MyDrive` path, and its saved
outputs come from at least two different sessions — see `_attic/README.md`.

## Quick start

```bash
cd <repository root>
source .venv/bin/activate        # or any Python 3.11 environment with requirements-lock.txt

bash code/tier1_edge_codec/repro/run_all.sh --dry-run     # order + cache check
bash code/tier1_edge_codec/repro/run_all.sh --cheap-only  # seconds
bash code/tier1_edge_codec/repro/run_all.sh               # hours; see below
```

Everything writes to `results/`. **Nothing ever writes to `results_frozen/`.**

## Prerequisites

| | |
|---|---|
| Interpreter | Python 3.11 with `requirements-lock.txt` installed. Scripts use `.venv/bin/python` if present, else `python3`; override with `PY=...`. |
| Data | `code/tier2_data/` — 139 MB, not in git. See `code/DATA.md`. |
| Model | `code/dlinear.pt` — 76 KB, in git. |
| Disk | ~1.5 GB transient for `_recon_cache/` (two 34 MB float32 arrays per configuration), plus ~1 MB of outputs. |
| Compute | CPU is the default and is what the paper used (`--device cpu`). No GPU required. |

## Runtime

| Script | Produces | Runtime |
|---|---|---|
| `tier1_tap_ablation.sh` | `tier1_tap_ablation.json` + `_pretrain_cache/` | **UNKNOWN** — not yet run here |
| `tier1_results.sh` | `tier1_results.json` | **~175 min measured on Colab**, UNKNOWN here |
| `tier1_sensitivity.sh` | `tier1_sensitivity.json` | **UNKNOWN** |
| `tier1_kmu_sweep.sh` | `tier1_kmu_sweep.json` | **UNKNOWN** |
| `tier1_loss_sensitivity.sh` | `tier1_loss_sensitivity.json` | **UNKNOWN** |
| `RESULTS_md.sh` | `RESULTS.md` | **1 s measured** |
| `figures.sh` | `figures/` | **2 s measured** |
| `tier1_tap_forecast.sh` | `tier1_tap_forecast.json` | **<1 s measured** |

The 175 minutes is not an estimate: `_attic/runner.ipynb` cell 9 recorded a
per-configuration time for each of 45 configurations, and they sum to 10,524 s.
That is the policy loop alone, excluding the DLinear scoring pass over 150,000
validation windows. Every `UNKNOWN` above is genuinely unmeasured — this
machine has run only the three cheap scripts. Do not replace an `UNKNOWN` with
a guess; measure it and write the measurement in.

## Dependency order

`run_all.sh` enforces this. It is not cosmetic:

```
tier1_tap_ablation ──> _pretrain_cache/{T7,T12}.json ──> tier1_results
                                                              │
tier1_sensitivity ────────────────┐                           │
                                  ▼                           ▼
                              figures  <──  RESULTS_md  <──  tier1_results.json
                                                              │
                                              tier1_tap_forecast
tier1_kmu_sweep        (independent)
tier1_loss_sensitivity (independent)
```

`eval/run_eval.py` calls `SystemExit` if `_pretrain_cache/<id>.json` is missing
for any `--extra_taps` id, so `tap_ablation` genuinely must precede it. The
scripts check for it up front and say so in one line rather than failing deep
inside the run.

## Configuration

One flat YAML per script in `configs/`, parsed by `cfg()` in `lib/common.sh`.

**Flat scalars only, deliberately.** The pinned interpreter has no `yaml`
module, and adding PyYAML would widen the dependency set the paper's numbers
rest on in order to parse eight files of `key: value`. The parser handles
comments, whitespace and quotes, and ignores indented lines rather than
misreading them. If a config ever truly needs nesting, add the dependency then
— and update this section.

## The `_pretrain_cache` staleness gate

This is the one piece of new behaviour, and it exists because of a real hazard
found in Phase A (`docs/reference/module-map.md`).

`eval/tap_ablation.py` memoises each tap set's global ridge fit to
`_pretrain_cache/<id>.json`, and on a later run reloads it if the file merely
**exists**. Nothing checks that the data, the lags, `k_mu` or the fitting code
are still the ones it was fitted under. The only validation anywhere is a
length check on `w_init`. So a stale entry is silently reused and silently
changes `w_init` in every downstream row using that tap id — and
`eval/run_eval.py` feeds it straight into `CodecParams`.

`tap_ablation.py` is **not modified** — TASK 02 requires its behaviour left
alone. Instead `lib/manifest.py` writes `_pretrain_cache/<id>.manifest.json`
recording the data-file checksums, the lags, `k_mu`, and the fitter's md5 and
git blob. `run_all.sh` verifies all of them before running anything and
**refuses to start** on any mismatch.

It never deletes. On a mismatch it prints what moved and stops; removing an
entry is the author's decision (PLAYBOOK standing rule 3).

```bash
# stamp entries this harness produced
python repro/lib/manifest.py write  _pretrain_cache ../tier2_data eval/tap_ablation.py
# check
python repro/lib/manifest.py verify _pretrain_cache ../tier2_data eval/tap_ablation.py
```

### The twelve adopted entries

The `_pretrain_cache/` currently on disk came from the Colab run, before any of
this existed. Its true origin is unknowable, so it was **adopted**, not
certified: each manifest carries `adopted_without_verification: true` and
records today's state, and `verify` prints a standing warning naming all twelve.

That means drift from here on is caught, while nothing false is claimed about
where those fits came from. The warning clears only when they are genuinely
refitted by `tier1_tap_ablation.sh`.

## `seed_from_frozen.sh`

Copies the frozen JSONs into `results/` so the three JSON-only consumers can be
exercised without paying for the upstream stages.

**This is not a reproduction.** It verifies the JSON → table/figure half of the
pipeline only. What it proved on 2026-09-14 is recorded in
`docs/status/reproduction.md`: the three CSV tables and `captions.tex` come back
bit-identical, and `tier1_tap_forecast.json` is bit-identical.
