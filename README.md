# Bounded-error dual prediction for fresh load forecasting over LoRa

Code and artifacts for:

> E. Dabidian, M. Amirzadeh Gero, S. Ershadi-Nasab, *A Bounded-Error
> Dual-Prediction Scheme for Freshness-Preserving Short-Term Load Forecasting
> over LoRa*, ICCKE 2026.

A seven-weight fixed-point linear predictor runs bit-identically on an 8-bit
meter node and on the gateway; the node transmits only when its prediction errs
by more than a per-node threshold, and the gateway imputes the silent hours
with the identical prediction. This repository contains the reference codec,
the policy simulator, every evaluation behind the paper's numbers, the
fixed-point C port with its byte-identical equivalence test, and the frozen
artifacts the paper reports.

## Layout

```
code/tier1_edge_codec/
  edge/        the reference codec: integer-only, no numpy, no float
  sim/         the store, the transmission policies, the reconstruction
  eval/        everything that produces a reported number
  tier2/       the one interface to the frozen forecaster
  vendor/      the forecaster's data and inference code (checksummed)
  export/      Python -> C header generation (golden vectors, configuration)
  firmware/    the C codec and its host equivalence test
  repro/       one script per reported artifact
  inputs/      the pre-trained edge weights
  tests/       the test suite
code/dlinear.pt      the frozen forecaster (md5 630d927e133f1360dd2f2628f1958771)
results_frozen/      the artifacts behind the submitted paper, with MD5s
results_revision/    the artifacts added in the revision, with MD5s
```

## Setup

Python 3.11; `pip install -r requirements-lock.txt`; `gcc` for the C test.

```bash
bash get_data.sh           # downloads the ECL benchmark, builds code/tier2_data/
```

The data is not redistributed. `code/DATA.md` describes the store.

## Reproduce

```bash
cd code/tier1_edge_codec
python -m pytest tests/ -q                      # the test suite (synthetic data)
make -C firmware test                           # C vs Python, byte for byte, 384 h
cd ../.. && bash code/tier1_edge_codec/repro/run_all.sh   # every reported artifact
```

`run_all.sh` writes to `results/` and never to `results_frozen/`; compare the
two with any JSON diff. On a 16-core CPU the full pipeline takes about 40
minutes. The revision's additional analyses:

```bash
cd code/tier1_edge_codec
python -m eval.run_eval ... --per_client_out ../../results/per_client.json   # see repro/configs/tier1_results.yaml
python -m eval.revision_stats --per_client ../../results/per_client.json --results ../../results/tier1_results.json
python -m eval.factorial      --data_dir ../tier2_data --checkpoint ../dlinear.pt
python -m eval.protocol_matrix --data_dir ../tier2_data --clients all --window 400 --stride_exc 5
python -m eval.tap_split      --data_dir ../tier2_data
python -m eval.sync_safe_cost --data_dir ../tier2_data --score   # Section IV-E: sync-safe at matched airtime
python -m eval.factorial_lead1                                   # Section VII-D: next-hour error, reads results_revision/factorial.json
```

## The bench node

`code/tier1_edge_codec/firmware/node/` is the complete node application
(codec, SX127x LoRa driver, scheduling) built for the ATmega328P. It is
released for the memory figures of Section V-F of the paper (8 742 B Flash,
544 B static SRAM, 335 B peak stack). `make -C code/tier1_edge_codec/firmware/node node-deploy`
builds the deployed configuration and prints those two sizes (needs `avr-gcc`);
`make … node` builds the larger bench configuration with its serial reporting.
The stack figure was measured on the bench build by stack painting
(`firmware/node/README.md`).

**It does not implement the paper's protocol.** This node applies its daily
weight update at once, on the reconstructed buffer, without waiting for the
gateway's acknowledgement (an *ungated* node). The protocol the paper reports
is the *gated* one: the node updates only after the resynchronisation is
acknowledged. That protocol is the reference codec in `edge/` and the
simulator, which produce every frame-loss result of the paper
(`eval/protocol_matrix.py`). The bench node is not a reference for frame-loss
behaviour. `firmware/hil/` holds the serial helpers the node build uses.

## Determinism and seeds

The codec and the policy simulation use no random numbers: given the data and
the parameters they have one output. Three things are random, each with a
fixed seed: the packet-loss draw of the frame-loss table (7; the revision
reports five draws), the validation subsample used to select the threshold
multiplier (1), and the origins of the sensitivity analysis (0).

## The forecaster

`code/dlinear.pt` is a decomposition-linear model (25-tap moving-average split,
two linear branches, reversible instance normalisation on a log1p input,
16 178 parameters), trained once and frozen. The checkpoint records its own
training command and environment:

```
20_train_tier2.py --model dlinear --lookback 336 --epochs 60 --steps_per_epoch 1727
  --batch_size 256 --eval_samples 60000 --no_early_stop --seed 0
  --converged_ref 0.7104539905500034 --converged_tol 0.005
```

with AdamW (learning rate 1e-3, 500 warm-up steps, decay to 1e-5, weight decay
1e-4, gradient clipping 1.0), a Huber loss (beta 1) in the normalised log
space, deterministic kernels, torch 2.11 on a GPU. A budget-doubling test
recorded in the checkpoint changed validation MASE by less than 0.1 percent.
Splits are chronological 7:1:2 over 26 304 hourly steps.

## Use of AI assistants

Generative AI assistants (Claude, Anthropic) were used throughout this work:
to help draft and edit the paper and to help write the analysis and
simulation code. Every reported number is produced by the code in this repository; all content was reviewed
and verified by the authors, who take full responsibility for it.

## Citation

See `CITATION.cff` (GitHub shows it as "Cite this repository").

## Licence

MIT (code). The ECL data keeps its own terms — see the UCI repository.
