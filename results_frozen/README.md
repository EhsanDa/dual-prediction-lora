# FROZEN — do not modify

Copied on **2026-09-14** from `code/tier1_edge_codec/` at commit
**`87fa6598f1edd66f1c1c46dead15778910ce78f0`**.
These are the artifacts the submitted conference paper's numbers came from.

Verify integrity:

    md5sum -c CHECKSUMS.md5

23 files: the six `tier1_*.json`, `RESULTS.md`, the two `edge_pretrained.*`
predictor-initialisation files, and `figures/` (5 figures × PDF+PNG, 3 CSV
tables, `captions.tex`).

## Provenance, stated honestly

**This directory was created by an agent (Claude), not by hand.** PLAYBOOK.md
§0.2 asked the author to do it manually so the freeze would be theirs; the
author explicitly delegated it on 2026-09-14. The copy is verifiable: every
file is byte-identical to its source, and `md5sum -c CHECKSUMS.md5` proves it.
The originals were left in place, not moved.

**Freezing does not make these numbers correct — it makes them stable**, so
that later reproduction has a fixed thing to diff against. Whether they are
correct is what the full TASK 04 reproduction determines, and that has not yet
been run (see `docs/status/reproduction.md`).

What is known about how these were produced, as of the freeze:

- They came from `code/runner.ipynb` on Colab, now archived at
  `_attic/runner.ipynb`.
- **That notebook is not a coherent record of a single run.** Its cell 9
  output matches `tier1_results.json` exactly, but its cell 5 output reports
  `138 passed` for a test suite that cannot have passed against the same code.
  At least two sessions are interleaved. See `docs/reference/provenance.md`.
- `edge_pretrained.json` and `edge_pretrained_per_client.npy` are inputs to
  `eval/run_eval.py`, not outputs of the notebook. The invocation of
  `sim/pretrain.py` that produced them is **UNKNOWN** and is not recorded
  anywhere in this repository.
- `RESULTS.md:71` contains a literal `nan pkt/day`, faithfully reproducing a
  real defect in `eval/report_tables.py` at the time it was generated. It is
  preserved here exactly as it was. The defect is fixed in the working tree
  (see `docs/findings/2026-09-14-budget-test.md`), so a regenerated `RESULTS.md` will
  legitimately differ on that line.

## Rules

Never modify anything in this directory. If a file here looks wrong, report it
and leave it alone — it is the evidentiary baseline for the conference
submission. Reproductions write to `results/`, never here.
