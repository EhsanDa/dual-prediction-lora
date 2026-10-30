"""paths.py  --  where artifacts live. One definition, imported everywhere.

Before this existed, every entry point defaulted its --out to a bare filename,
so running `python -m eval.run_eval` dropped tier1_results.json into whatever
directory you happened to be in -- in practice the package root, which is how
six result JSONs, RESULTS.md and a figures/ tree ended up scattered among the
source. The repro/ scripts always passed explicit paths, so the mess was only
ever visible to someone invoking a module directly.

THE CONTRACT
============
  results/          everything a run PRODUCES. Gitignored, disposable.
  results_frozen/   the artifacts backing the submitted paper. NEVER written.
  inputs/           checked-in inputs a run CONSUMES (the pre-trained weights).

Defaults read from and write to `results/`. That is deliberate and it is the
whole point: one directory, predictable, and never the source tree. To work
from the frozen baseline instead, copy it in first -- repro/seed_from_frozen.sh
does exactly that -- rather than pointing a default at results_frozen/ and
risking a write to it.
"""

import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FROZEN_DIR = os.path.join(REPO_ROOT, "results_frozen")
INPUTS_DIR = os.path.join(PKG_DIR, "inputs")


def result(name):
    """Path for a produced artifact. Callers write here."""
    return os.path.join(RESULTS_DIR, name)


def inp(name):
    """Path for a checked-in input the pipeline consumes."""
    return os.path.join(INPUTS_DIR, name)


def ensure_results():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR
