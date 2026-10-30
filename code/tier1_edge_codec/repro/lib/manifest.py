"""manifest.py -- cache invalidation for _pretrain_cache/.

PLAYBOOK TASK 02 step 4. Phase A established the hazard (docs/reference/module-map.md):
`eval/tap_ablation.py` memoises each tap set's global ridge fit to
`_pretrain_cache/<id>.json` and, on a later run, reloads it if the file merely
*exists*. Nothing checks that the data, the tap lags, `k_mu` or the fitting
code are still the ones the fit was produced under. A stale entry is therefore
reused in silence and changes `w_init` in every downstream row that uses that
tap id, with no warning anywhere in the pipeline.

This module does not change that behaviour -- tap_ablation.py is untouched, as
the task requires. It records, beside each cache entry, the state it was
produced under, so `run_all.sh` can refuse to start when that state has moved.

It never deletes anything. A mismatch is reported and the run stops; removing
a stale entry is the author's decision, per PLAYBOOK standing rule 3.

Usage:
    manifest.py write  <cache_dir> <data_dir> <fitter> [ids...]
    manifest.py adopt  <cache_dir> <data_dir> <fitter> [ids...]
    manifest.py verify <cache_dir> <data_dir> <fitter>

`write` stamps entries this harness just produced: the recorded state is known
to be the state they were fitted under.

`adopt` stamps pre-existing entries whose true origin is UNKNOWN -- the twelve
that arrived from Colab, for instance. It records today's state and flags the
entry `adopted_without_verification`, so drift from here on is caught while
nothing false is asserted about where the fit came from. `verify` keeps warning
about adopted entries until they are genuinely refitted.
"""

import hashlib
import json
import os
import subprocess
import sys

SCHEMA = 1

# Mirrors eval/tap_ablation.py::LADDER. Imported rather than copied where
# possible; the literal below is only the fallback for when the package is not
# importable (e.g. a bare checkout with no numpy).
_FALLBACK_LADDER = {
    "T1": (1,), "T2": (1, 2), "T3": (1, 2, 3), "T4": (1, 24),
    "T5": (1, 2, 24), "T6": (1, 2, 3, 24), "T7": (1, 2, 3, 24, 25),
    "T8": (1, 2, 3, 24, 25, 168), "T9": (1, 2, 3, 24, 25, 48, 168),
    "T10": (1, 2, 3, 24, 25, 168, 169, 336),
    "T11": tuple(range(1, 25)), "T12": tuple(range(1, 25)) + (168,),
}


def ladder():
    try:
        from eval.tap_ablation import LADDER
        return {k: tuple(v) for k, v in LADDER.items()}
    except Exception:
        return dict(_FALLBACK_LADDER)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_fingerprint(data_dir):
    """md5 of every file the fit reads, keyed by basename, sorted."""
    out = {}
    for name in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, name)
        if os.path.isfile(p) and (name.endswith(".npy")
                                  or name == "manifest.json"):
            out[name] = md5(p)
    if not out:
        raise SystemExit(f"manifest: no .npy/manifest.json under {data_dir}")
    return out


def git_blob(path):
    """The committed blob hash, or UNKNOWN. Recorded for traceability only --
    the working-tree md5 below is what verification actually compares, since
    an uncommitted edit must invalidate the cache just as a commit does."""
    try:
        rel = os.path.relpath(os.path.abspath(path))
        out = subprocess.run(["git", "rev-parse", f"HEAD:{rel}"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def build(tap_id, data_dir, fitter, k_mu):
    lags = ladder().get(tap_id)
    return {
        "schema": SCHEMA,
        "tap_id": tap_id,
        "lags": list(lags) if lags else None,
        "k_mu": int(k_mu),
        "data_dir": os.path.abspath(data_dir),
        "data_files_md5": data_fingerprint(data_dir),
        "fitter_path": os.path.relpath(os.path.abspath(fitter)),
        "fitter_md5": md5(fitter),
        "fitter_git_blob": git_blob(fitter),
    }


def _mpath(cache_dir, tap_id):
    return os.path.join(cache_dir, f"{tap_id}.manifest.json")


def cmd_write(cache_dir, data_dir, fitter, k_mu, ids, adopt=False):
    os.makedirs(cache_dir, exist_ok=True)
    if not ids:
        ids = [f[:-5] for f in sorted(os.listdir(cache_dir))
               if f.endswith(".json") and not f.endswith(".manifest.json")]
    n = 0
    for tap_id in ids:
        if not os.path.exists(os.path.join(cache_dir, f"{tap_id}.json")):
            continue                      # no cache entry, nothing to describe
        m = build(tap_id, data_dir, fitter, k_mu)
        if adopt:
            m["adopted_without_verification"] = True
            m["provenance"] = (
                "UNKNOWN. This entry predates manifest tracking; it was not "
                "produced by repro/. The state recorded here is the state at "
                "adoption time, NOT a verified record of the fit's origin. "
                "Refit via repro/tier1_tap_ablation.sh to replace it.")
        else:
            m["adopted_without_verification"] = False
        with open(_mpath(cache_dir, tap_id), "w") as f:
            json.dump(m, f, indent=2)
        n += 1
    verb = "adopted" if adopt else "wrote"
    print(f"manifest: {verb} {n} manifest(s) under {cache_dir}")
    if adopt and n:
        print("manifest: their origin remains UNKNOWN -- verify will keep "
              "saying so until they are refitted")
    return 0


def _diff(old, new):
    """Human-readable list of what moved. Data files are summarised rather
    than dumped, because eight md5s of 30 MB arrays is not a useful error."""
    out = []
    for k in ("tap_id", "lags", "k_mu", "fitter_md5", "data_dir"):
        if old.get(k) != new.get(k):
            out.append(f"      {k}: cached {old.get(k)!r} -> now {new.get(k)!r}")
    o, n = old.get("data_files_md5", {}), new.get("data_files_md5", {})
    if o != n:
        moved = sorted(set(o) | set(n))
        moved = [f for f in moved if o.get(f) != n.get(f)]
        out.append(f"      data files changed: {', '.join(moved)}")
    return out


def cmd_verify(cache_dir, data_dir, fitter, k_mu):
    if not os.path.isdir(cache_dir):
        print(f"manifest: no cache at {cache_dir}; nothing to verify")
        return 0

    entries = [f[:-5] for f in sorted(os.listdir(cache_dir))
               if f.endswith(".json") and not f.endswith(".manifest.json")]
    if not entries:
        print(f"manifest: cache {cache_dir} is empty; nothing to verify")
        return 0

    problems, adopted = [], []
    for tap_id in entries:
        mp = _mpath(cache_dir, tap_id)
        if not os.path.exists(mp):
            problems.append(
                f"  {tap_id}: cache entry exists with NO manifest. It predates "
                f"this check, so\n      the state it was fitted under is "
                f"UNKNOWN and it cannot be trusted.")
            continue
        try:
            with open(mp) as f:
                old = json.load(f)
        except Exception as exc:
            problems.append(f"  {tap_id}: manifest unreadable ({exc})")
            continue
        if old.get("adopted_without_verification"):
            adopted.append(tap_id)
        new = build(tap_id, data_dir, fitter, k_mu)
        d = _diff(old, new)
        if d:
            problems.append(f"  {tap_id}: fitted under different state\n"
                            + "\n".join(d))

    if not problems:
        print(f"manifest: {len(entries)} cache entr"
              f"{'y' if len(entries) == 1 else 'ies'} verified against "
              f"current data, lags, k_mu and fitter")
        if adopted:
            print(f"manifest: WARNING -- {len(adopted)} of them are ADOPTED "
                  f"({', '.join(adopted)}).")
            print("manifest:   Their true origin is UNKNOWN: they were not "
                  "produced by repro/.")
            print("manifest:   Drift from here on is caught, but the baseline "
                  "itself was never verified.")
            print("manifest:   Refit via repro/tier1_tap_ablation.sh to clear "
                  "this warning.")
        return 0

    print("", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print("STALE _pretrain_cache -- REFUSING TO RUN", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print("\n".join(problems), file=sys.stderr)
    print(f"""
Why this is fatal: eval/run_eval.py reads these fits straight into
CodecParams(w_init=...) for every tap-set row, and eval/tap_ablation.py
reloads an existing entry without re-checking anything but its length. A fit
from other data or another fitter would therefore be reported as if it were
current, and nothing downstream would say so.

Nothing has been deleted. Choose deliberately:

  * refit:  rm {cache_dir}/<id>.json {cache_dir}/<id>.manifest.json
            then rerun repro/tier1_tap_ablation.sh
  * keep:   restore the data/fitter the manifest names
  * adopt:  for entries with no manifest at all, if you accept them as-is,
            repro/lib/manifest.py adopt {cache_dir} <data_dir> <fitter>
            This records today's state and flags the entry as unverified. It
            asserts nothing about where the fit actually came from.

Deletion is the author's call, not this script's (PLAYBOOK standing rule 3).
""", file=sys.stderr)
    return 1


def main(argv):
    if len(argv) < 5:
        print(__doc__)
        return 2
    cmd, cache_dir, data_dir, fitter = argv[1], argv[2], argv[3], argv[4]
    rest = argv[5:]
    k_mu = int(os.environ.get("REPRO_K_MU", "5"))
    if cmd == "write":
        return cmd_write(cache_dir, data_dir, fitter, k_mu, rest)
    if cmd == "adopt":
        return cmd_write(cache_dir, data_dir, fitter, k_mu, rest, adopt=True)
    if cmd == "verify":
        return cmd_verify(cache_dir, data_dir, fitter, k_mu)
    print(f"manifest: unknown command {cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
