#!/usr/bin/env python3
"""assert_params.py -- a run must use the parameters the frozen baseline used.

WHY THIS EXISTS
===============
Twice a `repro/` config restated a script default, restated it WRONGLY, and
failed silently:

  * `tier1_tap_forecast.yaml` set `rates: "15,10"` and dropped the 6.0 that
    eval/tap_forecast_table.py defaults to.
  * `tier1_tap_ablation.yaml` set `alphas: "0.5,1"` and dropped the 2 that
    eval/tap_ablation.py defaults to, emptying every by_alpha.2 block and
    NaN-ing a published column.

Both were caught by hand, long afterwards. See
docs/findings/2026-09-14-repro-config-alphas.md.

THE CHECK THAT ACTUALLY CATCHES THAT, AND THE ONE THAT DOES NOT
===============================================================
That finding recommends asserting the artifact's recorded parameters against
the config claiming to have produced it. **That check is nearly vacuous**, and
it was implemented and measured to be so before this file was rewritten: when a
config states a wrong value the script honours it and the artifact faithfully
records the wrong value, so config and artifact agree and the guard passes. It
catches only a stale artifact or a flag the script ignored.

The check with teeth compares the produced artifact against the FROZEN one.
The frozen artifact is the paper's run, so any parameter the reproduction
changes is a difference from the baseline by definition -- which is exactly
what both historical defects were. Both checks run here; the frozen comparison
is the one that fails a stage.

USAGE
=====
    assert_params.py <artifact.json> [--frozen <frozen.json>] key=expected ...

`key` is a top-level key of the artifact, or a dotted path. Exit 0 on
agreement, 4 on a mismatch, 5 if the artifact or a key is missing. A `--frozen`
path that does not exist is reported and skipped, not failed: stages with no
frozen counterpart are legitimate.

COMPARISON
==========
Values arrive from the shell as strings, so comparison is by normalised form:
a comma-separated list of numbers, then a single number, then stripped strings.
So "0.5,1,2" == [0.5, 1, 2], "5" == 5.0, and "0" == False.

No dependency beyond the standard library: repro/ is parsed in awk precisely
because this project does not carry PyYAML, and this file keeps that promise.
"""
import json
import sys


def _numbers(v):
    """The value as a list of floats, or None if it is not numeric."""
    if isinstance(v, bool):
        return [float(v)]
    if isinstance(v, (int, float)):
        return [float(v)]
    if isinstance(v, (list, tuple)):
        try:
            return [float(x) for x in v]
        except (TypeError, ValueError):
            return None
    if isinstance(v, str):
        parts = [p.strip() for p in v.split(",") if p.strip() != ""]
        if not parts:
            return None
        try:
            return [float(p) for p in parts]
        except ValueError:
            return None
    return None


def equivalent(expected, actual):
    a, b = _numbers(expected), _numbers(actual)
    if a is not None and b is not None:
        return len(a) == len(b) and all(x == y for x, y in zip(a, b))
    return str(expected).strip() == str(actual).strip()


def lookup(doc, path):
    """Top-level key, or a dotted path. Raises KeyError naming what is missing."""
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def _load(path):
    with open(path) as f:
        return json.load(f)


def main(argv):
    argv = list(argv[1:])
    frozen_path = None
    if "--frozen" in argv:
        i = argv.index("--frozen")
        try:
            frozen_path = argv[i + 1]
        except IndexError:
            print("assert_params: --frozen needs a path", file=sys.stderr)
            return 2
        del argv[i:i + 2]
    if len(argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    artifact, pairs = argv[0], argv[1:]
    try:
        doc = _load(artifact)
    except OSError as e:
        print(f"assert_params: cannot read {artifact}: {e}", file=sys.stderr)
        return 5

    frozen = None
    if frozen_path:
        try:
            frozen = _load(frozen_path)
        except OSError:
            print(f"repro: no frozen counterpart at {frozen_path} -- "
                  f"baseline comparison skipped")

    drift, stale = [], []
    for pair in pairs:
        if "=" not in pair:
            print(f"assert_params: not a key=value pair: {pair!r}",
                  file=sys.stderr)
            return 2
        key, expected = pair.split("=", 1)
        try:
            actual = lookup(doc, key)
        except KeyError:
            print(f"assert_params: {artifact} has no key {key!r} -- the "
                  f"artifact does not record it, so nothing can be checked "
                  f"against it. Drop the assertion or record the parameter.",
                  file=sys.stderr)
            return 5
        # (a) did the script honour the config, and is this artifact this run's?
        if not equivalent(expected, actual):
            stale.append((key, expected, actual))
        # (b) THE CHECK WITH TEETH: does this run match the paper's run?
        if frozen is not None:
            try:
                base = lookup(frozen, key)
            except KeyError:
                continue
            if not equivalent(base, actual):
                drift.append((key, base, actual))

    if stale:
        print(f"\nassert_params: {artifact} does not match the config that "
              f"just ran", file=sys.stderr)
        for key, expected, actual in stale:
            print(f"  {key}: config says {expected!r}, "
                  f"artifact recorded {actual!r}", file=sys.stderr)
        print("\nEither the script ignored the flag, or this artifact is left\n"
              "over from an earlier run. Do not use it.\n", file=sys.stderr)
        return 4

    if drift:
        print(f"\nassert_params: PARAMETER DRIFT FROM THE FROZEN BASELINE\n"
              f"  produced: {artifact}\n  baseline: {frozen_path}",
              file=sys.stderr)
        for key, base, actual in drift:
            print(f"  {key}: baseline {base!r}, this run {actual!r}",
                  file=sys.stderr)
        print("\nThis run did NOT use the parameters the paper's run used, so\n"
              "its output is not a reproduction of the frozen artifact. The\n"
              "usual cause is a config restating a script default and getting\n"
              "it wrong -- see docs/findings/2026-09-14-repro-config-alphas.md.\n"
              "Reconcile the config before trusting anything downstream.\n",
              file=sys.stderr)
        return 4

    where = " and the frozen baseline" if frozen is not None else ""
    print(f"repro: params verified against {artifact.split('/')[-1]}{where} "
          f"({len(pairs)} checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
