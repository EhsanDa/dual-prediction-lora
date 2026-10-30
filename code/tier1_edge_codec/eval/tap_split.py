"""
tap_split.py  --  Table VII's `err` column, split by the hours it is measured on.
Conference revision, request D (2026-09-29).

Reviewer 3 asked whether the tap set was fixed on validation before any test
inspection. Table VII's `err` column (reconstruction error / client scale at 8
packets per day) is measured by eval/tap_ablation.py over EVERY active hour of
the series -- train, validation and test together. This recomputes it three
ways from the same runs:

    all     every active hour -- must reproduce Table VII exactly (the check)
    valid   hours [train_end, valid_end) only -- the choice-relevant column
    test    hours [valid_end, T) only

The runs are eval/tap_ablation.py's own: its LADDER, its params_for, its
verified ridge fits in _pretrain_cache/ (repro/lib/manifest.py), the same
alphas, and its log-interpolation onto the matched packet rate. Nothing is
refitted. Reads only; writes one JSON.

    python -m eval.tap_split --data_dir ../tier2_data \\
        --out ../../results/revision/tap_split.json
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import paths                                                      # noqa: E402

from eval.tap_ablation import LADDER, params_for                  # noqa: E402
from sim.run_codec import Store, run_policy                       # noqa: E402


def measure_window(store, res, lo, hi):
    """eval/tap_ablation.py::measure restricted to hours [lo, hi): the same
    packet count and the same per-client reconstruction error / mase_scale."""
    win = np.zeros(store.T, bool)
    win[lo:hi] = True
    act = res.active & ~store.missing & win[:, None]
    hours = act.sum(axis=0).astype(np.float64)
    days = np.maximum(hours / 24.0, 1e-9)
    n_exc = (res.tx & act).sum(axis=0).astype(np.float64)
    # a RESYNC belongs to the day it closes: count those whose boundary hour
    # lies in the window (tap_ablation counts every resync over the series)
    d_lo, d_hi = -(-lo // 24), -(-hi // 24)
    n_res = (res.resync_bytes[d_lo:d_hi] > 0).sum(axis=0).astype(np.float64)
    pkt_day = (n_exc + n_res) / days
    d = np.abs(res.u_within - res.u_q)
    kw = d.astype(np.float64) * (2.0 ** res.k.astype(np.float64))[None, :]
    mae = np.array([kw[act[:, c], c].mean() if act[:, c].any() else 0.0
                    for c in range(store.M)])
    scale = np.asarray(store.mase_scale, np.float64)
    ok = np.isfinite(scale) & (scale > 0) & (hours > 0)
    return {"packets_per_day": float(pkt_day[ok].mean()),
            "recon_mae_over_scale_mean": float((mae[ok] / scale[ok]).mean())}


def at_rate(points, rate):
    """eval/tap_ablation.py's matched-rate rule: log-linear in the error."""
    p = np.array([x[0] for x in points])
    q = np.log(np.array([x[1] for x in points]))
    o = np.argsort(p)
    if rate < p[o][0] or rate > p[o][-1]:
        return float("nan")
    return float(np.exp(np.interp(rate, p[o], q[o])))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="../tier2_data")
    ap.add_argument("--alphas", default="0.5,1,2")
    ap.add_argument("--k_mu", type=int, default=5)
    ap.add_argument("--rate", type=float, default=8.0)
    ap.add_argument("--pretrain_cache", default="_pretrain_cache")
    ap.add_argument("--frozen", default=os.path.join(paths.FROZEN_DIR,
                                                     "tier1_tap_ablation.json"))
    ap.add_argument("--out", default=paths.result("tap_split.json"))
    a = ap.parse_args(argv)

    store = Store.load(a.data_dir)
    alphas = [float(x) for x in a.alphas.split(",")]
    windows = {"all": (0, store.T),
               "valid": (store.train_end, store.valid_end),
               "test": (store.valid_end, store.T)}
    frozen = json.load(open(a.frozen))
    out = {"rate_pkt_per_day": a.rate, "alphas": alphas, "windows": windows,
           "rows": {}}
    print(f"{'id':>4s} {'lags':<28s} {'all':>8s} {'frozen':>8s} "
          f"{'valid':>8s} {'test':>8s}")
    for tid, taps in LADDER.items():
        fit = json.load(open(os.path.join(a.pretrain_cache, f"{tid}.json")))
        p = params_for(taps, a.k_mu, fit["w_global_q"])
        pts = {w: [] for w in windows}
        for al in alphas:
            r = run_policy(store, al, p, backend="batch")
            for w, (lo, hi) in windows.items():
                m = measure_window(store, r, lo, hi)
                pts[w].append((m["packets_per_day"], m["recon_mae_over_scale_mean"]))
        row = {w: at_rate(pts[w], a.rate) for w in windows}
        row["frozen_all"] = frozen["rows"][tid]["at_rate"][f"{a.rate:g}"]
        row["valid_mse"] = fit["valid_mse"]
        out["rows"][tid] = row
        print(f"{tid:>4s} {str(taps):<28.28s} {row['all']:8.4f} "
              f"{row['frozen_all']:8.4f} {row['valid']:8.4f} {row['test']:8.4f}",
              flush=True)
    rank = lambda w: sorted(out["rows"], key=lambda t: out["rows"][t][w])
    out["ranking"] = {w: rank(w) for w in ("all", "valid", "test")}
    out["check_all_reproduces_frozen"] = all(
        abs(v["all"] - v["frozen_all"]) < 1e-9 for v in out["rows"].values())
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nall-hours column reproduces the frozen Table VII: "
          f"{out['check_all_reproduces_frozen']}")
    for w in ("all", "valid", "test"):
        print(f"  ranking on {w:>5s}: {out['ranking'][w]}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
