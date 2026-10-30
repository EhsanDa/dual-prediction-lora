"""
kmu_sweep.py  --  section 2, Fix 2. One axis, three runs, pick one.

The step size is the only free parameter left in the predictor and it is chosen
on VALIDATION hours at alpha = 1. The criterion is codec-level -- packets per
day and mean |e| -- because this is a question about predictor quality, and
routing it through DLinear would add noise without adding information.

WHY THIS SWEEP IS SMALL ON PURPOSE
==================================
The Q12 design had no usable step-size axis at all: at realistic magnitudes the
update quantised to 2 LSBs at k_mu = 2 and to zero by k_mu = 4, so the filter
could only be too fast or frozen. At Q20 the same three values differ by real
factors of two, which is why a three-point sweep is now meaningful and why it
does not need to be any larger.
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

from edge.codec import CodecParams                    # noqa: E402
from sim.run_codec import Store, run_policy           # noqa: E402


def evaluate(store, params, alpha, lo, hi):
    r = run_policy(store, alpha, params, backend="batch")
    win = np.zeros(store.T, bool)
    win[lo:hi] = True
    live = r.active & win[:, None] & ~store.missing
    days = np.maximum((r.active & win[:, None]).sum(0) / 24.0, 1e-9)
    n_res = (r.resync_bytes[lo // 24:hi // 24] > 0).sum(0)
    pkt = (r.tx[lo:hi].sum(0) + n_res) / days
    q = (2.0 ** r.k.astype(np.float64))[None, :]
    d = np.abs(r.u_within - r.u_q) * q
    return {"packets_per_day": float(pkt.mean()),
            "mean_abs_e_units": float(np.abs(r.e[live]).mean()),
            "recon_mae_kwh": float(d[live].mean()),
            "tx_rate": float(r.tx[live].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="tier2_data")
    ap.add_argument("--pretrained", default=paths.inp("edge_pretrained.json"))
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--k_mu", default="4,5,6")
    ap.add_argument("--out", default=paths.result("tier1_kmu_sweep.json"))
    a = ap.parse_args()

    store = Store.load(a.data_dir)
    w = tuple(json.load(open(a.pretrained))["w_global_q"])
    base = CodecParams().with_w_init(w)
    lo, hi = store.train_end, store.valid_end

    print(f"k_mu sweep on VALIDATION hours [{lo:,d}, {hi:,d}) at alpha="
          f"{a.alpha:g}, w_init = the global ridge fit")
    print(f"{'k_mu':>5s} {'mu':>8s} {'pkt/day':>9s} {'mean|e|':>9s} "
          f"{'recon MAE':>10s}")
    rows = {}
    for k in [int(x) for x in a.k_mu.split(",")]:
        r = evaluate(store, CodecParams(**{**base.__dict__, "k_mu": k}),
                     a.alpha, lo, hi)
        rows[k] = r
        print(f"{k:5d} {2.0 ** -k:8.4f} {r['packets_per_day']:9.3f} "
              f"{r['mean_abs_e_units']:9.1f} {r['recon_mae_kwh']:10.3f}")
    # a frozen reference on the same axis, so the sweep is readable
    froz = evaluate(store, CodecParams(**{**base.__dict__, "adapt": False}),
                    a.alpha, lo, hi)
    print(f"{'none':>5s} {'--':>8s} {froz['packets_per_day']:9.3f} "
          f"{froz['mean_abs_e_units']:9.1f} {froz['recon_mae_kwh']:10.3f}"
          f"   (adaptation disabled)")

    best = min(rows, key=lambda k: rows[k]["packets_per_day"])
    gain = 1.0 - rows[best]["packets_per_day"] / froz["packets_per_day"]
    print(f"\n-> k_mu = {best} (mu = {2.0 ** -best:g}), "
          f"{gain:+.1%} packets/day against the frozen global filter")
    if gain <= 0:
        print("   Adaptation contributes nothing measurable here. That is a "
              "finding, not a failure:\n   the shipped model is then the "
              "globally pre-trained filter with adaptation disabled,\n   which "
              "still beats both naive baselines (section 3.3).")
    with open(a.out, "w") as f:
        json.dump({"alpha": a.alpha, "rows": {str(k): v for k, v in rows.items()},
                   "frozen": froz, "selected_k_mu": best,
                   "gain_vs_frozen": gain}, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
