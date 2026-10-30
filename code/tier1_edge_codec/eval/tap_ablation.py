"""
tap_ablation.py  --  why seven weights. The accuracy/memory trade-off.

A reviewer asked why the edge predictor has seven weights and asked for an
accuracy-versus-memory ablation. This is it.

TWO AXES, NOT ONE, AND THIS IS THE POINT
========================================
    MEMORY is driven by the LARGEST LAG, not by the number of weights. The node
    state is dominated by the ring buffer, whose length is 24 + max_tap. Adding
    lag 4 and lag 5 costs eight bytes; adding lag 336 costs 288.

    ACCURACY and ADAPTATION are driven by the NUMBER OF WEIGHTS. NLMS
    steady-state misadjustment is approximately mu*p/2, linear in filter length,
    and convergence time grows with p as well. With the weights updated once a
    day over twenty-four samples, a long filter never settles.

So the ladder below varies both, and four specific comparisons isolate specific
effects. T8 vs T12 is the decisive one: the SAME 192-slot ring, so the same
dominant memory cost, with seven weights against twenty-six (+76 B). If T8 holds up there, the sparsity is a
statistical choice rather than a memory compromise, which is what section V-A
of the paper claims.

WHY THIS IS A SEPARATE SCRIPT
=============================
The predictor-level numbers -- transmissions, reconstruction error, SRAM --
need no forecaster at all, so a configuration costs one run_policy call rather
than a streaming DLinear pass. The whole ladder runs in the time run_eval.py
takes to score two rows. Forecast-level numbers for a couple of selected tap
sets belong in run_eval.py, not here.

THE TRAP
========
The globally pre-trained weight vector is TAP-SET SPECIFIC. edge_pretrained.json
is fitted for (1, 2, 3, 24, 25, 168) and is meaningless for any other lag set;
using it anyway produces numbers that look entirely plausible and are wrong.
Every tap set therefore gets its own ridge fit here, by the same procedure as
section 3.2, and `--verify_pretrained` checks that the fit reproduces the
shipped vector for the shipped taps before any other row is believed.
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

from edge.codec import CodecParams                       # noqa: E402
from edge.protocol import EXCEPTION_BYTES                # noqa: E402
from sim.run_codec import Store, run_policy              # noqa: E402

# ---------------------------------------------------------------- the ladder
# id -> lag set. buf_len is always 24 + max_tap, so n_carry is zero throughout.
LADDER = {
    "T1":  (1,),
    "T2":  (1, 2),
    "T3":  (1, 2, 3),
    "T4":  (1, 24),
    "T5":  (1, 2, 24),
    "T6":  (1, 2, 3, 24),
    "T7":  (1, 2, 3, 24, 25),
    "T8":  (1, 2, 3, 24, 25, 168),          # the shipped configuration
    "T9":  (1, 2, 3, 24, 25, 48, 168),
    "T10": (1, 2, 3, 24, 25, 168, 169, 336),
    "T11": tuple(range(1, 25)),             # dense, one day
    "T12": tuple(range(1, 25)) + (168,),    # dense + weekly; same buffer as T8
}
PROPOSED = "T8"

# The comparisons that carry the argument, printed as explicit deltas so the
# table does not have to be read carefully to see them.
COMPARISONS = [
    ("T3", "T6",  "the diurnal tap: +1 weight, but the ring must reach lag 24"),
    ("T6", "T7",  "the lag-25 seasonal slope, at essentially no memory cost"),
    ("T7", "T8",  "the weekly tap: +1 weight, +290 B (the decision the paper "
                  "reports reversing)"),
    ("T7", "T11", "6 weights vs 25 at a SMALLER ring"),
    ("T8", "T12", "SAME RING (192 slots, 384 B), 7 weights vs 26 for +76 B "
                  "-- sparsity as a statistical choice, not a memory "
                  "compromise"),
]

# Rates at which the tap sets are compared. COMPARING AT FIXED ALPHA IS WRONG
# AND INVERTS EVERY CONCLUSION: at a fixed deadband a WORSE predictor triggers
# more often, so it transmits more and therefore reconstructs better. The
# project's own rule (section 4.1) is that predictors are compared at matched
# packet rate, and that is what these are.
MATCH_RATES = (6.0, 8.0, 10.0)


def params_for(taps, k_mu, w_init=None):
    """Build the params in ONE construction.

    CodecParams.__post_init__ asserts len(w_init) == n_w, so the weight vector
    cannot be attached after the fact: constructing with a non-default tap set
    and the default seven-entry w_init fails before with_w_init is ever
    reached. It happens to work for the shipped taps, which is exactly the
    configuration an ablation does not stay in.
    """
    taps = tuple(taps)
    if w_init is None:
        # persistence: unit weight on lag 1 if present, zero elsewhere, bias 0
        w = [0] * (len(taps) + 1)
        if 1 in taps:
            w[taps.index(1)] = 1 << 20
        w_init = w
    w_init = tuple(int(v) for v in w_init)
    if len(w_init) != len(taps) + 1:
        raise SystemExit(
            f"w_init has {len(w_init)} entries for {len(taps)} lags; expected "
            f"{len(taps) + 1} (lags + bias). The cached fit and the tap set "
            f"have gone out of step -- delete the cache entry and refit.")
    return CodecParams(taps=taps, buf_len=24 + max(taps), k_mu=k_mu,
                       w_init=w_init)


def node_state_bytes(p):
    """Exact SRAM footprint of codec_t. Mirrors the field list in
    firmware/edge_codec.h; if that struct changes this must change with it."""
    ring = 2 * p.buf_len            # int16 ring of reconstructed values
    day_true = 2 * 24               # int16 true values staged for the RESYNC
    weights = 4 * p.n_w             # int32 Q20, lags + bias
    carry = 2 * p.n_carry
    fixed = 22                      # day_tx, counters, u_hat, s, eps, head
    return ring + day_true + weights + carry + fixed


# ------------------------------------------------------------- the ridge fit
def fit_global(store, taps, lambdas=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)):
    """Section 3.2, for an arbitrary tap set.

    Per client, divide the training slice by that client's own standard
    deviation; the coefficients of a homogeneous linear predictor are invariant
    under that rescaling, so weights fitted in normalised units transfer to
    device units unchanged and clients spanning four orders of magnitude
    contribute equally. Pool the standardised normal equations across clients
    and solve. NO INTERCEPT: a bias is not scale-invariant across meters, and
    omitting it lets the lag weights absorb the level themselves, which is why
    they sum to essentially unity.
    """
    taps = tuple(taps)
    p = len(taps)
    mx = max(taps)
    XtX = np.zeros((p, p), np.float64)
    Xty = np.zeros(p, np.float64)
    vX, vy = [], []
    n_rows = 0

    for c in range(store.M):
        t0 = max(int(store.onboarding[c]) + mx, mx)
        if t0 >= store.train_end:
            continue
        col = store.series_filled[:, c].astype(np.float64)
        sigma = float(np.std(col[t0:store.train_end]))
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        z = col / sigma

        tr = np.arange(t0, store.train_end)
        Xtr = np.stack([z[tr - l] for l in taps], axis=1)
        ytr = z[tr]
        XtX += Xtr.T @ Xtr
        Xty += Xtr.T @ ytr
        n_rows += tr.size

        va = np.arange(max(store.train_end, t0), store.valid_end)
        if va.size:
            vX.append(np.stack([z[va - l] for l in taps], axis=1))
            vy.append(z[va])

    if n_rows == 0:
        raise SystemExit("no training rows for taps %r" % (taps,))
    vX = np.concatenate(vX) if vX else None
    vy = np.concatenate(vy) if vy else None

    best, curve = None, []
    for lam in lambdas:
        w = np.linalg.solve(XtX + lam * n_rows * np.eye(p), Xty)
        if vX is None:
            mse = float("nan")
        else:
            mse = float(np.mean((vX @ w - vy) ** 2))
        curve.append({"lambda": lam, "valid_mse": mse})
        if best is None or mse < best[1]:
            best = (w, mse, lam)
    w, mse, lam = best

    # persistence reference on the same validation rows, for context
    pers = (float(np.mean((vX[:, 0] - vy) ** 2))
            if vX is not None and 1 in taps else float("nan"))

    scale = 1 << 20
    clamp = 4 << 20
    wq = [int(np.clip(round(x * scale), -clamp, clamp)) for x in w]
    return {"taps": list(taps), "w_real": [float(x) for x in w],
            "w_global_q": wq + [0],            # + bias, initialised at zero
            "lambda": lam, "valid_mse": mse, "valid_mse_persistence": pers,
            "sum_lag_weights": float(np.sum(w)), "n_train_rows": int(n_rows),
            "lambda_curve": curve}


# -------------------------------------------------------------- measurement
def measure(store, res, params):
    """Predictor-level numbers, straight off the PolicyResult."""
    act = res.active & ~store.missing
    hours = act.sum(axis=0).astype(np.float64)
    days = np.maximum(hours / 24.0, 1e-9)

    n_exc = (res.tx & act).sum(axis=0).astype(np.float64)
    n_res = (res.resync_bytes > 0).sum(axis=0).astype(np.float64)
    pkt_day = (n_exc + n_res) / days
    byte_day = (n_exc * EXCEPTION_BYTES
                + res.resync_bytes.sum(axis=0).astype(np.float64)) / days

    d = np.abs(res.u_within - res.u_q)
    viol = int((d > res.eps_eff)[act].sum())
    kw = d.astype(np.float64) * (2.0 ** res.k.astype(np.float64))[None, :]
    mae = np.array([kw[act[:, c], c].mean() if act[:, c].any() else 0.0
                    for c in range(store.M)])
    scale = np.asarray(store.mase_scale, np.float64)
    ok = np.isfinite(scale) & (scale > 0) & (hours > 0)
    rel = mae[ok] / scale[ok]
    mean_abs_e = float(np.abs(res.e)[act].mean()) if act.any() else 0.0

    return {"packets_per_day": float(pkt_day[ok].mean()),
            "bytes_per_day": float(byte_day[ok].mean()),
            "recon_mae_kwh": float(mae[ok].mean()),
            "recon_mae_over_scale_mean": float(rel.mean()),
            "recon_mae_over_scale_p90": float(np.percentile(rel, 90)),
            "mean_abs_e_units": mean_abs_e,
            "linf_violations": viol,
            "n_clients_scored": int(ok.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="synth_data")
    ap.add_argument("--alphas", default="0.5,1,2")
    ap.add_argument("--k_mu", type=int, default=5)
    ap.add_argument("--tap_sets", default="all",
                    help="comma-separated ids, e.g. T8,T12")
    ap.add_argument("--pretrain_cache", default="_pretrain_cache")
    ap.add_argument("--pretrained", default=paths.inp("edge_pretrained.json"),
                    help="the shipped fit; used only as a correctness check")
    ap.add_argument("--max_clients", type=int, default=0,
                    help="0 = all. A small value gives a fast smoke "
                         "run; results are NOT reportable.")
    ap.add_argument("--out", default=paths.result("tier1_tap_ablation.json"))
    a = ap.parse_args()

    ids = (list(LADDER) if a.tap_sets == "all"
           else [s.strip() for s in a.tap_sets.split(",") if s.strip()])
    unknown = [i for i in ids if i not in LADDER]
    if unknown:
        ap.error(f"unknown tap set(s): {unknown}")
    alphas = [float(s) for s in a.alphas.split(",")]

    store = Store.load(a.data_dir)
    store.assert_scales_are_train_only()
    os.makedirs(a.pretrain_cache, exist_ok=True)

    # ---- agreement check against the shipped fit --------------------------
    # NOT exact equality: two least-squares implementations will not agree bit
    # for bit across a different ridge convention or a few rows' difference in
    # the training window, and demanding it would reject a fit that is for all
    # practical purposes the same solution. What matters is (a) that the
    # agreement is close enough that T8 here behaves like the shipped model,
    # and (b) that EVERY row below uses this one fitter, so the comparison
    # between tap sets is internally consistent.
    W_SCALE = float(1 << 20)
    TOL_REAL = 5e-3            # absolute, in real weight units
    if os.path.exists(a.pretrained):
        shipped = json.load(open(a.pretrained))
        if tuple(shipped.get("taps", ())) == LADDER[PROPOSED]:
            got = np.asarray(fit_global(store, LADDER[PROPOSED])["w_global_q"],
                             dtype=np.float64)
            want = np.asarray(shipped["w_global_q"], dtype=np.float64)
            dev = np.abs(got - want) / W_SCALE
            worst = float(dev.max())
            ok = worst <= TOL_REAL
            print(f"agreement with {a.pretrained} ({PROPOSED}): "
                  f"max |dw| = {worst:.2e} in real units "
                  f"({'OK' if ok else 'TOO LARGE'}, tolerance {TOL_REAL:g})")
            if not ok:
                print(f"  shipped {want.astype(int).tolist()}")
                print(f"  refit   {got.astype(int).tolist()}")
                print("  That is a different solution, not a numerical "
                      "difference. Resolve it before\n  reading any row below.")
            else:
                print("  The two fits are the same solution up to the ridge "
                      "convention. All rows below\n  use the fitter in this "
                      "file, so they are comparable with each other; T8 will "
                      "differ\n  from the paper's C row in the third decimal.")

    fits, rows = {}, {}
    print(f"\nfitting a global ridge per tap set ({len(ids)} sets)")
    for tid in ids:
        taps = LADDER[tid]
        cache = os.path.join(a.pretrain_cache, f"{tid}.json")
        if os.path.exists(cache):
            fit = json.load(open(cache))
        else:
            fit = fit_global(store, taps)
            json.dump(fit, open(cache, "w"), indent=2)
        fits[tid] = fit
        print(f"  {tid:>4s}  {str(taps):<34s} lambda {fit['lambda']:g}  "
              f"valid MSE {fit['valid_mse']:.5f}  sum(w) {fit['sum_lag_weights']:.4f}")

    print(f"\n{'id':>5s} {'lags':<34s} {'w':>3s} {'maxlag':>7s} {'buf':>5s} "
          f"{'SRAM B':>7s} {'alpha':>6s} {'pkt/day':>8s} {'MAE/scale':>10s} "
          f"{'mean|e|':>8s} {'viol':>5s}")
    for tid in ids:
        taps = LADDER[tid]
        p0 = params_for(taps, a.k_mu, fits[tid]["w_global_q"])
        assert p0.n_carry == 0, f"{tid}: n_carry {p0.n_carry}, expected 0"
        sram = node_state_bytes(p0)
        rows[tid] = {"taps": list(taps), "n_weights": p0.n_w,
                     "max_lag": p0.max_tap, "buf_len": p0.buf_len,
                     "sram_bytes": sram, "proposed": tid == PROPOSED,
                     "by_alpha": {}}
        sub = (list(range(min(a.max_clients, store.M)))
               if a.max_clients else None)
        for al in alphas:
            print(f"    running {tid} alpha={al:g} ...",
                  end="\r", flush=True)
            r = run_policy(store, al, p0, backend="batch",
                           clients=sub)
            m = measure(store, r, p0)
            rows[tid]["by_alpha"][f"{al:g}"] = m
            assert m["linf_violations"] == 0, (
                f"{tid}@{al:g}: {m['linf_violations']} L-inf violations")
            print(" " * 40, end="\r")
            star = "*" if tid == PROPOSED else " "
            print(f"{tid + star:>5s} {str(taps):<34s} {p0.n_w:3d} "
                  f"{p0.max_tap:7d} {p0.buf_len:5d} {sram:7d} {al:6g} "
                  f"{m['packets_per_day']:8.2f} "
                  f"{m['recon_mae_over_scale_mean']:10.4f} "
                  f"{m['mean_abs_e_units']:8.1f} {m['linf_violations']:5d}",
                  flush=True)

    # ---- matched packet rate: the only comparison that means anything -----
    def at_rate(tid, rate):
        pts = [(m["packets_per_day"], m["recon_mae_over_scale_mean"])
               for m in rows[tid]["by_alpha"].values()]
        p = np.array([x[0] for x in pts])
        q = np.log(np.array([x[1] for x in pts]))
        o = np.argsort(p)
        if rate < p[o][0] or rate > p[o][-1]:
            return float("nan")
        return float(np.exp(np.interp(rate, p[o], q[o])))

    for tid in ids:
        rows[tid]["at_rate"] = {f"{r:g}": at_rate(tid, r) for r in MATCH_RATES}

    print("\nreconstruction error / mase_scale AT MATCHED PACKET RATE "
          "(lower is better)")
    print("comparing at fixed alpha would invert this: a worse predictor "
          "triggers more often,\nso it transmits more and reconstructs better. "
          "Rate is the axis, not the deadband.")
    print(f"{'id':>5s} {'w':>3s} {'SRAM':>6s} {'validMSE':>9s} "
          + "".join(f"{'@' + f'{r:g}':>9s}" for r in MATCH_RATES))
    for tid in ids:
        r = rows[tid]
        star = "*" if tid == PROPOSED else " "
        cells = "".join(
            (f"{r['at_rate'][f'{x:g}']:9.4f}"
             if np.isfinite(r["at_rate"][f"{x:g}"]) else f"{'-':>9s}")
            for x in MATCH_RATES)
        print(f"{tid + star:>5s} {r['n_weights']:3d} {r['sram_bytes']:6d} "
              f"{fits[tid]['valid_mse']:9.5f}" + cells)

    ref = f"{MATCH_RATES[len(MATCH_RATES) // 2]:g}"
    print(f"\ncomparisons that carry the argument (at {ref} packets/day)")
    for lo, hi, why in COMPARISONS:
        if lo not in rows or hi not in rows:
            continue
        e0, e1 = rows[lo]["at_rate"][ref], rows[hi]["at_rate"][ref]
        if not (np.isfinite(e0) and np.isfinite(e1)):
            print(f"  {lo} -> {hi}: outside the swept range at {ref} pkt/day")
            continue
        db = rows[hi]["sram_bytes"] - rows[lo]["sram_bytes"]
        print(f"  {lo} -> {hi}: err {e0:.4f} -> {e1:.4f} "
              f"({(e1 - e0) / e0:+.1%}), SRAM {db:+d} B   ({why})")

    print(f"\nPareto frontier at {ref} packets/day (nothing to its left is "
          f"both smaller AND better)")
    cand = sorted(((rows[t]["sram_bytes"], rows[t]["at_rate"][ref], t)
                   for t in ids if np.isfinite(rows[t]["at_rate"][ref])))
    best = float("inf")
    for sram, err, tid in cand:
        on = err < best
        if on:
            best = err
        print(f"  {tid:>4s}  {sram:5d} B  err {err:.4f}"
              f"{'   PARETO' if on else ''}"
              f"{'   <-- shipped' if tid == PROPOSED else ''}")

    json.dump({"alphas": alphas, "k_mu": a.k_mu, "proposed": PROPOSED,
               "ladder": {k: list(v) for k, v in LADDER.items()},
               "fits": fits, "rows": rows},
              open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    print("If the knee is not at T8 -- in particular if T12 beats it -- that is "
          "a real finding\nand the paper's sparsity argument needs revising. "
          "Report it either way.")


if __name__ == "__main__":
    main()
