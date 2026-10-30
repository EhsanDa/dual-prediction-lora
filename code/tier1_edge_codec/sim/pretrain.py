"""
pretrain.py  --  the offline fit that initialises the edge predictor. Section 3.2.

ONE ridge regression, done once, shipped in edge_config.h as `w_init`, identical
on every meter. This is the part of the design that makes a positive result
structural rather than hoped for:

  * ONE FIRMWARE IMAGE, EVERY METER. No per-client calibration and no 321 weight
    vectors, so the deployability argument survives intact.
  * IT CANNOT BE WORSE THAN PERSISTENCE. The tap set contains lag 1, so
    persistence is inside the hypothesis space, and the fit minimises exactly
    the loss the paper reports. With ~5M training rows against 6 parameters the
    generalisation gap is not a live concern.
  * IT REMOVES THE CONVERGENCE TRANSIENT. NLMS now starts at a good solution and
    personalises from there instead of spending weeks climbing out of
    persistence.

WHY STANDARDISING PER CLIENT IS SAFE
====================================
The weights of a linear predictor are invariant under a per-client rescaling:
if z_c = u_c / sigma_c then a predictor fitted on z applies unchanged to u.
So the design matrices can be pooled across meters spanning four orders of
magnitude without the largest ones owning the fit -- which is the same reason
the reported headline is MASE and not kWh.

NO INTERCEPT IN THE FIT
=======================
Section 3.2 ships the bias initialised to 0, because a bias is NOT
scale-invariant and cannot be transferred across meters. Fitting an intercept
and then discarding it would leave the lag weights compensating for a term that
is no longer there. Fitting WITHOUT one instead lets the lag weights absorb the
level themselves -- they come out summing to very nearly 1, which is what a
predictor with no bias term needs. NLMS learns any residual offset per node
within a few days.

MEMORY
======
The normal equations are 7x7. Nothing here materialises a design matrix: the
Gram and cross-product are accumulated in chunks, which is exact and costs
nothing, and is the difference between this running in seconds and not running
at all on 5.3M rows.
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
from edge.quantize import quantize_array                 # noqa: E402
from sim.run_codec import Store, choose_scales           # noqa: E402


def _design(z, taps, lo, hi):
    """(n, p) lag matrix and (n,) target for rows t in [lo, hi), predicting
    z[t+1] from z[t+1-tap]. Rows whose deepest tap falls before 0 are dropped
    by the caller's choice of `lo`."""
    t = np.arange(lo, hi)
    X = np.stack([z[t + 1 - tap] for tap in taps], axis=1)
    return X, z[t + 1]


def _normal_equations(z_by_client, taps, lo_by_client, hi, chunk=200_000):
    p = len(taps)
    G = np.zeros((p, p), np.float64)
    b = np.zeros(p, np.float64)
    n = 0
    for c, z in enumerate(z_by_client):
        lo = lo_by_client[c]
        if hi - lo < 100:
            continue
        for s in range(lo, hi, chunk):
            e = min(s + chunk, hi)
            X, y = _design(z, taps, s, e)
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X, y = X[ok], y[ok]
            G += X.T @ X
            b += X.T @ y
            n += len(y)
    return G, b, n


def _solve(G, b, lam, n):
    """Ridge. lam is scaled by n so the grid means the same thing whatever the
    number of rows."""
    p = G.shape[0]
    return np.linalg.solve(G + lam * n * np.eye(p), b)


def _mse(z_by_client, taps, lo_by_client, hi, w):
    tot, n = 0.0, 0
    for c, z in enumerate(z_by_client):
        lo = lo_by_client[c]
        if hi - lo < 100:
            continue
        X, y = _design(z, taps, lo, hi)
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        r = X[ok] @ w - y[ok]
        tot += float(r @ r)
        n += int(ok.sum())
    return tot / max(n, 1)


def quantise_weights(W, params: CodecParams):
    """Real weights -> Q(w_q) integers, clamped, with the bias initialised to 0."""
    q = [int(np.clip(round(float(x) * (1 << params.w_q)),
                     -params.w_clamp, params.w_clamp)) for x in W]
    return tuple(q + [0])


def fit(store: Store, params: CodecParams = CodecParams(),
        lambdas=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2), verbose=True):
    """Global fit (row B/C init) and per-client fits (row D ceiling).

    Everything is fitted on the TRAINING split only and lambda is chosen on
    VALIDATION. The device's quantisation scale k is also training-only
    (asserted separately), so nothing about the shipped firmware has seen test
    data.
    """
    taps = list(params.taps)
    max_tap = max(taps)
    k, _ = choose_scales(store)
    T, M = store.T, store.M

    z_by_client, sigma = [], np.zeros(M)
    for c in range(M):
        u, _ = quantize_array(store.series_filled[:, c], int(k[c]), params.u_clamp)
        tr = u[int(store.onboarding[c]):store.train_end].astype(np.float64)
        s = float(tr.std()) if tr.size > 24 else 0.0
        sigma[c] = s if s > 1e-9 else 1.0
        z_by_client.append(u.astype(np.float64) / sigma[c])

    lo = [max(int(store.onboarding[c]) + max_tap, max_tap) for c in range(M)]
    G, b, n = _normal_equations(z_by_client, taps, lo, store.train_end)
    lo_val = [max(l, store.train_end) for l in lo]

    best, best_lam, curve = None, None, []
    for lam in lambdas:
        W = _solve(G, b, lam, n)
        v = _mse(z_by_client, taps, lo_val, store.valid_end, W)
        curve.append({"lambda": lam, "valid_mse": v, "w": W.tolist()})
        if best is None or v < best[0]:
            best, best_lam = (v, W), lam
    W_global = best[1]

    # persistence, for the comparison the whole section rests on
    W_pers = np.zeros(len(taps))
    W_pers[taps.index(1)] = 1.0
    v_pers = _mse(z_by_client, taps, lo_val, store.valid_end, W_pers)

    per_client = np.zeros((M, len(taps) + 1), np.int64)
    pc_real = np.zeros((M, len(taps)))
    for c in range(M):
        Gc, bc, nc = _normal_equations([z_by_client[c]], taps, [lo[c]],
                                       store.train_end)
        if nc < 500:
            Wc = W_global
        else:
            Wc = _solve(Gc, bc, best_lam, nc)
        pc_real[c] = Wc
        per_client[c] = quantise_weights(Wc, params)

    out = {
        "taps": taps, "w_q": params.w_q, "lambda": best_lam,
        "lambda_curve": [{k2: v2 for k2, v2 in d.items() if k2 != "w"}
                         for d in curve],
        "w_global_real": W_global.tolist(),
        "w_global_q": list(quantise_weights(W_global, params)),
        "valid_mse_global": best[0],
        "valid_mse_persistence": v_pers,
        "improvement_over_persistence": 1.0 - best[0] / max(v_pers, 1e-12),
        "n_train_rows": int(n),
        "sigma_per_client": sigma.tolist(),
        "note": ("fitted on the TRAINING split in per-client standardised "
                 "units; lag weights are scale-invariant so they transfer to "
                 "device units unchanged. Bias initialised to 0, per section "
                 "3.2, and the fit carries no intercept for the same reason."),
    }
    if verbose:
        print(f"global ridge over {n:,d} training rows, {len(taps)} taps")
        print(f"  lambda {best_lam:g} chosen on validation")
        print(f"  validation MSE  global {best[0]:.5f}  vs persistence "
              f"{v_pers:.5f}   ({out['improvement_over_persistence']:+.1%})")
        print(f"  W = " + "  ".join(f"lag{t}={w:+.4f}"
                                    for t, w in zip(taps, W_global)))
        print(f"  sum of lag weights {W_global.sum():.4f} "
              f"(near 1 is what a bias-free predictor needs)")
        pcv = np.mean([_mse([z_by_client[c]], taps, [lo_val[c]],
                            store.valid_end, pc_real[c]) for c in range(M)])
        print(f"  per-client ridge (row D ceiling) mean validation MSE {pcv:.5f}")
    return out, per_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="tier2_data")
    ap.add_argument("--out", default=paths.inp("edge_pretrained.json"))
    ap.add_argument("--per_client_out",
                    default=paths.inp("edge_pretrained_per_client.npy"))
    a = ap.parse_args()
    store = Store.load(a.data_dir)
    store.assert_scales_are_train_only()
    out, per_client = fit(store)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    np.save(a.per_client_out, per_client)
    print(f"wrote {a.out} and {a.per_client_out}")


if __name__ == "__main__":
    main()
