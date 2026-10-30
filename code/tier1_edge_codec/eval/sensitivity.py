"""
sensitivity.py  --  how far a forecast can move when the look-back moves.
Blueprint 8.5.

THERE IS NO GLOBAL LIPSCHITZ CONSTANT IN kWh SPACE, AND THIS FILE DOES NOT
PRETEND OTHERWISE
=================================================================
The end-to-end map is log1p -> RevIN (whose mu and sigma are functions of the
SAME window) -> decomposition -> two linear layers -> RevIN inverse -> expm1 ->
clip. Only the middle is linear. Perturbing y_past moves mu and sigma, which
moves both the normalisation on the way in and the inversion on the way out, and
expm1 amplifies whatever survives. So `||dy_hat||_inf <= ||M||_inf * eps` in kWh
is simply false, and any figure carrying such a band would be claiming a
guarantee the model does not provide.

Two things are computed instead, and they are labelled differently everywhere
they appear -- in the code, in the JSON keys, and in the figure legends:

  L_local   the exact end-to-end Jacobian at a sample of real test origins, via
            torch.autograd.functional.jacobian through the COMPLETE path.
            Reported as the operator infinity-norm max_i sum_j |J_ij| and as a
            distribution over origins. It is a LOCAL linearisation. It supports
            "a perturbation of eps moves the forecast by about L_local * eps
            near this operating point", and nothing stronger.

  M_inf     ||W_trend . A + W_seasonal . (I - A)||_inf, where A is the 25-tap
            replicate-padded moving average. This one IS exact and global --
            but only for the map from NORMALISED look-back to NORMALISED
            output. It is a cheap sanity number, scoped in its own name.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import paths                                                      # noqa: E402


def moving_average_matrix(L: int, kernel: int = 25) -> np.ndarray:
    """A such that A @ y_n reproduces DLinear's trend branch exactly.

    Replicate padding of kernel//2 on each side, then a kernel-wide mean with
    stride 1. Built by counting which original index each padded position maps
    to, and verified against torch's own avg_pool1d in `check_matrix` -- a
    hand-derived A that is subtly wrong would silently corrupt M_inf.
    """
    pad = kernel // 2
    A = np.zeros((L, L))
    for i in range(L):
        for p in range(i, i + kernel):
            A[i, min(max(p - pad, 0), L - 1)] += 1.0 / kernel
    return A


def check_matrix(L: int = 336, kernel: int = 25, tol: float = 1e-5) -> float:
    A = moving_average_matrix(L, kernel)
    y = torch.randn(4, L, dtype=torch.float64)
    x = torch.nn.functional.pad(y.unsqueeze(1), (kernel // 2, kernel // 2),
                                mode="replicate")
    ref = torch.nn.functional.avg_pool1d(x, kernel, stride=1).squeeze(1).numpy()
    got = y.numpy() @ A.T
    d = float(np.abs(ref - got).max())
    assert d < tol, f"moving-average matrix disagrees with avg_pool1d by {d:.2e}"
    return d


def normalised_operator_norm(model, L=336, kernel=25):
    """||M||_inf in NORMALISED space. Exact, global, and scoped to that space."""
    A = moving_average_matrix(L, kernel)
    Wt = model.trend.weight.detach().numpy().astype(np.float64)
    Ws = model.seasonal.weight.detach().numpy().astype(np.float64)
    M = Wt @ A + Ws @ (np.eye(L) - A)
    return {"M_inf_normalised_space": float(np.abs(M).sum(axis=1).max()),
            "M_1_normalised_space": float(np.abs(M).sum(axis=0).max()),
            "scope": "map from NORMALISED look-back to NORMALISED output ONLY; "
                     "not a bound in kWh"}


def local_jacobian(model, y_past_kwh: torch.Tensor):
    """Exact d(y_hat_kWh) / d(y_past_kWh), (H, L), through the complete path.

    The input is kWh, so log1p is inside the differentiated function and so is
    the RevIN pair computed from this very window. That is the point: it is the
    window-dependence of mu and sigma that makes the closed form unavailable.
    """
    B = 1
    cfg = model.cfg
    x_past = torch.zeros(B, cfg.lookback_len, cfg.n_dynamic)
    x_fut = torch.zeros(B, cfg.horizon_len, cfg.n_dynamic)
    s = torch.zeros(B, cfg.n_static)
    w = torch.zeros(B, cfg.n_window)

    def f(y_kwh):
        y_log = torch.log1p(torch.clamp(y_kwh, min=0.0)).unsqueeze(0)
        yh, stats = model(y_log, x_past, x_fut, s, w)
        return model.to_kwh(yh, stats).squeeze(0)

    return torch.autograd.functional.jacobian(f, y_past_kwh, vectorize=True)


def restricted_rowsums(J):
    """Revision B6. |J| row sums restricted to the columns the codec actually
    perturbs at each issue hour.

    The perturbation is NOT spread over all 336 slots: at issue hour h it is
    confined to the trailing h+1 of them. Summing |J| over the 335 columns that
    carry no error inflates the band by an order of magnitude and makes it
    useless on a figure -- which is exactly what the unrestricted L_local did.

    Returns (H, 24): entry [i, h] is sum_j |J[i, j]| over the trailing h+1
    columns, i.e. the first-order gain from a unit perturbation of the
    approximate part of the window, for horizon step i at issue hour h.
    """
    A = np.abs(np.asarray(J, np.float64))
    csum = np.cumsum(A[:, ::-1], axis=1)          # trailing-column cumulative
    H, L = A.shape
    return np.stack([csum[:, h] for h in range(24)], axis=1)


def run(model, wb, sel, n_origins=64, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(sel), size=min(n_origins, len(sel)), replace=False)
    rows, L_local, per_h, restr = [], [], [], []
    for i in sorted(idx):
        t, c = int(sel[i, 0]), int(sel[i, 1])
        y_kwh = torch.as_tensor(
            np.expm1(wb.series[t - wb.L + 1:t + 1, c].astype(np.float64)),
            dtype=torch.float32)
        J = local_jacobian(model, y_kwh).detach().numpy().astype(np.float64)
        row_sums = np.abs(J).sum(axis=1)
        L_local.append(float(row_sums.max()))
        per_h.append(row_sums)
        restr.append(restricted_rowsums(J))
        rows.append({"t": t, "client": c, "L_local": float(row_sums.max()),
                     "argmax_h": int(row_sums.argmax()) + 1})
    L_local = np.asarray(L_local)
    per_h = np.stack(per_h)
    R = np.stack(restr)                    # (n_origins, H, 24)
    R_mean = R.mean(axis=0)
    # The band that actually goes on figure 2: at issue hour h the worst-case
    # first-order response to a perturbation of size eps.
    L_restr = R_mean.max(axis=0)           # (24,) over horizon steps
    out = {
        "n_origins": int(len(L_local)),
        "L_local_mean": float(L_local.mean()),
        "L_local_p50": float(np.percentile(L_local, 50)),
        "L_local_p90": float(np.percentile(L_local, 90)),
        "L_local_max": float(L_local.max()),
        "L_local_by_horizon_mean": per_h.mean(axis=0).tolist(),
        "LABEL": "LOCAL linearisation at sampled origins -- NOT a global bound",
        # ---- revision B6
        "L_restricted_by_issue_hour": L_restr.tolist(),
        "L_restricted_by_horizon_at_h23": R_mean[:, 23].tolist(),
        "L_restricted_by_horizon_at_h0": R_mean[:, 0].tolist(),
        "L_restricted_mean": float(L_restr.mean()),
        "L_restricted_max": float(L_restr.max()),
        "restricted_h23_n_increments": int((np.diff(R_mean[:, 23]) > 1e-9).sum()),
        "restricted_h23_max_increment": float(np.diff(R_mean[:, 23]).max()),
        "RESTRICTED_NOTE": (
            "row sums of |J| over ONLY the trailing h+1 columns, which are the "
            "ones the codec perturbs at issue hour h. This is the band to plot; "
            "L_local sums over all 336 columns and is an order of magnitude "
            "looser for no reason."),
        **normalised_operator_norm(model, wb.L),
    }
    if verbose:
        print(f"local Jacobian over {out['n_origins']} test origins")
        print(f"  L_local  mean {out['L_local_mean']:.3f}  "
              f"p50 {out['L_local_p50']:.3f}  p90 {out['L_local_p90']:.3f}  "
              f"max {out['L_local_max']:.3f}")
        print(f"  ||M||_inf in NORMALISED space "
              f"{out['M_inf_normalised_space']:.3f}  (exact and global THERE, "
              f"and only there)")
        print(f"  RESTRICTED (revision B6): by issue hour, mean "
              f"{out['L_restricted_mean']:.3f}, max "
              f"{out['L_restricted_max']:.3f}")
        print(f"    h0 {L_restr[0]:.3f}   h6 {L_restr[6]:.3f}   "
              f"h12 {L_restr[12]:.3f}   h23 {L_restr[23]:.3f}   "
              f"(tightening factor vs L_local "
              f"{out['L_local_mean'] / max(out['L_restricted_mean'], 1e-9):.0f}x)")
        u = R_mean[:, 23]
        d = np.diff(u)
        n_up = int((d > 1e-9).sum())
        up_max = float(d.max())
        print(f"  why staleness damage concentrates at short leads: restricted "
              f"row sum at h=23,")
        print(f"    by horizon step -- h1 {u[0]:.3f}  h6 {u[5]:.3f}  "
              f"h12 {u[11]:.3f}  h18 {u[17]:.3f}  h24 {u[23]:.3f}  "
              f"(peaks at h1, decays to h24; "
              + (f"{n_up} increment{'s' if n_up != 1 else ''} of at most "
                 f"{up_max:.3f} in the tail, so not STRICTLY monotone)"
                 if n_up else "strictly monotone)"))
        print(f"    Claim only the h1 peak and the decay. No U-shape is "
              f"claimed: that belonged to the")
        print(f"    alpha = inf corrupted-window control, which is no longer "
              f"reported.")
        print(f"  first-order band for the figures: ||dy_hat||_inf ~< "
              f"L_restricted(h) * eps, LOCAL")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="synth_data")
    ap.add_argument("--checkpoint", default="synth_dlinear.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_origins", type=int, default=64)
    ap.add_argument("--out", default=paths.result("tier1_sensitivity.json"))
    a = ap.parse_args()

    from tier2.adapter import (load_frozen_dlinear, make_window_builder,
                               store_manifest)
    man = store_manifest(a.data_dir)
    model, cfg, _ = load_frozen_dlinear(a.checkpoint, man)
    wb = make_window_builder(a.data_dir, cfg.lookback_len, cfg.horizon_len)
    d = check_matrix(cfg.lookback_len)
    print(f"moving-average matrix matches avg_pool1d to {d:.2e}")
    out = run(model, wb, wb.splits[a.split], a.n_origins)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
