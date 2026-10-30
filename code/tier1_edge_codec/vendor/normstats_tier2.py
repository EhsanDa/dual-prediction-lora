"""
normstats_tier2.py  --  ONE definition of the RevIN look-back statistics.

Review findings M4/M5/M6. `windows_tier2` claimed in its docstring that both
sides called `RevIN.statistics()`. They did not: the model used
`torch.Tensor.std(dim=-1)`, which defaults to the UNBIASED estimator
(correction=1), while `WindowBuilder` reimplemented it inline as
`numpy.ndarray.std(axis=1)`, which defaults to ddof=0. The two differ by
sqrt(L/(L-1)) -- 0.15% at L=336, 0.52% at L=96 -- so the covariate anchors and
the target lived in slightly different spaces, in exactly the place the code
claimed to be careful.

This module is the fix. Both back ends are defined here, side by side, and the
self-test asserts they agree to 1e-7 on random input. `eps` and `sigma_floor`
are arguments with no defaults anywhere they matter, so a config change cannot
silently desynchronise the training target space from the inversion space.

CONVENTION: ddof = 0 (population). The window is the whole population of
interest, not a sample from a larger one, and ddof=0 is what the numpy side
was already doing on the data path.
"""

import numpy as np


def norm_stats_np(y, sigma_floor):
    """y: (..., L) float array. Returns (mu, sigma) keeping the last axis."""
    mu = y.mean(axis=-1, keepdims=True)
    sigma = np.maximum(y.std(axis=-1, keepdims=True), sigma_floor)
    return mu, sigma


def norm_stats_torch(y, sigma_floor):
    """Same statistic, torch back end. `correction=0` is not the torch default
    and omitting it is the M4 defect."""
    mu = y.mean(dim=-1, keepdim=True)
    sigma = y.std(dim=-1, unbiased=False, keepdim=True).clamp_min(sigma_floor)
    return mu, sigma


def z_of(y, mu, sigma, eps):
    """The one normalisation formula. Everything that normalises calls this."""
    return (y - mu) / (sigma + eps)


if __name__ == "__main__":
    import torch
    rng = np.random.default_rng(0)
    ok = 0
    for L in (96, 168, 336, 720):
        a = rng.normal(3.0, 2.0, (64, L))
        mn, sn = norm_stats_np(a, 0.05)
        mt, st = norm_stats_torch(torch.from_numpy(a), 0.05)
        dm = float(np.abs(mn - mt.numpy()).max())
        ds = float(np.abs(sn - st.numpy()).max())
        assert dm < 1e-7 and ds < 1e-7, (L, dm, ds)
        # and demonstrate what the old torch default would have cost
        biased = torch.from_numpy(a).std(dim=-1, keepdim=True).numpy()
        drift = float(np.abs(biased / sn - 1).max())
        print(f"  L={L:4d}  numpy vs torch agree to {max(dm, ds):.2e}   "
              f"(unbiased default would drift {drift:.3%})")
        ok += 1
    # sigma floor must bind
    flat = np.full((4, 336), 7.0)
    _, s = norm_stats_np(flat, 0.05)
    assert np.allclose(s, 0.05)
    print(f"  sigma floor binds on a flat window: OK")
    print(f"all {ok + 1} checks passed")
