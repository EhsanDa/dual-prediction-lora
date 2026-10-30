"""
evalaccum_tier2.py  --  streaming metric accumulators.

WHY THIS EXISTS
===============
The evaluator used to materialise every model's predictions over the whole
test split before scoring: 12 rows x 1,675,489 windows x 24 steps x 4 B =
1.93 GB, plus float64 temporaries inside the metric functions, on top of a
10-20 GB ridge design matrix. On a machine with a few GB of RAM that is an
out-of-memory kill, and it presents as the process dying with no message.

Every metric this project reports is a function of sums, so none of it needs
the full arrays. This class accumulates the sufficient statistics in one pass
and reproduces the batch metrics EXACTLY -- `metrics_tier2` is still the
definition of record, and the self-test at the bottom asserts agreement to
1e-9 against it.

Memory is O(n_clients + H) per row, independent of the number of windows.
"""

import numpy as np

EPS = 1e-12


class Accum:
    """One row of the results table."""

    def __init__(self, n_clients, H, corr_steps=(3, 6, 12, 24)):
        self.M, self.H = n_clients, H
        self.corr_steps = tuple(h for h in corr_steps if h <= H)
        z = lambda *s: np.zeros(s, np.float64)
        self.n = 0.0
        self.sae = self.sse = self.smape = 0.0
        self.h_ae, self.h_se, self.h_n = z(H), z(H), z(H)
        self.c_ae, self.c_se, self.c_y, self.c_n = z(n_clients), z(n_clients), \
            z(n_clients), z(n_clients)
        # per selected horizon: pooled sums for RSE, per-client sums for CORR
        k = len(self.corr_steps)
        self.k_n, self.k_se = z(k), z(k)
        self.k_y, self.k_y2 = z(k), z(k)
        self.k_cn, self.k_cy, self.k_cp = z(k, n_clients), z(k, n_clients), z(k, n_clients)
        self.k_cy2, self.k_cp2, self.k_cyp = z(k, n_clients), z(k, n_clients), z(k, n_clients)

    def add(self, y, p, valid, client):
        y = np.asarray(y, np.float64)
        p = np.asarray(p, np.float64)
        mk = np.asarray(valid, bool) & np.isfinite(y) & np.isfinite(p)
        e = np.where(mk, p - y, 0.0)
        w = mk.astype(np.float64)
        ae, se = np.abs(e), e ** 2

        self.n += w.sum()
        self.sae += ae.sum()
        self.sse += se.sum()
        # The mask must be applied INSIDE, not as a multiplier. A NaN
        # prediction (an imputed seasonal-naive reference) gives NaN/NaN, and
        # NaN * 0 is still NaN, which poisons the whole sum. Every other
        # metric here is built from `e`, which np.where has already zeroed, so
        # only sMAPE was affected -- it printed NaN for Seasonal-Naive-24 and
        # Persistence in the first full-split run.
        self.smape += np.where(
            mk, 2.0 * ae / np.maximum(np.abs(y) + np.abs(p), EPS), 0.0).sum()

        self.h_ae += ae.sum(0); self.h_se += se.sum(0); self.h_n += w.sum(0)

        ci = np.asarray(client)
        M = self.M
        self.c_ae += np.bincount(ci, weights=ae.sum(1), minlength=M)
        self.c_se += np.bincount(ci, weights=se.sum(1), minlength=M)
        self.c_y += np.bincount(ci, weights=(y * w).sum(1), minlength=M)
        self.c_n += np.bincount(ci, weights=w.sum(1), minlength=M)

        for j, h in enumerate(self.corr_steps):
            col = h - 1
            yy, pp, ww = y[:, col], p[:, col], w[:, col]
            self.k_n[j] += ww.sum()
            self.k_se[j] += (se[:, col]).sum()
            self.k_y[j] += (yy * ww).sum()
            self.k_y2[j] += (yy * yy * ww).sum()
            self.k_cn[j] += np.bincount(ci, weights=ww, minlength=M)
            self.k_cy[j] += np.bincount(ci, weights=yy * ww, minlength=M)
            self.k_cp[j] += np.bincount(ci, weights=pp * ww, minlength=M)
            self.k_cy2[j] += np.bincount(ci, weights=yy * yy * ww, minlength=M)
            self.k_cp2[j] += np.bincount(ci, weights=pp * pp * ww, minlength=M)
            self.k_cyp[j] += np.bincount(ci, weights=yy * pp * ww, minlength=M)

    # ------------------------------------------------------------- readout
    def pooled(self):
        mae = self.sae / self.n
        rmse = np.sqrt(self.sse / self.n)
        return {"MAE": float(mae), "RMSE": float(rmse),
                "RMSE/MAE": float(rmse / mae) if mae > EPS else float("nan"),
                "sMAPE%": float(self.smape / self.n * 100), "n": int(self.n)}

    def per_horizon(self):
        n = np.maximum(self.h_n, 1)
        return {"MAE_h": self.h_ae / n, "RMSE_h": np.sqrt(self.h_se / n),
                "n_h": self.h_n}

    def per_client(self, mase_scale, sn24_mae_pc=None):
        out_mae, out_mase, out_nrmse, out_ss = {}, {}, {}, {}
        for c in np.nonzero(self.c_n > 0)[0]:
            ci = int(c)
            n = self.c_n[c]
            mae = self.c_ae[c] / n
            out_mae[ci] = float(mae)
            sc = mase_scale[c]
            out_mase[ci] = float(mae / sc) if np.isfinite(sc) and sc > EPS else np.nan
            ybar = self.c_y[c] / n
            out_nrmse[ci] = (float(np.sqrt(self.c_se[c] / n) / ybar)
                             if ybar > EPS else np.nan)
            if sn24_mae_pc is not None:
                b = sn24_mae_pc.get(ci, np.nan)
                out_ss[ci] = float(1.0 - mae / b) if b and b > EPS else np.nan
        return {"MAE": out_mae, "MASE": out_mase, "nRMSE": out_nrmse, "SS": out_ss}

    def lstnet(self):
        """RSE with the pooled grand mean; CORR as the unweighted mean of
        per-series Pearson. Same definitions as metrics_tier2."""
        out = {}
        for j, h in enumerate(self.corr_steps):
            n = self.k_n[j]
            ss_tot = self.k_y2[j] - (self.k_y[j] ** 2) / max(n, 1)
            rse = float(np.sqrt(self.k_se[j] / ss_tot)) if ss_tot > EPS else np.nan
            cn = self.k_cn[j]
            ok = cn >= 2
            cov = self.k_cyp[j][ok] - self.k_cy[j][ok] * self.k_cp[j][ok] / cn[ok]
            vy = self.k_cy2[j][ok] - self.k_cy[j][ok] ** 2 / cn[ok]
            vp = self.k_cp2[j][ok] - self.k_cp[j][ok] ** 2 / cn[ok]
            # RELATIVE guard. The streaming form E[x^2] - E[x]^2 suffers
            # catastrophic cancellation for a constant predictor: with loads
            # around 1e3 and ~1e2 terms, sum(p^2) is ~1e8 and the difference
            # leaves float residue near 1e-8 -- comfortably above an absolute
            # 1e-12 floor, so a constant series passed the "has variance"
            # test and produced a meaningless correlation near 0 instead of
            # being excluded. The "Per-client mean" row is exactly such a
            # predictor, so this was visible in the first table produced.
            good = (vy > 1e-10 * np.maximum(self.k_cy2[j][ok], 1.0)) & \
                   (vp > 1e-10 * np.maximum(self.k_cp2[j][ok], 1.0))
            r = cov[good] / np.sqrt(vy[good] * vp[good])
            out[h] = {"RSE": rse,
                      "CORR": float(r.mean()) if r.size else np.nan,
                      "excluded": int((~good).sum())}
        return out


if __name__ == "__main__":
    from metrics_tier2 import (lstnet_appendix, per_client, per_horizon,
                               pooled_metrics)
    rng = np.random.default_rng(0)
    M, N, H = 40, 5000, 24
    client = rng.integers(0, M, N)
    y = np.abs(rng.lognormal(4, 1.2, (N, H)))
    p = y + rng.normal(0, 40, y.shape)
    valid = rng.random(y.shape) > 0.05
    scale = rng.uniform(5, 50, M)

    a = Accum(M, H)
    for i in range(0, N, 137):                       # deliberately ragged
        s = slice(i, i + 137)
        a.add(y[s], p[s], valid[s], client[s])

    ok = 0
    def chk(name, x, z, tol=1e-9):
        global ok
        # absolute when the reference is 0 (the "max deviation" checks),
        # relative otherwise
        d = abs(x - z) if abs(z) < 1e-12 else abs(x - z) / abs(z)
        print(f"  {'PASS' if d < tol else 'FAIL'}  {name}  ({d:.2e})")
        assert d < tol, name
        ok += 1

    ref = pooled_metrics(y, p, valid)
    got = a.pooled()
    for k in ("MAE", "RMSE", "RMSE/MAE", "sMAPE%"):
        chk(f"pooled {k}", got[k], ref[k])
    chk("pooled n", got["n"], ref["n"])

    rh, gh = per_horizon(y, p, valid), a.per_horizon()
    chk("MAE_h max dev", float(np.abs(rh["MAE_h"] - gh["MAE_h"]).max()), 0.0, 1e-9)
    chk("RMSE_h max dev", float(np.abs(rh["RMSE_h"] - gh["RMSE_h"]).max()), 0.0, 1e-9)

    rp = per_client(y, p, client, scale, mask=valid)
    gp = a.per_client(scale)
    for key in ("MAE", "MASE", "nRMSE"):
        dev = max(abs(rp[key][c] - gp[key][c]) / max(abs(rp[key][c]), 1e-12)
                  for c in rp[key])
        chk(f"per-client {key} max rel dev", dev, 0.0, 1e-9)

    rl = lstnet_appendix(y, p, client, mask=valid)
    gl = a.lstnet()
    for h in rl:
        chk(f"RSE h={h}", gl[h]["RSE"], rl[h]["RSE"])
        chk(f"CORR h={h}", gl[h]["CORR"], rl[h]["CORR"])

    # a CONSTANT predictor (the per-client-mean baseline) must be excluded
    # from CORR, not scored as ~0. Streaming variance is the fragile case.
    const_p = np.repeat(rng.uniform(500, 5000, M)[client][:, None], H, 1)
    ac = Accum(M, H)
    for i in range(0, N, 137):
        sl = slice(i, i + 137)
        ac.add(y[sl], const_p[sl], valid[sl], client[sl])
    gc = ac.lstnet()
    rc = lstnet_appendix(y, const_p, client, mask=valid)
    for h in gc:
        both_nan = np.isnan(gc[h]["CORR"]) and np.isnan(rc[h]["CORR"])
        print(f"  {'PASS' if both_nan else 'FAIL'}  constant predictor -> "
              f"CORR nan at h={h}  (streaming {gc[h]['CORR']}, "
              f"reference {rc[h]['CORR']}, excluded {gc[h]['excluded']}/{M})")
        assert both_nan
        ok += 1

    print(f"\nall {ok} checks passed -- streaming reproduces metrics_tier2 exactly")
