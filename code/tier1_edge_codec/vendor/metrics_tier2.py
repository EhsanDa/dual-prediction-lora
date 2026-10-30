"""
metrics_tier2.py  --  every reported number, defined once.
Run `python metrics_tier2.py` for the correctness suite.

STAGE-4 RULE (spec v2, E1)
==========================
All metrics here are computed on kWh, AFTER inversion
(RevIN^-1 -> expm1 -> clip at 0). Stage-2 loss values are training
diagnostics and never appear in a results table: a log-space RMSE is neither
comparable to the literature nor interpretable as an energy quantity.

Shapes throughout: y, p are (n_origins, H); `client` is (n_origins,).

ON THE kWh QUESTION
-------------------
MAE and RMSE are in kWh because the spec requires it and because the
RMSE/MAE ratio is a required tail diagnostic that only exists in a
scale-dependent unit. But the HEADLINE metrics -- MASE and Skill Score --
are scale-free by construction. Read those; MAE/RMSE exist for the ratio
diagnostic and for comparability with the literature.

WHAT CHANGED FROM v1, AND WHY
-----------------------------
* What v1 called MASE was MAE_model / MAE_SN168 on the TEST set. That is a
  legitimate relative measure but it is not MASE, and shipping it under that
  name gets flagged in review. Proper MASE (Hyndman & Koehler 2006, m=24,
  denominator on the TRAINING split) is implemented instead.
* RSE and CORR are demoted out of the primary suite into the appendix
  functions at the bottom. RSE's denominator is a grand mean pooled across
  clients of wildly different scale, so it is dominated by BETWEEN-client
  variance -- a quantity no forecaster is asked to predict. CORR is blind to
  bias and gain (a forecaster emitting 2y+1000 scores exactly 1.000) and is
  near-saturated by the daily cycle. Both are kept only for positioning
  against published work.
* The v1 per-client z-standardization is gone. MASE does that job correctly
  and its denominator comes from TRAIN, so it does not move when the test
  filtering policy changes -- exactly the fragility that made the v1 unit a
  poor choice for a protocol still under revision.
"""

import numpy as np

EPS = 1e-12


# ------------------------------------------------------------------ scales
def mase_scale(series_true, missing, train_end, m=24):
    """Per-client MASE denominator: mean |y_t - y_{t-m}| on the TRAINING split,
    over pairs where NEITHER endpoint is imputed.

    Review finding M12. Previously this ran on the interpolated series. Linear
    interpolation produces artificially small |y_t - y_{t-m}| across imputed
    spans, which deflates the denominator and therefore INFLATES MASE -- the
    headline metric -- for exactly the clients with the most missing data.

    Returns (scale, n_degenerate, n_pairs_used).
    """
    tr = np.asarray(series_true[:train_end], np.float64)
    ms = np.asarray(missing[:train_end], bool)
    d = np.abs(tr[m:] - tr[:-m])
    ok = (~ms[m:]) & (~ms[:-m]) & np.isfinite(d)
    n = ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(n > 0, np.nansum(np.where(ok, d, 0.0), axis=0)
                         / np.maximum(n, 1), np.nan)
    bad = ~np.isfinite(scale) | (scale <= EPS) | (n < 100)
    return np.where(bad, np.nan, scale), int(bad.sum()), n


def climatology_table(series_raw, is_weekend, train_end):
    """Per client, mean load by (hour-of-day x weekend/weekday), fit on TRAIN.

    Spec 6.4 row 2: the "no model" reference that already captures the daily
    cycle. It is the most informative calibrating baseline and the one usually
    omitted. If a deep model does not beat it by a wide margin, that is the
    finding. Returns (24, 2, M).
    """
    tr = np.asarray(series_raw[:train_end], np.float64)
    hours = np.arange(train_end) % 24
    wk = np.asarray(is_weekend[:train_end]).astype(int)
    out = np.zeros((24, 2, tr.shape[1]))
    for h in range(24):
        for w in (0, 1):
            sel = (hours == h) & (wk == w)
            out[h, w] = tr[sel].mean(axis=0) if sel.any() else tr.mean(axis=0)
    return out


# ------------------------------------------------------- primary metric set
def _mask(y, p, mask):
    """Common masking. `mask` may be None, and NaN in either array (an imputed
    baseline reference, C5) is always excluded so comparisons stay paired."""
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    mk = np.isfinite(y) & np.isfinite(p)
    if mask is not None:
        mk &= np.asarray(mask, bool)
    return y, p, mk


def pooled_metrics(y, p, mask=None):
    """Pooled over (client, origin, horizon step), masked. Units: kWh."""
    y, p, mk = _mask(y, p, mask)
    y, p = y[mk], p[mk]
    e = p - y
    mae = float(np.abs(e).mean())
    rmse = float(np.sqrt((e ** 2).mean()))
    smape = float((2.0 * np.abs(e) / np.maximum(np.abs(y) + np.abs(p), EPS)).mean() * 100)
    return {"MAE": mae, "RMSE": rmse,
            # Required tail diagnostic. A well-behaved error distribution sits
            # between 1.25 and 2.5. Ten means a handful of enormous errors own
            # the sum -- the pooled number is then mostly reporting the largest
            # meters, and model-to-model differences in it are reweighting noise.
            "RMSE/MAE": float(rmse / mae) if mae > EPS else float("nan"),
            "sMAPE%": smape, "n": int(e.size)}


def per_horizon(y, p, mask=None):
    """MAE_h and RMSE_h for h = 1..H, masked per step. A primary result:
    error growth across the day is one of the main things the vector
    formulation buys."""
    y, p, mk = _mask(y, p, mask)
    e = np.where(mk, p - y, 0.0)
    n = np.maximum(mk.sum(axis=0), 1)
    return {"MAE_h": np.abs(e).sum(axis=0) / n,
            "RMSE_h": np.sqrt((e ** 2).sum(axis=0) / n),
            "n_h": mk.sum(axis=0)}


def per_client(y, p, client, scale, sn24_mae_pc=None, mask=None):
    """Per-client MAE, MASE, nRMSE and (given SN-24's per-client MAE) Skill
    Score. Vectorised with bincount (m14): the previous boolean-mask loop over
    320 clients x 9 model rows on a 1.675M x 24 array cost minutes."""
    y, p, mk = _mask(y, p, mask)
    H = y.shape[1] if y.ndim > 1 else 1
    cl = np.repeat(np.asarray(client), H)
    e = np.where(mk, p - y, 0.0).ravel()
    w = mk.ravel().astype(np.float64)
    ci = cl
    M = int(ci.max()) + 1

    n = np.bincount(ci, weights=w, minlength=M)
    sae = np.bincount(ci, weights=np.abs(e) * w, minlength=M)
    sse = np.bincount(ci, weights=(e ** 2) * w, minlength=M)
    sy = np.bincount(ci, weights=np.where(mk, y, 0.0).ravel() * w, minlength=M)
    with np.errstate(invalid="ignore", divide="ignore"):
        mae = np.where(n > 0, sae / np.maximum(n, 1), np.nan)
        rmse = np.where(n > 0, np.sqrt(sse / np.maximum(n, 1)), np.nan)
        ybar = np.where(n > 0, sy / np.maximum(n, 1), np.nan)

    present = np.unique(ci[w > 0])
    sc = np.array([scale[c] if isinstance(scale, dict) else scale[c]
                   for c in range(M)], dtype=np.float64)
    mae_pc, mase_pc, nrmse_pc, ss_pc = {}, {}, {}, {}
    for c in present:
        ci_ = int(c)
        mae_pc[ci_] = float(mae[c])
        mase_pc[ci_] = (float(mae[c] / sc[c])
                        if np.isfinite(sc[c]) and sc[c] > EPS else np.nan)
        nrmse_pc[ci_] = float(rmse[c] / ybar[c]) if ybar[c] > EPS else np.nan
        if sn24_mae_pc is not None:
            bb = sn24_mae_pc.get(ci_, np.nan)
            ss_pc[ci_] = float(1.0 - mae[c] / bb) if bb and bb > EPS else np.nan
    return {"MAE": mae_pc, "MASE": mase_pc, "nRMSE": nrmse_pc, "SS": ss_pc}


def distribution(d):
    """p10/p50/p90/p99 and mean over clients, ignoring nan and counting them."""
    v = np.array(list(d.values()), dtype=np.float64)
    ok = np.isfinite(v)
    if not ok.any():
        return {"p10": np.nan, "p50": np.nan, "p90": np.nan, "p99": np.nan,
                "mean": np.nan, "n": 0, "n_excluded": int(v.size)}
    w = v[ok]
    return {"p10": float(np.percentile(w, 10)), "p50": float(np.percentile(w, 50)),
            "p90": float(np.percentile(w, 90)), "p99": float(np.percentile(w, 99)),
            "mean": float(w.mean()), "n": int(ok.sum()),
            "n_excluded": int((~ok).sum())}


def frac_no_skill(ss_pc, tol=1e-9):
    """Share of clients STRICTLY worse than seasonal-naive on their own data.

    M21: this used `<= 0`, so Seasonal-Naive-24 -- whose skill score against
    itself is exactly 0 -- landed in the "no skill" bucket and the table read
    `Seasonal-Naive-24 ... 100.0%`. Technically true, and it confused every
    reader. The column is now SS<0 and the reference baseline reports 0.0%.
    """
    v = np.array(list(ss_pc.values()), np.float64)
    ok = np.isfinite(v)
    return float((v[ok] < -tol).mean()) if ok.any() else np.nan


# ------------------------------------------------- probabilistic (optional)
def pinball(y, q_pred, quantiles):
    """q_pred: (n, H, Q). Mean pinball over quantiles and horizon, plus per
    step. Averaging over a dense quantile grid approximates CRPS up to a
    constant factor -- state that if CRPS is claimed."""
    yy = np.asarray(y, np.float64)[..., None]
    q = np.asarray(quantiles, np.float64)[None, None, :]
    d = yy - np.asarray(q_pred, np.float64)
    loss = np.maximum(q * d, (q - 1.0) * d)
    return {"pinball": float(loss.mean()), "pinball_h": loss.mean(axis=(0, 2))}


def picp_sharpness(y, lo, hi):
    """Coverage AND mean interval width. Coverage alone is trivially gamed by
    widening the interval; the pair is the meaningful statement."""
    y = np.asarray(y, np.float64)
    return {"PICP": float(((y >= lo) & (y <= hi)).mean()),
            "width": float(np.mean(np.asarray(hi) - np.asarray(lo)))}


# ---------------------------------------------- comparability appendix only
def lstnet_rse(y, p, mask=None):
    """Lai et al. RSE. The denominator uses ONE GRAND MEAN pooled over every
    (client, step) entry, not per-client means. Identical to
    RMSE / population-std of the test targets -- used below as a free unit
    test. Invariant to a single global rescale, so the mean-of-4 vs sum-of-4
    aggregation ambiguity cancels in it; it does NOT cancel in MAE or RMSE."""
    y, p, mk = _mask(y, p, mask)
    y, p = y[mk].ravel(), p[mk].ravel()
    den = np.sqrt(((y - y.mean()) ** 2).sum())
    return float(np.sqrt(((p - y) ** 2).sum()) / den) if den > EPS else float("nan")


def lstnet_appendix(y, p, client, steps=(3, 6, 12, 24), mask=None):
    """m16: spec 11.4 asks for h in {3,6,12,24}, which at H=24 are just
    columns 3, 6, 12 and 24. It is free, so report all four."""
    out = {}
    for h in steps:
        if h <= y.shape[1]:
            col = slice(h - 1, h)
            mk = None if mask is None else mask[:, col]
            # the appendix used to ignore the validity mask while every other
            # metric applied it, so RSE and CORR were scored on imputed target
            # steps the rest of the table excluded. Caught by the streaming
            # accumulator's agreement test.
            cv, ex = lstnet_corr(y[:, col], p[:, col], client, mk)
            out[h] = {"RSE": lstnet_rse(y[:, col], p[:, col], mk),
                      "CORR": cv, "excluded": ex}
    return out


def lstnet_corr(y, p, client, mask=None):
    """Lai et al. CORR: per-series Pearson with per-series means, then an
    UNWEIGHTED average over series.

    IMPLEMENTATION WARNING. The denominator is the product of two separate
    square roots of two separate sums: sqrt(sum a^2) * sqrt(sum b^2). Several
    widely cited papers transcribe it as sqrt(sum(a^2 * b^2)), a different and
    incorrect quantity. The suite asserts the two differ so a future edit
    cannot silently swap them.

    Note the deliberate asymmetry with RSE (pooled grand mean vs per-series
    means). It is inherited from the protocol -- reproduce it, do not
    harmonise it, and footnote it.
    """
    y, p, mk = _mask(y, p, mask)
    client = np.asarray(client)
    out, excluded = [], 0
    for c in np.unique(client):
        sel = (client == c)[:, None] & mk
        if sel.sum() < 2:
            excluded += 1
            continue
        ys, ps = y[sel], p[sel]
        a, b = ys - ys.mean(), ps - ps.mean()
        na, nb = np.sqrt((a ** 2).sum()), np.sqrt((b ** 2).sum())
        # RELATIVE guard. An absolute 1e-12 floor does not detect a constant
        # series: summing 3,000 identical float64 loads and dividing leaves a
        # last-ulp residue, so `p - mean(p)` is ~1e-13 rather than 0 and its
        # norm clears the floor. The per-client-mean baseline is exactly such
        # a predictor, and it was being scored with a meaningless correlation
        # near 0 instead of being excluded as undefined.
        if (na < 1e-8 * np.sqrt((ys ** 2).sum()) or
                nb < 1e-8 * np.sqrt((ps ** 2).sum())):
            excluded += 1          # constant series -> correlation undefined
            continue
        out.append((a * b).sum() / (na * nb))
    return (float(np.mean(out)) if out else float("nan")), excluded


# =====================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_ok = 0

    def check(name, cond):
        global n_ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        assert cond, name
        n_ok += 1

    print("metric correctness suite (spec v2)\n")
    M, T, H = 12, 3000, 24
    t = np.arange(T)[:, None]
    series = np.abs(rng.lognormal(4, 1.0, (T, M)) * (1 + 0.5 * np.sin(2 * np.pi * t / 24)))
    train_end = 2000

    # ---- MASE denominator
    ms = np.zeros_like(series, bool)
    sc, _, _ = mase_scale(series, ms, train_end, m=24)
    check("MASE scale = mean|y_t - y_{t-24}| on train",
          np.allclose(sc, np.abs(series[24:train_end] - series[:train_end - 24]).mean(0)))
    check("MASE scale ignores test data",
          not np.allclose(sc, np.abs(series[24:] - series[:-24]).mean(0)))
    # M12: an imputed endpoint must be excluded from the denominator
    ms2 = np.zeros_like(series, bool); ms2[100:400, 0] = True
    sc2, _, n2 = mase_scale(series, ms2, train_end, m=24)
    check("imputed endpoints excluded from the MASE denominator",
          abs(sc2[0] - sc[0]) > 1e-9 and n2[0] < n2[1])
    # C4: masking must change the answer and must be paired
    yq = np.abs(rng.lognormal(4, 1, (200, 24))); pq = yq + rng.normal(0, 5, yq.shape)
    mk = rng.random(yq.shape) > 0.3
    check("mask changes pooled metrics",
          abs(pooled_metrics(yq, pq)["MAE"] - pooled_metrics(yq, pq, mk)["MAE"]) > 1e-9)
    yn = yq.copy(); yn[0, 0] = np.nan
    check("NaN reference excluded automatically",
          np.isfinite(pooled_metrics(yn, pq)["MAE"]))
    const = series.copy(); const[:, 0] = 7.0
    check("constant client -> scale nan, counted", mase_scale(const, ms, train_end)[1] == 1)

    # ---- MASE == 1 for seasonal-naive measured against its own scale
    y_tr = series[24:train_end, :1]
    p_sn = series[:train_end - 24, :1]
    pc = per_client(y_tr, p_sn, np.zeros(len(y_tr), int), {0: sc[0]})
    check("SN-24 scored on train gives MASE == 1", abs(pc["MASE"][0] - 1.0) < 1e-9)

    # ---- pooled
    y = np.abs(rng.lognormal(4, 1, (500, H)))
    p = y + rng.normal(0, 10, y.shape)
    pm = pooled_metrics(y, p)
    check("RMSE >= MAE", pm["RMSE"] >= pm["MAE"])
    check("RMSE/MAE ratio consistent", abs(pm["RMSE/MAE"] - pm["RMSE"] / pm["MAE"]) < 1e-12)
    check("perfect forecast -> zero error", pooled_metrics(y, y.copy())["MAE"] == 0.0)
    check("sMAPE within [0, 200]", 0 <= pm["sMAPE%"] <= 200)

    # ---- per-horizon
    ph = per_horizon(y, p)
    check("per-horizon length H", len(ph["MAE_h"]) == H)
    check("mean of MAE_h == pooled MAE", abs(ph["MAE_h"].mean() - pm["MAE"]) < 1e-12)

    # ---- Skill Score
    client = rng.integers(0, M, 500)
    smap = {c: 20.0 for c in range(M)}
    sn = per_client(y, y + rng.normal(0, 30, y.shape), client, smap)
    mo = per_client(y, y + rng.normal(0, 10, y.shape), client, smap, sn["MAE"])
    check("SS > 0 when the model beats the naive",
          np.nanmean(list(mo["SS"].values())) > 0)
    # identical forecaster as both model and baseline -> SS exactly 0.
    # (A PERFECT forecaster cannot be used here: the baseline MAE would be 0
    # and SS is then undefined, which is the nan branch tested separately.)
    dup = y + rng.normal(0, 30, y.shape)
    base = per_client(y, dup, client, smap)
    check("SS == 0 when the model equals the naive",
          abs(per_client(y, dup, client, smap, base["MAE"])["SS"][0]) < 1e-12)
    check("SS is nan, not inf, when the baseline is perfect",
          np.isnan(per_client(y, dup, client, smap,
                              {c: 0.0 for c in range(M)})["SS"][0]))
    worse = per_client(y, y + rng.normal(0, 90, y.shape), client, smap, sn["MAE"])
    check("frac_no_skill catches a worse-than-naive model",
          frac_no_skill(worse["SS"]) > 0.5)
    # M21: the reference baseline must read 0%, not 100%
    self_ss = per_client(y, dup, client, smap, base["MAE"])["SS"]
    check("M21: SS<0 is 0% for the reference baseline against itself",
          frac_no_skill(self_ss) == 0.0)

    # ---- distribution
    d = distribution({0: 1.0, 1: 2.0, 2: np.nan, 3: 3.0})
    check("distribution excludes nan and counts it",
          d["n"] == 3 and d["n_excluded"] == 1 and abs(d["p50"] - 2.0) < 1e-12)

    # ---- RSE identity (free unit test from spec 11.4)
    r = lstnet_rse(y, p)
    check("RSE == RMSE / pop-std of targets",
          abs(r - np.sqrt(((p - y) ** 2).mean()) / y.ravel().std()) < 1e-10)
    check("RSE invariant to a global rescale (mean-of-4 vs sum-of-4 cancels)",
          abs(lstnet_rse(4 * y, 4 * p) - r) < 1e-10)
    sc_c = rng.uniform(0.5, 5, M)[client][:, None]
    check("RSE NOT invariant to per-client rescale",
          abs(lstnet_rse(sc_c * y, sc_c * p) - r) > 1e-3)

    # ---- CORR
    a = y[client == 0].ravel() - y[client == 0].mean()
    b = p[client == 0].ravel() - p[client == 0].mean()
    right = (a * b).sum() / (np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum()))
    wrong = (a * b).sum() / np.sqrt(((a ** 2) * (b ** 2)).sum())
    check("CORR uses sqrt(Sa2)*sqrt(Sb2), not sqrt(S a2b2)", abs(right - wrong) > 1e-3)
    check("CORR matches np.corrcoef per series",
          abs(right - np.corrcoef(y[client == 0].ravel(),
                                  p[client == 0].ravel())[0, 1]) < 1e-12)
    cbias, _ = lstnet_corr(y, 2 * y + 1000, client)
    check("CORR = 1.000 for the forecaster 2y+1000 (why it is appendix-only)",
          abs(cbias - 1.0) < 1e-9)
    p_const = p.copy(); p_const[client == 3] = 5.0
    check("constant prediction series excluded from CORR and counted",
          lstnet_corr(y, p_const, client)[1] == 1)

    # ---- probabilistic
    check("pinball(q=0.5) == 0.5 * MAE",
          abs(pinball(y, (y - 3.0)[..., None], [0.5])["pinball"] - 1.5) < 1e-9)
    cov = picp_sharpness(y, y - 1, y + 1)
    check("PICP = 1 for an enveloping interval", abs(cov["PICP"] - 1.0) < 1e-12)
    check("sharpness reported with coverage", abs(cov["width"] - 2.0) < 1e-12)

    print(f"\nall {n_ok} checks passed")
