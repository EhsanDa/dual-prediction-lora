"""
revision_stats.py  --  paired client-level uncertainty for the revision.
Conference revision, request B (2026-09-29).

Both reviewers asked for "paired client-level uncertainty, confidence
intervals, and per-client win/loss distributions for next-hour error and MASE
increments". Reads the per-client vectors eval/run_eval.py writes with
--per_client_out (a separate file; tier1_results.json is unchanged) and
reports, over the scored clients, by percentile bootstrap (clients resampled
with replacement, PAIRED -- every client contributes to both sides):

  1. NEXT-HOUR ERROR. Per client, the mean absolute lead-1 error over its
     test windows, in kWh and divided by the client's mase_scale (scale-free).
     Batch-daily vs the scheme, and the scheme vs the unconstrained control.
     NOTE the paper's 185.5 / 121.6 kWh are POOLED means (every window weighs
     the same, averaged over issue hours), dominated by the largest meters;
     the per-client means here weigh every meter the same. Both are reported.
  2. MASE INCREMENTS. Per-client dMASE against the control.
  3. WIN / LOSS. Share of clients on each side, and the quantiles of the
     per-client paired differences.
  4. THE ABLATIONS AT MATCHED PACKET RATE (Table IV, VI). Each resample
     recomputes every row's mean packets/day and mean dMASE and re-interpolates
     the family curves exactly as eval/run_eval.py does; the point estimate is
     computed on the paper's axis and must reproduce the printed tables.

    python -m eval.revision_stats --per_client ../../results/revision/per_client_v2.json \\
        --results ../../results/revision/tier1_results.json \\
        --out ../../results/revision/stats.json
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

LO, HI = 2.5, 97.5
QUANTS = (5, 10, 25, 50, 75, 90, 95)


def vec(d, clients):
    return np.array([d.get(str(c), np.nan) for c in clients], np.float64)


def paired(a, b, rng, n_boot, name_a, name_b):
    """a - b per client (positive = a larger); bootstrap on the mean and the
    ratio of means; win/loss shares and quantiles of the differences."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    md = d[idx].mean(1)
    ma, mb = a[idx].mean(1), b[idx].mean(1)
    return {
        "n_clients": int(len(d)),
        f"mean_{name_a}": float(a.mean()), f"mean_{name_b}": float(b.mean()),
        f"ci_{name_a}": [float(np.percentile(ma, LO)), float(np.percentile(ma, HI))],
        f"ci_{name_b}": [float(np.percentile(mb, LO)), float(np.percentile(mb, HI))],
        "mean_difference": float(d.mean()),
        "ci_difference": [float(np.percentile(md, LO)), float(np.percentile(md, HI))],
        "ratio_of_means": float(a.mean() / b.mean()),
        "ci_ratio": [float(np.percentile(ma / mb, LO)),
                     float(np.percentile(ma / mb, HI))],
        "share_draws_difference_positive": float((md > 0).mean()),
        "clients_a_larger": int((d > 0).sum()),
        "clients_b_larger": int((d < 0).sum()),
        "clients_equal": int((d == 0).sum()),
        "difference_quantiles": {str(q): float(np.percentile(d, q)) for q in QUANTS},
        "relative_difference_quantiles": {
            str(q): float(np.percentile(d / np.where(b != 0, b, np.nan), q))
            for q in QUANTS},
    }


def one_sided(x, rng, n_boot):
    x = x[np.isfinite(x)]
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(1)
    return {"n_clients": int(len(x)), "mean": float(x.mean()),
            "ci": [float(np.percentile(m, LO)), float(np.percentile(m, HI))],
            "clients_positive": int((x > 0).sum()),
            "clients_negative": int((x < 0).sum()),
            "clients_zero": int((x == 0).sum()),
            "quantiles": {str(q): float(np.percentile(x, q)) for q in QUANTS}}


def interp(xs, ys, rate):
    xs, ys = np.asarray(xs), np.asarray(ys)
    o = np.argsort(xs)
    if rate < xs[o][0] or rate > xs[o][-1]:
        return None
    return float(np.interp(rate, xs[o], ys[o]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per_client", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--frozen", default=os.path.join(paths.FROZEN_DIR,
                                                     "tier1_results.json"))
    ap.add_argument("--scheme", default="C@0.7")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=paths.result("revision_stats.json"))
    a = ap.parse_args(argv)

    pc = json.load(open(a.per_client))
    res = json.load(open(a.results))
    frozen = json.load(open(a.frozen))
    rng = np.random.default_rng(a.seed)
    S, B, CTRL = a.scheme, "batch-daily", "control"

    mase_ctrl = pc["metrics_per_client"][CTRL]["MASE"]
    scored = sorted(int(c) for c, v in mase_ctrl.items() if np.isfinite(v))
    M = len(pc["mase_scale_per_client"])
    scale = vec(pc["mase_scale_per_client"], scored)

    def lead1(row):
        s = vec(pc["lead1_abs_err_sum_kwh_per_client"][row], scored)
        n = vec(pc["lead1_n_per_client"][row], scored)
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)

    out = {"scheme": S, "n_clients_scored": len(scored), "bootstrap": a.bootstrap,
           "seed": a.seed, "ci": [LO, HI]}

    # ---- 1. next-hour error ------------------------------------------------
    l1 = {r: lead1(r) for r in (B, CTRL, S)}
    out["next_hour_error"] = {
        "pooled_reported_kwh": {
            r: float(np.mean(res["metrics"][r]["MAE_lead1_by_issue_hour"]))
            for r in (B, CTRL, S)},
        "per_client_kwh": {
            "batch_minus_scheme": paired(l1[B], l1[S], rng, a.bootstrap, "batch", "scheme"),
            "scheme_minus_control": paired(l1[S], l1[CTRL], rng, a.bootstrap, "scheme", "control"),
        },
        "per_client_scaled": {
            "batch_minus_scheme": paired(l1[B] / scale, l1[S] / scale, rng,
                                         a.bootstrap, "batch", "scheme"),
            "scheme_minus_control": paired(l1[S] / scale, l1[CTRL] / scale, rng,
                                           a.bootstrap, "scheme", "control"),
        },
    }

    # ---- 2, 3. MASE increments ---------------------------------------------
    dmc = pc["delta_MASE_vs_control_per_client"]
    out["mase_increment"] = {
        "batch_minus_scheme": paired(vec(dmc[B], scored), vec(dmc[S], scored),
                                     rng, a.bootstrap, "batch", "scheme"),
        "scheme_vs_control": one_sided(vec(dmc[S], scored), rng, a.bootstrap),
        "batch_vs_control": one_sided(vec(dmc[B], scored), rng, a.bootstrap),
    }

    # ---- 4. ablations at matched packet rate -------------------------------
    pkt = pc["packets_per_day_per_client"]
    all_idx = [str(c) for c in range(M)]
    fam_rows = {}
    for lab, c in res["cost"].items():
        fam = c.get("family")
        if fam in ("A", "B", "C", "D"):
            fam_rows.setdefault(fam, []).append(lab)
        elif fam == "T":
            fam_rows.setdefault(c["tap_id"], []).append(lab)
    fam_rows["T8"] = list(fam_rows["C"])
    P = {lab: np.array([pkt[lab][k] for k in all_idx]) for fam in fam_rows
         for lab in fam_rows[fam]}
    D = {lab: np.array([dmc[lab].get(str(c), np.nan) for c in range(M)])
         for fam in fam_rows for lab in fam_rows[fam]}
    sc = np.array(scored)

    def curve(fam, rate, xi, yi):
        xs = [P[l][xi].mean() for l in fam_rows[fam]]
        ys = [np.nanmean(D[l][yi]) for l in fam_rows[fam]]
        return interp(xs, ys, rate)

    sel_rate = round(res["selection"]["packets_per_day"], 2)
    comparisons = {"ablation": [("C", "A"), ("C", "B"), ("C", "D")],
                   "taps": [("T8", "T12"), ("T8", "T7")]}
    rates = {"ablation": sorted({15.0, 10.0, 6.0, sel_rate}, reverse=True),
             "taps": [15.0, 10.0, 6.0]}
    idx = rng.integers(0, len(sc), size=(a.bootstrap, len(sc)))
    abl = {}
    for group, pairs in comparisons.items():
        for rate in rates[group]:
            key = f"{rate:g}"
            point = {f: curve(f, rate, slice(None), sc) for f in {x for p in pairs for x in p}}
            draws = {f: [curve(f, rate, sc[i], sc[i]) for i in idx]
                     for f in point}
            for f1, f2 in pairs:
                if point[f1] is None or point[f2] is None:
                    continue
                dd = np.array([x - y for x, y in zip(draws[f1], draws[f2])
                               if x is not None and y is not None], np.float64)
                abl.setdefault(group, {}).setdefault(key, {})[f"{f1}-{f2}"] = {
                    "point": point[f1] - point[f2],
                    f"{f1}": point[f1], f"{f2}": point[f2],
                    "ci_difference": [float(np.percentile(dd, LO)),
                                      float(np.percentile(dd, HI))],
                    "share_draws_negative": float((dd < 0).mean()),
                    "n_draws": int(len(dd))}
    out["matched_rate"] = abl

    # the point estimates must reproduce the printed tables
    chk = {}
    for f in ("A", "B", "C", "D"):
        for r in rates["ablation"]:
            got = curve(f, r, slice(None), sc)
            want = frozen["ablation"][f]["dMASE_vs_control_at"].get(str(float(r)))
            chk[f"{f}@{r:g}"] = (got, want)
    for t in ("T7", "T12"):
        for r in rates["taps"]:
            got = curve(t, r, slice(None), sc)
            want = frozen["tap_ablation"][t]["dMASE_vs_control_at"].get(str(float(r)))
            chk[f"{t}@{r:g}"] = (got, want)
    out["reproduces_frozen"] = {
        k: {"got": g, "frozen": w,
            "match": (g is None and w is None) or
                     (g is not None and w is not None and abs(g - w) < 1e-9)}
        for k, (g, w) in chk.items()}
    out["all_reproduce"] = all(v["match"] for v in out["reproduces_frozen"].values())

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)

    nh = out["next_hour_error"]
    print(f"scored clients: {len(scored)}   resamples: {a.bootstrap}   95% CI")
    print(f"\nNEXT-HOUR ERROR  (pooled, as printed: batch "
          f"{nh['pooled_reported_kwh'][B]:.1f} -> scheme "
          f"{nh['pooled_reported_kwh'][S]:.1f} kWh)")
    for unit in ("per_client_kwh", "per_client_scaled"):
        x = nh[unit]["batch_minus_scheme"]
        print(f"  {unit:<18s} batch {x['mean_batch']:.4g} {x['ci_batch']}  scheme "
              f"{x['mean_scheme']:.4g} {x['ci_scheme']}")
        print(f"  {'':<18s} difference {x['mean_difference']:.4g} {x['ci_difference']}"
              f"  ratio {x['ratio_of_means']:.3f} {x['ci_ratio']}  clients where "
              f"the scheme wins {x['clients_a_larger']}/{x['n_clients']}")
    m = out["mase_increment"]["batch_minus_scheme"]
    print(f"\nMASE INCREMENT  batch {m['mean_batch']:.4f} {m['ci_batch']}  scheme "
          f"{m['mean_scheme']:.4f} {m['ci_scheme']}  scheme wins on "
          f"{m['clients_a_larger']}/{m['n_clients']} clients")
    print("\nMATCHED PACKET RATE (difference, 95% CI)")
    for group, byrate in abl.items():
        for r, cmp_ in byrate.items():
            for k, v in cmp_.items():
                print(f"  {group:<8s} {r:>5s} pkt/d  {k:<7s} {v['point']:+.4f} "
                      f"{[round(x, 4) for x in v['ci_difference']]}")
    print(f"\npoint estimates reproduce the frozen tables: {out['all_reproduce']}")
    if not out["all_reproduce"]:
        for k, v in out["reproduces_frozen"].items():
            if not v["match"]:
                print(f"  MISMATCH {k}: {v}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
