"""How robust is the schedule effect of the factorial? (request F, 2026-10-11)

eval/factorial.py separates when to transmit (event vs periodic) from how the
gateway fills silent hours (predictor vs zero-order hold) on the MASE
increment, the paper's scale-free criterion. With the predictor, event
triggering beats a periodic schedule by 0.0015 (0.0003-0.0025) at the
operating point's airtime -- resolved, but small. This reads the factorial's
own per-client arrays (no re-simulation, no re-scoring) and asks three further
questions of the same comparison, C (proposed) against PP (periodic refresh +
the same shared predictor):

  1. next-hour error, the paper's other headline criterion, interpolated along
     each family's airtime curve to 11.74 s/day, in kWh and per meter divided
     by its MASE scale; paired client bootstrap, the factorial's method;
  2. per-meter win/loss at the two nearest raw rows (C@0.7 at 11.74 s/day
     against PP@2 at 11.89 s/day -- PP spends slightly more);
  3. the distribution of the per-meter MASE-increment difference, tails
     included, at the same raw rows.

    python -m eval.factorial_lead1
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

ROOT = os.path.dirname(os.path.dirname(REPO))
QS = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def interp(xs, ys, target):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    o = np.argsort(xs)
    if target < xs[o][0] or target > xs[o][-1]:
        return None
    return float(np.interp(target, xs[o], ys[o]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--factorial", default=os.path.join(ROOT, "results_revision",
                                                        "factorial.json"))
    ap.add_argument("--per_client", default=os.path.join(ROOT, "results_revision",
                                                         "per_client_v2.json"))
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=paths.result(os.path.join(
        "revision", "factorial_lead1.json")))
    a = ap.parse_args(argv)

    f = json.load(open(a.factorial))
    pc = f["per_client"]
    scale_d = json.load(open(a.per_client))["mase_scale_per_client"]
    M = len(pc["lead1_n"]["control"])
    scale = np.array([scale_d[str(c)] for c in range(M)], float)
    target = f["factorial"]["11.740200"]["target_airtime_s"]

    def lead1(row):
        s = np.asarray(pc["lead1_abs_err_sum_kwh"][row], float)
        n = np.asarray(pc["lead1_n"][row], float)
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)

    dm = {r: np.asarray(v, float) for r, v in pc["delta_MASE_vs_control"].items()}
    scored = np.array([c for c in range(M) if np.isfinite(dm["C@0.7"][c])
                       and np.isfinite(lead1("control")[c])])
    fam = {"C": ["C@0.5", "C@0.7", "C@1"], "PP": ["PP@2", "PP@3", "PP@4", "PP@6"]}
    air = {r: np.asarray(pc["airtime_per_day_s"][r], float)
           for labs in fam.values() for r in labs}
    l1 = {r: lead1(r) for labs in fam.values() for r in labs}
    for r in ("control", "batch-daily"):
        l1[r] = lead1(r)

    def at_target(labs, cs, scaled):
        xs = [air[r][cs].mean() for r in labs]
        ys = [np.nanmean(l1[r][cs] / (scale[cs] if scaled else 1.0)) for r in labs]
        return interp(xs, ys, target)

    allc = np.arange(M)
    out = {"what": "schedule effect of the factorial on next-hour error, and "
                   "its per-meter distribution -- conference revision request F",
           "source": os.path.relpath(a.factorial, ROOT),
           "target_airtime_s": target, "n_clients_scored": int(len(scored)),
           "bootstrap": a.bootstrap, "seed": a.seed}

    rng = np.random.default_rng(a.seed)
    boots = rng.integers(0, len(scored), size=(a.bootstrap, len(scored)))
    lead = {}
    for scaled in (False, True):
        key = "per_meter_scaled" if scaled else "kwh"
        # point: airtime axis over ALL clients (as the paper and the factorial
        # report it), error over the scored clients
        pt = {}
        for name, labs in fam.items():
            xs = [air[r][allc].mean() for r in labs]
            ys = [np.nanmean(l1[r][scored] / (scale[scored] if scaled else 1.0))
                  for r in labs]
            pt[name] = interp(xs, ys, target)
        ref = {r: float(np.nanmean(l1[r][scored] / (scale[scored] if scaled else 1.0)))
               for r in ("control", "batch-daily")}
        d = []
        for bi in range(a.bootstrap):
            cs = scored[boots[bi]]
            p_, c_ = at_target(fam["PP"], cs, scaled), at_target(fam["C"], cs, scaled)
            if p_ is not None and c_ is not None:
                d.append(p_ - c_)
        d = np.array(d)
        lead[key] = {
            "C": pt["C"], "PP": pt["PP"], **ref,
            "difference_PP_minus_C": {
                "point": pt["PP"] - pt["C"],
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "share_draws_positive": float(np.mean(d > 0)), "n_draws": int(d.size)},
            "excess_over_control": {"C": pt["C"] - ref["control"],
                                    "PP": pt["PP"] - ref["control"]},
            "gain_over_batch": {"C": ref["batch-daily"] - pt["C"],
                                "PP": ref["batch-daily"] - pt["PP"]},
        }
    out["next_hour_error_at_target"] = lead

    # raw rows: C@0.7 (11.74 s/day) vs PP@2 (11.89 s/day)
    raw = {}
    for crit, va, vb in (("next_hour_error_kwh", l1["C@0.7"], l1["PP@2"]),
                         ("next_hour_error_scaled", l1["C@0.7"] / scale,
                          l1["PP@2"] / scale),
                         ("dMASE", dm["C@0.7"], dm["PP@2"])):
        x, y = va[scored], vb[scored]
        diff = y - x  # positive: the scheme is better
        raw[crit] = {"mean_C@0.7": float(np.mean(x)), "mean_PP@2": float(np.mean(y)),
                     "meters_C_better": int(np.sum(x < y)),
                     "meters_PP_better": int(np.sum(x > y)),
                     "meters_equal": int(np.sum(x == y)),
                     "difference_PP_minus_C_quantiles": {
                         str(q): float(np.percentile(diff, q)) for q in QS},
                     "C_quantiles": {str(q): float(np.percentile(x, q)) for q in QS},
                     "PP_quantiles": {str(q): float(np.percentile(y, q)) for q in QS}}
    out["raw_rows"] = {"C@0.7_airtime_s": float(air["C@0.7"].mean()),
                       "PP@2_airtime_s": float(air["PP@2"].mean()), **raw}

    # check: the raw-row means equal the factorial's own reported metrics
    for r in ("C@0.7", "PP@2"):
        rep = f["metrics"][r]["lead1_mae_kwh_mean_over_issue_hours"]
        mine = float(np.nanmean(l1[r][scored]))
        out.setdefault("check_lead1_vs_factorial_metrics", {})[r] = {
            "factorial_metric": rep, "per_meter_mean": mine}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("next_hour_error_at_target",
                                          "check_lead1_vs_factorial_metrics")},
                     indent=1))
    for crit, v in raw.items():
        print(crit, "C better", v["meters_C_better"], "PP better",
              v["meters_PP_better"])
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
