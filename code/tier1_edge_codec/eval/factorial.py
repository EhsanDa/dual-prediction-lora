"""
factorial.py  --  the 2 x 2 factorial the reviewers asked for (request F).
Conference revision, 2026-09-29.

    WHEN to transmit   x   HOW the gateway fills an untold hour
    event-triggered        shared predictor      policy C (the proposed scheme)
    event-triggered        zero-order hold       policy A (persistence deadband)
    periodic               zero-order hold       sim/periodic.py
    periodic               shared predictor      sim/periodic_predict.py

compared at MATCHED AIRTIME (SF12 seconds per day), which is what the budget
is and what Reviewer 3 asked for -- not at matched packets per day, the
submitted paper's axis, because the frame sizes differ.

THE ASSEMBLY FIX THIS MODULE DEPENDS ON
=======================================
A periodic refresh corrects the previous `period` hours retroactively. The
submitted evaluation assembled every policy's look-back with the day rule
(sim/reconstruct.py::assemble_window), which reads every current-day hour from
`u_within` -- so for periodic, hours already delivered exactly by a later
refresh were scored at their old held value. Here the periodic rows use
assemble_window_horizon with refresh_horizon(period); the codec rows use the
day rule, unchanged. Two periodic rows are ALSO scored with the old rule: they
must reproduce the submitted Table V (0.0587 at 2 h, 0.1281 at 3 h) before any
corrected number is believed, and their difference from the corrected rows is
the size of the bias. docs/findings/2026-09-29-periodic-baseline-assembly.md.

Run from code/tier1_edge_codec/:

    python -m eval.factorial --data_dir ../tier2_data --checkpoint ../dlinear.pt \\
        --out ../../results/revision/factorial.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import paths                                                      # noqa: E402

from edge.codec import CodecParams, PARAMS_SEND_ON_DELTA          # noqa: E402
from eval.metrics_tier1 import communication_cost, delta_per_client  # noqa: E402
from eval.run_eval import (build_daily_batch_forecasts, held_forecast,  # noqa: E402
                           per_client_cost)
from sim.periodic import periodic_nbytes, run_periodic            # noqa: E402
from sim.periodic_predict import run_periodic_predict             # noqa: E402
from sim.reconstruct import (assemble_window_horizon, build_log1p_arrays,  # noqa: E402
                             day_start_horizon, refresh_horizon)
from sim.run_codec import Store, run_policy                       # noqa: E402
from tier2.adapter import (Accum, distribution, forecast_kwh,     # noqa: E402
                           frac_no_skill, iter_batches, load_frozen_dlinear,
                           make_window_builder, store_manifest)

CONTROL, BATCH = "control", "batch-daily"
CELLS = {("event", "predictor"): "C", ("event", "zoh"): "A",
         ("periodic", "zoh"): "PZ", ("periodic", "predictor"): "PP"}


def row_specs(alphas_c, alphas_a, periods):
    """(label, family, knob, horizon_fn, policy key, bytes per periodic frame).
    Rows sharing a policy key share one simulation and one pair of arrays."""
    specs = []
    for al in alphas_c:
        specs.append((f"C@{al:g}", "C", al, day_start_horizon, ("C", al), None))
    for al in alphas_a:
        specs.append((f"A@{al:g}", "A", al, day_start_horizon, ("A", al), None))
    for p in periods:
        nb = periodic_nbytes(p)
        specs.append((f"PZ@{p}", "PZ", p, refresh_horizon(p), ("PZ", p), nb))
        if p in (2, 3):   # the submitted assembly, to reproduce Table V
            specs.append((f"PZ-dayrule@{p}", "PZ-dayrule", p, day_start_horizon,
                          ("PZ", p), nb))
        specs.append((f"PP@{p}", "PP", p, refresh_horizon(p), ("PP", p), nb))
    return specs


def simulate(store, key, w_global, k_mu, workers):
    kind, knob = key
    if kind == "C":
        return run_policy(store, knob, CodecParams(k_mu=k_mu).with_w_init(w_global),
                          backend="batch")
    if kind == "A":
        return run_policy(store, knob, PARAMS_SEND_ON_DELTA, backend="batch")
    if kind == "PZ":
        return run_periodic(store, knob)
    return run_periodic_predict(store, knob, w_global, k_mu, workers=workers)


@torch.no_grad()
def score(model, wb, sel, recon, batch_fc, batch_size, device):
    names = [CONTROL, BATCH] + list(recon)
    acc = {n: Accum(wb.M, wb.H, (24,)) for n in names}
    acc_sn24 = Accum(wb.M, wb.H, (24,))
    l1_ae = {n: np.zeros(24) for n in names}
    l1_n = {n: np.zeros(24) for n in names}
    l1c_ae = {n: np.zeros(wb.M) for n in names}
    l1c_n = {n: np.zeros(wb.M) for n in names}
    done, t0 = 0, time.time()
    for chunk, b in iter_batches(wb, sel, batch_size, device):
        y = b["y_true_kwh"].cpu().numpy()
        v = b["valid"].cpu().numpy()
        cl = chunk[:, 1]
        hod = chunk[:, 0] % 24
        acc_sn24.add(y, b["sn24_kwh"].cpu().numpy(), v, cl)
        fcs = {CONTROL: forecast_kwh(model, b).cpu().numpy()}
        fb, cov = held_forecast(batch_fc, chunk, wb.H)
        fcs[BATCH] = fb
        for n, (ex, wi, hfn) in recon.items():
            yp = assemble_window_horizon(chunk[:, 0], chunk[:, 1], ex, wi,
                                         wb.L, hfn)
            yp = torch.as_tensor(np.ascontiguousarray(yp), dtype=torch.float32,
                                 device=device)
            fcs[n] = forecast_kwh(model, b, yp).cpu().numpy()
        for n, f in fcs.items():
            vv = v & cov if n == BATCH else v
            acc[n].add(y, f, vv, cl)
            m1 = vv[:, 0]
            e1 = np.abs(np.where(m1, f[:, 0] - y[:, 0], 0.0))
            np.add.at(l1_ae[n], hod, e1)
            np.add.at(l1_n[n], hod, m1.astype(float))
            np.add.at(l1c_ae[n], cl, e1)
            np.add.at(l1c_n[n], cl, m1.astype(float))
        done += len(chunk)
        if done % (batch_size * 400) < batch_size:
            print(f"  scored {done:,d} / {len(sel):,d} windows "
                  f"({time.time() - t0:.0f}s)", flush=True)
    lead1 = {n: (l1_ae[n] / np.maximum(l1_n[n], 1)) for n in names}
    return acc, acc_sn24, lead1, l1c_ae, l1c_n


def interp_family(xs, ys, target):
    """Linear interpolation of a family's curve at `target` airtime; None if
    the target lies outside the family's measured range (no extrapolation)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    o = np.argsort(xs)
    if target < xs[o][0] or target > xs[o][-1]:
        return None
    return float(np.interp(target, xs[o], ys[o]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="../tier2_data")
    ap.add_argument("--checkpoint", default="../dlinear.pt")
    ap.add_argument("--pretrained", default=paths.inp("edge_pretrained.json"))
    ap.add_argument("--k_mu", type=int, default=5)
    ap.add_argument("--alphas_c", default="0.5,0.7,1")
    ap.add_argument("--alphas_a", default="0.5,1,2")
    ap.add_argument("--periods", default="2,3,4,6")
    ap.add_argument("--targets", default="",
                    help="airtimes (s/day, SF12) at which to compare the cells; "
                         "default = the airtimes of C@0.7 and C@1")
    ap.add_argument("--sf", type=int, default=12)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--recon_dir", default="_recon_cache_factorial",
                    help="spill directory; separate from run_eval's so the two "
                         "can run side by side")
    ap.add_argument("--max_samples", type=int, default=0,
                    help="dev only: subsample the test windows. NOT reportable")
    ap.add_argument("--out", default=paths.result("factorial.json"))
    a = ap.parse_args(argv)

    w_global = tuple(int(x) for x in json.load(open(a.pretrained))["w_global_q"])
    man = store_manifest(a.data_dir)
    model, cfg, ck = load_frozen_dlinear(a.checkpoint, man, device=a.device)
    wb = make_window_builder(a.data_dir, cfg.lookback_len, cfg.horizon_len)
    store = Store.load(a.data_dir)
    sel = wb.splits["test"]
    if a.max_samples:
        rng = np.random.default_rng(0)
        sel = sel[np.sort(rng.choice(len(sel), min(a.max_samples, len(sel)),
                                     replace=False))]
        print(f"[dev mode] {len(sel):,d} windows -- NOT reportable")

    # One policy at a time: simulate, spill its two arrays to disk, keep only
    # the cost vectors, free the rest. Holding every PolicyResult at once
    # (about ten (T, M) arrays each) exhausts the machine's memory.
    import gc
    from edge.protocol import EXCEPTION_BYTES
    t0 = time.time()
    specs = row_specs([float(x) for x in a.alphas_c.split(",")],
                      [float(x) for x in a.alphas_a.split(",")],
                      [int(x) for x in a.periods.split(",")])
    os.makedirs(a.recon_dir, exist_ok=True)
    done = {}
    recon, cost, air_pc, pkt_pc = {}, {}, {}, {}
    for lab, fam, knob, hfn, key, nb in specs:
        if key not in done:
            r = simulate(store, key, w_global, a.k_mu, a.workers)
            ex, wi = build_log1p_arrays(r, store)
            tag = f"{key[0]}_{key[1]:g}".replace(".", "p")
            pe = os.path.join(a.recon_dir, f"{tag}_exact.npy")
            pw = os.path.join(a.recon_dir, f"{tag}_within.npy")
            np.save(pe, ex)
            np.save(pw, wi)
            del ex, wi
            if nb is not None:
                zeros = np.zeros_like(r.resync_bytes)
                cc = communication_cost(r.tx, zeros, r.active, bytes_per_tx=nb)
                pcc = per_client_cost(r.tx, zeros, r.active, nb, sf=a.sf)
            else:
                cc = communication_cost(r.tx, r.resync_bytes, r.active)
                pcc = per_client_cost(r.tx, r.resync_bytes, r.active,
                                      EXCEPTION_BYTES, sf=a.sf)
            assert (float(pcc["airtime_per_day_s"].mean())
                    == cc["airtime_per_day_s"][a.sf])
            done[key] = (np.load(pe, mmap_mode="r"), np.load(pw, mmap_mode="r"),
                         cc, pcc)
            del r
            gc.collect()
        ex, wi, cc, pcc = done[key]
        recon[lab] = (ex, wi, hfn)
        cost[lab] = {"family": fam, "knob": knob,
                     "packets_per_day_mean": cc["packets_per_day_mean"],
                     "airtime_per_day_s": cc["airtime_per_day_s"][a.sf],
                     "assembly": ("day rule" if hfn is day_start_horizon
                                  else "refresh horizon")}
        air_pc[lab] = pcc["airtime_per_day_s"]
        pkt_pc[lab] = pcc["packets_per_day"]
        print(f"  {lab:>16s}  pkt/day {cc['packets_per_day_mean']:6.2f}  "
              f"SF{a.sf} {cc['airtime_per_day_s'][a.sf]:7.3f} s/day  "
              f"[{cost[lab]['assembly']}]  ({time.time() - t0:.0f}s)", flush=True)

    print("building the daily-batch baseline", flush=True)
    batch_fc, _ = build_daily_batch_forecasts(model, wb, sel, a.batch_size,
                                              a.device)
    print(f"scoring {len(recon) + 2} rows over {len(sel):,d} windows", flush=True)
    acc, acc_sn24, lead1, l1c_ae, l1c_n = score(model, wb, sel, recon, batch_fc,
                                                a.batch_size, a.device)
    sn_pc = acc_sn24.per_client(wb.mase_scale)["MAE"]
    names = [CONTROL, BATCH] + list(recon)
    pcs = {n: acc[n].per_client(wb.mase_scale, sn_pc) for n in names}
    batch_l1 = float(lead1[BATCH].mean())
    metrics = {}
    for n in names:
        dm = delta_per_client(pcs[n]["MASE"], pcs[CONTROL]["MASE"])
        metrics[n] = {
            "MASE": distribution(pcs[n]["MASE"])["mean"],
            "SS<0": frac_no_skill(pcs[n]["SS"]),
            "delta_MASE_vs_control": distribution(dm),
            "lead1_mae_kwh_mean_over_issue_hours": float(lead1[n].mean()),
            "gain_at_lead1_vs_batch_kwh": batch_l1 - float(lead1[n].mean()),
        }

    # ---------------------------------------------------- the factorial table
    clients = sorted(int(c) for c in pcs[CONTROL]["MASE"]
                     if np.isfinite(pcs[CONTROL]["MASE"][c]))
    dm_vec = {}
    for n in recon:
        d = delta_per_client(pcs[n]["MASE"], pcs[CONTROL]["MASE"])
        dm_vec[n] = np.array([d.get(c, np.nan) for c in range(wb.M)])
    fam_rows = {}
    for lab in recon:
        fam_rows.setdefault(cost[lab]["family"], []).append(lab)
    targets = ([float(x) for x in a.targets.split(",")] if a.targets else
               [cost["C@0.7"]["airtime_per_day_s"], cost["C@1"]["airtime_per_day_s"]])
    idx_all = np.arange(wb.M)
    scored = np.array(clients)
    table = {}
    rng = np.random.default_rng(a.seed)
    boots = rng.integers(0, len(scored), size=(a.bootstrap, len(scored)))
    for tgt in targets:
        # point estimate: airtime axis over ALL clients (as the paper reports
        # it), dMASE over the scored clients (as the paper reports it)
        point = {}
        for fam, labs in fam_rows.items():
            xs = [air_pc[l][idx_all].mean() for l in labs]
            ys = [np.nanmean(dm_vec[l][scored]) for l in labs]
            point[fam] = interp_family(xs, ys, tgt)
        draws = {fam: [] for fam in fam_rows}
        for bi in range(a.bootstrap):
            cs = scored[boots[bi]]
            for fam, labs in fam_rows.items():
                xs = [air_pc[l][cs].mean() for l in labs]
                ys = [np.nanmean(dm_vec[l][cs]) for l in labs]
                draws[fam].append(interp_family(xs, ys, tgt))

        def ci(v):
            v = np.array([x for x in v if x is not None], float)
            return ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                    if v.size else None)
        contrasts = {
            "schedule | predictor  (PP - C)": ("PP", "C"),
            "schedule | ZOH        (PZ - A)": ("PZ", "A"),
            "reconstruction | event    (A - C)": ("A", "C"),
            "reconstruction | periodic (PZ - PP)": ("PZ", "PP"),
            "assembly bias (PZ-dayrule - PZ)": ("PZ-dayrule", "PZ"),
        }
        con = {}
        for name, (f1, f2) in contrasts.items():
            if point.get(f1) is None or point.get(f2) is None:
                con[name] = None
                continue
            d = [x - y for x, y in zip(draws[f1], draws[f2])
                 if x is not None and y is not None]
            con[name] = {"point": point[f1] - point[f2], "ci95": ci(d),
                         "n_draws": len(d),
                         "share_draws_positive": float(np.mean(np.array(d) > 0))
                         if d else None}
        table[f"{tgt:.6f}"] = {
            "target_airtime_s": tgt,
            "cells": {fam: {"point": point[fam], "ci95": ci(draws[fam])}
                      for fam in fam_rows},
            "contrasts": con,
        }

    doc = {"what": "2x2 factorial at matched SF12 airtime -- conference "
                   "revision request F",
           "store": a.data_dir, "checkpoint": a.checkpoint,
           "n_windows": int(len(sel)), "reportable": not a.max_samples,
           "n_clients_scored": len(clients), "bootstrap": a.bootstrap,
           "seed": a.seed, "sf": a.sf, "rows": cost, "metrics": metrics,
           "factorial": table,
           "per_client": {
               "delta_MASE_vs_control": {n: dm_vec[n].tolist() for n in recon},
               "airtime_per_day_s": {n: air_pc[n].tolist() for n in recon},
               "packets_per_day": {n: pkt_pc[n].tolist() for n in recon},
               "lead1_abs_err_sum_kwh": {n: l1c_ae[n].tolist() for n in names},
               "lead1_n": {n: l1c_n[n].tolist() for n in names}}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1, default=float)
    print(f"\nwrote {a.out}")
    print("\nrow                 pkt/day  air s/day  dMASE    MASE    SS<0   gain@l1")
    for n in names:
        c = cost.get(n, {})
        m = metrics[n]
        print(f"{n:>16s} {c.get('packets_per_day_mean', float('nan')):8.2f} "
              f"{c.get('airtime_per_day_s', float('nan')):9.3f} "
              f"{m['delta_MASE_vs_control']['mean']:8.4f} {m['MASE']:7.4f} "
              f"{m['SS<0']:6.1%} {m['gain_at_lead1_vs_batch_kwh']:+8.2f}")
    for k, t in table.items():
        print(f"\nat {t['target_airtime_s']:.3f} s/day:")
        for fam, v in t["cells"].items():
            print(f"  {fam:>11s} {v['point']}  CI {v['ci95']}")
        for name, v in t["contrasts"].items():
            print(f"  {name:<40s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
