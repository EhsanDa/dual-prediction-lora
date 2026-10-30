"""What does the sync_safe mode cost at the reported configuration?

Conference revision, request A (2026-10-11). The revised paper names the
compile-time `sync_safe` mode as the one that removes the lost-acknowledgement
failure of the reported (legacy) mode. The submitted paper priced it at
"roughly ten percent more transmissions", a figure no artifact supports; the
revision dropped it. This measures the price on the reported pipeline.

Both modes run through the same code path the paper's codec rows use
(sim/run_codec.py::run_policy, batch backend, the shipped global weights,
k_mu = 5), on a lossless channel, so the only difference is the weight-update
rule: legacy updates on the TRUE day once the RESYNC is acknowledged;
sync_safe updates unconditionally on the RECONSTRUCTED day both parties hold.

Reported per mode and alpha: packets/day, SF12 airtime/day (the paper's
formula, eval.metrics_tier1.communication_cost), reconstruction MAE, and --
with --score -- the forecast-level MASE increment against the control over the
full test split, scored exactly as eval/factorial.py scores its rows. The two
modes are also compared at the operating point's airtime (11.74 s/day) by
linear interpolation along each mode's airtime curve, with a paired client
bootstrap (the factorial's method).

Check before believing anything: legacy at alpha = 0.7 must reproduce the
frozen artifact's 9.367 pkt/day and 11.740 s/day.

    python -m eval.sync_safe_cost --data_dir ../tier2_data --score
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import paths                                                      # noqa: E402

from edge.codec import CodecParams                                # noqa: E402
from edge.protocol import EXCEPTION_BYTES                         # noqa: E402
from eval.metrics_tier1 import (communication_cost, delta_per_client,  # noqa: E402
                                recon_fidelity)
from eval.run_eval import per_client_cost                         # noqa: E402
from sim.run_codec import Store, run_policy                       # noqa: E402

FROZEN_C07 = {"packets_per_day": 9.366974179698751,
              "airtime_sf12_s": 11.74019982883386}


def interp(xs, ys, target):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    o = np.argsort(xs)
    if target < xs[o][0] or target > xs[o][-1]:
        return None
    return float(np.interp(target, xs[o], ys[o]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../tier2_data")
    ap.add_argument("--checkpoint", default="../dlinear.pt")
    ap.add_argument("--pretrained", default=paths.inp("edge_pretrained.json"))
    ap.add_argument("--k_mu", type=int, default=5)
    ap.add_argument("--alphas_legacy", default="0.5,0.7,1")
    ap.add_argument("--alphas_sync_safe", default="0.5,0.7,1,1.5")
    ap.add_argument("--target", type=float, default=FROZEN_C07["airtime_sf12_s"])
    ap.add_argument("--score", action="store_true",
                    help="also score the MASE increment (streams the test split)")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--recon_dir", default="_recon_cache_sync_safe")
    ap.add_argument("--out", default=paths.result(os.path.join("revision",
                                                               "sync_safe_cost.json")))
    a = ap.parse_args(argv)

    w_global = tuple(int(x) for x in json.load(open(a.pretrained))["w_global_q"])
    store = Store.load(a.data_dir)
    rows = ([("legacy", float(x)) for x in a.alphas_legacy.split(",")]
            + [("sync_safe", float(x)) for x in a.alphas_sync_safe.split(",")])

    os.makedirs(a.recon_dir, exist_ok=True)
    cost, air_pc, recon = {}, {}, {}
    t0 = time.time()
    for mode, al in rows:
        lab = f"{'L' if mode == 'legacy' else 'S'}@{al:g}"
        prm = CodecParams(k_mu=a.k_mu, sync_safe=(mode == "sync_safe")
                          ).with_w_init(w_global)
        r = run_policy(store, al, prm, backend="batch")
        cc = communication_cost(r.tx, r.resync_bytes, r.active)
        pcc = per_client_cost(r.tx, r.resync_bytes, r.active, EXCEPTION_BYTES, sf=12)
        assert float(pcc["airtime_per_day_s"].mean()) == cc["airtime_per_day_s"][12]
        live = r.active & ~store.missing
        fid = recon_fidelity(r.u_within, r.u_q, r.k, live, store.mase_scale)
        cost[lab] = {"mode": mode, "alpha": al,
                     "packets_per_day": cc["packets_per_day_mean"],
                     "airtime_sf12_s": cc["airtime_per_day_s"][12],
                     "mean_resync_bytes": cc.get("mean_resync_bytes"),
                     "recon_mae_kwh": fid["recon_mae_kwh"]}
        air_pc[lab] = pcc["airtime_per_day_s"]
        print(f"  {lab:>8s}  pkt/day {cost[lab]['packets_per_day']:7.3f}  "
              f"SF12 {cost[lab]['airtime_sf12_s']:7.3f} s/day  "
              f"recon MAE {fid['recon_mae_kwh']:8.2f} kWh  ({time.time() - t0:.0f}s)",
              flush=True)
        if a.score:
            from sim.reconstruct import build_log1p_arrays, day_start_horizon
            ex, wi = build_log1p_arrays(r, store)
            tag = lab.replace("@", "_").replace(".", "p")
            pe = os.path.join(a.recon_dir, f"{tag}_exact.npy")
            pw = os.path.join(a.recon_dir, f"{tag}_within.npy")
            np.save(pe, ex)
            np.save(pw, wi)
            del ex, wi
            recon[lab] = (np.load(pe, mmap_mode="r"), np.load(pw, mmap_mode="r"),
                          day_start_horizon)
        del r
        gc.collect()

    c07 = cost["L@0.7"]
    reproduces = (abs(c07["packets_per_day"] - FROZEN_C07["packets_per_day"]) < 1e-9
                  and abs(c07["airtime_sf12_s"] - FROZEN_C07["airtime_sf12_s"]) < 1e-9)
    print(f"legacy @0.7 reproduces the frozen artifact: {reproduces}")
    assert reproduces, "legacy alpha=0.7 does not reproduce the frozen cost"

    doc = {"what": "sync_safe vs legacy at the reported configuration -- "
                   "conference revision request A",
           "store": a.data_dir, "k_mu": a.k_mu, "channel": "lossless",
           "reproduces_frozen_C@0.7": reproduces, "rows": cost}

    # same alpha: the price in packets and airtime
    same = {}
    for al in sorted({c["alpha"] for c in cost.values()}):
        L, S = cost.get(f"L@{al:g}"), cost.get(f"S@{al:g}")
        if L and S:
            same[f"{al:g}"] = {
                "packets_ratio": S["packets_per_day"] / L["packets_per_day"],
                "airtime_ratio": S["airtime_sf12_s"] / L["airtime_sf12_s"],
                "recon_mae_ratio": S["recon_mae_kwh"] / L["recon_mae_kwh"]}
    doc["same_alpha"] = same

    if a.score:
        import torch  # noqa: F401
        from eval.factorial import CONTROL, BATCH, score
        from eval.run_eval import build_daily_batch_forecasts
        from tier2.adapter import (distribution, frac_no_skill, load_frozen_dlinear,
                                   make_window_builder, store_manifest)
        man = store_manifest(a.data_dir)
        model, cfg, _ = load_frozen_dlinear(a.checkpoint, man, device=a.device)
        wb = make_window_builder(a.data_dir, cfg.lookback_len, cfg.horizon_len)
        sel = wb.splits["test"]
        batch_fc, _ = build_daily_batch_forecasts(model, wb, sel, a.batch_size,
                                                  a.device)
        print(f"scoring {len(recon) + 2} rows over {len(sel):,d} windows", flush=True)
        acc, acc_sn24, lead1, _, _ = score(model, wb, sel, recon, batch_fc,
                                           a.batch_size, a.device)
        sn_pc = acc_sn24.per_client(wb.mase_scale)["MAE"]
        names = [CONTROL, BATCH] + list(recon)
        pcs = {n: acc[n].per_client(wb.mase_scale, sn_pc) for n in names}
        clients = np.array(sorted(int(c) for c in pcs[CONTROL]["MASE"]
                                  if np.isfinite(pcs[CONTROL]["MASE"][c])))
        dm = {}
        for n in recon:
            d = delta_per_client(pcs[n]["MASE"], pcs[CONTROL]["MASE"])
            dm[n] = np.array([d.get(c, np.nan) for c in range(wb.M)])
            cost[n]["dMASE_mean"] = float(np.nanmean(dm[n][clients]))
            cost[n]["dMASE_p90"] = distribution(d)["p90"]
            cost[n]["SS<0"] = frac_no_skill(pcs[n]["SS"])
            cost[n]["lead1_mae_kwh"] = float(lead1[n].mean())
        # matched airtime, paired client bootstrap (the factorial's method)
        fam = {"legacy": [l for l in recon if l.startswith("L@")],
               "sync_safe": [l for l in recon if l.startswith("S@")]}
        idx_all = np.arange(wb.M)

        def point(labs, cs_air, cs_dm, key):
            xs = [air_pc[l][cs_air].mean() for l in labs]
            ys = ([np.nanmean(dm[l][cs_dm]) for l in labs] if key == "dm"
                  else [cost[l]["recon_mae_kwh"] for l in labs])
            return interp(xs, ys, a.target)
        pt = {f: point(labs, idx_all, clients, "dm") for f, labs in fam.items()}
        rng = np.random.default_rng(a.seed)
        boots = rng.integers(0, len(clients), size=(a.bootstrap, len(clients)))
        diffs = []
        for bi in range(a.bootstrap):
            cs = clients[boots[bi]]
            s = point(fam["sync_safe"], cs, cs, "dm")
            l_ = point(fam["legacy"], cs, cs, "dm")
            if s is not None and l_ is not None:
                diffs.append(s - l_)
        diffs = np.array(diffs)
        doc["matched_airtime"] = {
            "target_airtime_s": a.target,
            "dMASE": pt,
            "difference_sync_safe_minus_legacy": {
                "point": (pt["sync_safe"] - pt["legacy"]
                          if None not in pt.values() else None),
                "ci95": ([float(np.percentile(diffs, 2.5)),
                          float(np.percentile(diffs, 97.5))] if diffs.size else None),
                "n_draws": int(diffs.size),
                "share_draws_positive": float(np.mean(diffs > 0)) if diffs.size else None},
        }
        doc["control_lead1_mae_kwh"] = float(lead1[CONTROL].mean())
        doc["n_windows"] = int(len(sel))
        doc["n_clients_scored"] = int(len(clients))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
