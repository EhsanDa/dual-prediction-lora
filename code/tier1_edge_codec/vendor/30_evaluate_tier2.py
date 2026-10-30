"""
30_evaluate_tier2.py  --  the final results table.

Defaults follow the spec: FULL test set, no subsampling (E5), no outlier
filter (E6), all metrics in kWh after inversion (E1), all scored against
`series_true` (C5) with the per-step validity mask live (C4).

REVIEW FIXES
------------
C3  The ridge (L, lambda) is READ from sweep_selection.json. There is no
    lambda default any more. The previous run silently evaluated lambda=1000
    while validation had selected 1e4 -- a 10x under-regularised model, i.e.
    selection on validation followed by evaluation of a different model.
C2  Ridge predictions use the per-horizon Grams: step h is predicted from the
    covariates at step h, not from step 1's.
M14 Every checkpoint is checked against the manifest's feature-spec hash and
    against every other checkpoint on (lookback, horizon, n_dynamic,
    n_static). Raises, does not warn.
S1  Persistence baseline added.
S2  Climatology is now hour-of-day x day-of-week with a holiday override.
S3  SparseTSF is annotated with its effective parameterisation so its row
    cannot be read as evidence about the method at its published setting.
m16 RSE/CORR at h in {3, 6, 12, 24}, not h=24 alone.
m17 RMSE_h and nRMSE are printed.
"""
import argparse, json
import numpy as np
import torch

from evalaccum_tier2 import Accum
from metrics_tier2 import distribution, frac_no_skill
from models_tier2 import TiDEConfig, build
from windows_tier2 import WindowBuilder


@torch.no_grad()
def score(models, wb, sel, batch_size, device, ridge=None, corr_steps=(3, 6, 12, 24)):
    """ONE streaming pass. Nothing is materialised.

    The previous version held every row's predictions for the whole split:
    12 rows x 1,675,489 windows x 24 steps x 4 B = 1.93 GB, on top of a ridge
    design matrix that peaked near 20 GB. Both are gone -- memory is now
    O(n_clients + H) per row and independent of the split size. The
    accumulators reproduce metrics_tier2 exactly; evalaccum_tier2's self-test
    asserts it to machine precision.
    """
    for m in models.values():
        m.eval().to(device)
    names = (["Per-client mean", "Climatology h x DoW", "Persistence y[t]",
              "Seasonal-Naive-24", "Seasonal-Naive-168"]
             + ([ridge["label"]] if ridge else []) + list(models))
    acc = {n: Accum(wb.M, wb.H, corr_steps) for n in names}

    done = 0
    for i in range(0, len(sel), batch_size):
        chunk = sel[i:i + batch_size]
        b = wb.build(chunk, device=device)
        y = b["y_true_kwh"].cpu().numpy()
        v = b["valid"].cpu().numpy()
        cl = chunk[:, 1]

        acc["Per-client mean"].add(y, b["mean_kwh"].cpu().numpy(), v, cl)
        acc["Climatology h x DoW"].add(y, b["clim_kwh"].cpu().numpy(), v, cl)
        acc["Persistence y[t]"].add(y, b["persist_kwh"].cpu().numpy(), v, cl)
        acc["Seasonal-Naive-24"].add(y, b["sn24_kwh"].cpu().numpy(), v, cl)
        acc["Seasonal-Naive-168"].add(y, b["sn168_kwh"].cpu().numpy(), v, cl)
        if ridge:
            acc[ridge["label"]].add(y, ridge["fn"](chunk), v, cl)
        for name, m in models.items():
            yh, st = m(b["y_past"], b["x_past"], b["x_fut"], b["s"], b["w"])
            acc[name].add(y, m.to_kwh(yh, st).cpu().numpy(), v, cl)

        done += len(chunk)
        if i % (batch_size * 200) == 0:
            print(f"  scored {done:,d} / {len(sel):,d} windows", flush=True)
    return acc, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--gram", default=None)
    ap.add_argument("--columns", default="gram_L720_columns.json")
    ap.add_argument("--selection", default="sweep_selection.json",
                    help="C3: (L, lambda) chosen on validation. Required with --gram.")
    ap.add_argument("--data_dir", default="tier2_data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_samples", type=int, default=0,
                    help="0 = full split (the only reportable setting)")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    models, meta, ref_cfg = {}, {}, None
    for path in a.checkpoints:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = TiDEConfig(**ck["cfg"])
        m = build(ck["model"], cfg)
        m.load_state_dict(ck["state_dict"])
        models[ck["model"]] = m
        meta[ck["model"]] = {
            "params": sum(p.numel() for p in m.parameters()),
            "converged": ck.get("converged"),
            "convergence_test": ck.get("convergence_test"),
            "epochs": ck.get("epochs_run"),
            "val_full": ck.get("val_full"),
            "feature_spec_hash": ck.get("feature_spec_hash"),
            "cfg": cfg,
        }
        if ref_cfg is None:
            ref_cfg, ref_name = cfg, ck["model"]
        else:                                                       # M14
            for f in ("lookback_len", "horizon_len", "n_dynamic", "n_static"):
                assert getattr(cfg, f) == getattr(ref_cfg, f), (
                    f"checkpoint '{ck['model']}' has {f}={getattr(cfg, f)} but "
                    f"'{ref_name}' has {getattr(ref_cfg, f)}. They cannot be "
                    f"compared on one window set.")

    L = ref_cfg.lookback_len if ref_cfg else 336
    wb = WindowBuilder(a.data_dir, lookback=L, horizon_len=24)
    for n, mt in meta.items():                                      # M14
        assert mt["feature_spec_hash"] == wb.manifest["feature_spec_hash"], (
            f"checkpoint '{n}' was trained under feature spec "
            f"{mt['feature_spec_hash'][:16]} but the data is "
            f"{wb.manifest['feature_spec_hash'][:16]}. Retrain or repreprocess.")

    sel = wb.splits[a.split]
    if a.max_samples:
        sel = sel[np.sort(np.random.default_rng(0).choice(
            len(sel), size=min(a.max_samples, len(sel)), replace=False))]
        print(f"[dev mode] {len(sel):,d} windows -- NOT reportable")

    ridge_note, ridge_spec = "", None
    if a.gram:
        from gram_tier2 import (HorizonGrams, Standardizer, column_names,
                                predict_per_horizon, select_columns)
        s = json.load(open(a.selection))                            # C3
        gm = json.load(open(a.columns))
        assert gm["feature_spec_hash"] == wb.manifest["feature_spec_hash"]
        best = dict(s["selected"])
        # BLOCKING 3 / V6. If Ridge runs at the sweep-optimal L while the
        # neural models run at another, Ridge and DLinear no longer share a
        # feature space and the superset relationship -- the entire point of
        # the matched comparison -- collapses. For the headline table Ridge is
        # PINNED to the checkpoints' look-back. The sweep-optimal row is still
        # printed, clearly labelled as unmatched.
        sweep_L = best["L"]
        if a.checkpoints and sweep_L != L:
            print(f"[matched-table constraint] the sweep selected L={sweep_L} "
                  f"but the checkpoints are at L={L}. Ridge is pinned to "
                  f"L={L} so the feature-superset comparison holds; the "
                  f"sweep-optimal configuration is reported separately.")
            match = [r for r in s["all"] if r["L"] == L and not r["extra"]]
            assert match, (f"the sweep contains no L={L} row, so no matched "
                           f"Ridge can be built. Re-run 15_sweep_L_lambda with "
                           f"L={L} among the configurations.")
            best = dict(match[0])
        names = gm["names"]
        std = Standardizer.from_dict(names, gm["standardizer"])
        cols = select_columns(names, best["L"], tuple(best["extra"]))
        W = HorizonGrams.load(a.gram).solve(cols, best["lambda"])
        ridge_spec = {"label": f"Ridge (L={best['L']}, matched)",
                      "fn": lambda ch: predict_per_horizon(
                          W, wb, ch, names, cols, std, batch=len(ch))}
        ridge_note = (f"Ridge: L={best['L']} extra={best['extra']} "
                      f"lambda={best['lambda']:g}, selected on validation "
                      f"({s['criterion']}), gram sha {s['gram_sha']}")
    cl = sel[:, 1]
    print(f"\nsplit '{a.split}': {len(sel):,d} windows x 24 steps = "
          f"{len(sel)*24:,d} forecasts, {len(np.unique(cl))} clients")
    print(f"L={L}  H=24  purge={wb.purge}h  index L={wb.index_L}  "
          f"{'FULL split' if not a.max_samples else 'SUBSAMPLED'}")
    if ridge_note:
        print(ridge_note)
    wb.print_index_report()
    print("scoring in a single streaming pass (memory is O(clients + H) per row)")
    acc, rows = score(models, wb, sel, a.batch_size, a.device, ridge_spec)
    sn_pc = acc["Seasonal-Naive-24"].per_client(wb.mase_scale)["MAE"]


    res, pcs = {}, {}
    print(f"\n{'model':<22s} {'MAE':>9s} {'RMSE':>10s} {'R/M':>5s} {'sMAPE':>7s} "
          f"{'MASE':>7s} {'MASEp50':>8s} {'SS':>7s} {'SS<0':>6s} {'nRMSE':>7s}")
    print("-" * 96)
    for n in rows:
        m = acc[n].pooled()
        pc = acc[n].per_client(wb.mase_scale, sn_pc)
        dm, ds, dn = (distribution(pc["MASE"]), distribution(pc["SS"]),
                      distribution(pc["nRMSE"]))
        res[n] = {**m, "MASE": dm["mean"], "MASE_p50": dm["p50"],
                  "SS": ds["mean"], "SS<0": frac_no_skill(pc["SS"]),
                  "nRMSE": dn["mean"]}
        pcs[n] = pc
        print(f"{n:<22s} {m['MAE']:9.2f} {m['RMSE']:10.2f} {m['RMSE/MAE']:5.2f} "
              f"{m['sMAPE%']:7.2f} {dm['mean']:7.4f} {dm['p50']:8.4f} "
              f"{ds['mean']:+7.3f} {res[n]['SS<0']:6.1%} {dn['mean']:7.3f}")
    print("  MASE and SS are scale-free and are the headline. MAE/RMSE are kWh")
    print("  and are dominated by the largest meters; read R/M (RMSE/MAE) as a")
    print("  tail diagnostic -- a well-behaved distribution sits at 1.25-2.5,")
    print("  though pooled across 320 meters it also absorbs between-client")
    print("  scale spread, so do not over-read model-to-model differences in it.")
    print("  SS<0 counts clients STRICTLY worse than seasonal-naive (M21).")

    print(f"\nper-client MASE distribution   (lead with p90; p99 over 320 "
          f"clients is effectively the max of ~3, m15)")
    print(f"{'model':<22s} {'p10':>8s} {'p50':>8s} {'p90':>8s} {'p99':>8s} {'n':>5s}")
    for n in rows:
        d = distribution(pcs[n]["MASE"])
        print(f"{n:<22s} {d['p10']:8.4f} {d['p50']:8.4f} {d['p90']:8.4f} "
              f"{d['p99']:8.4f} {d['n']:5d}")

    show = [0, 3, 7, 11, 15, 19, 23]
    for label, key in (("MAE", "MAE_h"), ("RMSE", "RMSE_h")):        # m17
        print(f"\n{label} by horizon step (kWh)")
        print(f"{'model':<22s}" + "".join(f"{'h'+str(h+1):>9s}" for h in show))
        for n in rows:
            c = acc[n].per_horizon()[key]
            print(f"{n:<22s}" + "".join(f"{c[h]:9.1f}" for h in show))

    print(f"\nappendix -- LSTNet protocol (RSE/CORR; positioning only)")     # m16
    hs = (3, 6, 12, 24)
    print(f"{'model':<22s}" + "".join(f"{'RSE h'+str(h):>10s}" for h in hs)
          + "".join(f"{'CORR h'+str(h):>11s}" for h in hs))
    for n in rows:
        ap_ = acc[n].lstnet()
        print(f"{n:<22s}"
              + "".join(f"{ap_[h]['RSE']:10.4f}" for h in hs)
              + "".join(f"{ap_[h]['CORR']:11.4f}" for h in hs))

    if meta:
        print(f"\ntraining budget and convergence")
        for n, mt in meta.items():
            t = mt.get("convergence_test") or {}
            print(f"  {n:<20s} {mt['params']:>9,d} params  epochs={mt['epochs']}  "
                  f"convergence={t.get('kind', '?')}: {mt['converged']}")
            if mt.get("val_full"):
                v = mt["val_full"]
                print(f"{'':24s}full-split val MASE {v['MASE']:.4f} "
                      f"vs test {res[n]['MASE']:.4f}")
    if "sparsetsf" in rows:                                          # S3
        print(f"\n  NOTE on SparseTSF: at L={L}, H=24, period=24 it reduces to "
              f"Linear({L//24} -> 1)\n  shared across all 24 phases -- "
              f"{L//24 + 1} weights. That is far below the published LTSF "
              f"setting\n  (L=720). This row is evidence about THIS "
              f"configuration, not about the method.")

    # ---- ship rule (spec 1.4), gated on convergence. This block was lost in
    # the streaming rewrite and is restored here.
    print("\n" + "=" * 68)
    if "tide_lite" in res:
        alts = [k for k in res if k != "tide_lite"]
        best_alt = min(alts, key=lambda k: res[k]["MASE"])
        gain = (res[best_alt]["MASE"] - res["tide_lite"]["MASE"]) / res[best_alt]["MASE"]
        print(f"SHIP RULE: TiDE-Lite vs best alternative ({best_alt})")
        print(f"  mean per-client MASE {res['tide_lite']['MASE']:.4f} vs "
              f"{res[best_alt]['MASE']:.4f}  ->  {gain:+.1%}")
        unconv = [n for n, m in meta.items() if not m.get("converged")]
        if unconv:
            print(f"  VERDICT WITHHELD. Not converged: {unconv}. The comparison "
                  f"would measure the training budget, not the architecture.")
        elif gain >= 0.03:
            print("  VERDICT: ship TiDE-Lite. It clears the 3% bar against a "
                  "converged field.")
        else:
            print(f"  VERDICT: ship {best_alt}. TiDE-Lite does not clear 3%, "
                  f"and a smaller model is strictly better for the edge and "
                  f"FL constraints.")
        print("  NOTE: single seed, single chronological split, no rolling "
              "origin, no MCS.\n  This is a point estimate, not a defensible "
              "margin claim.")
    print("=" * 68)

    json.dump({"n_windows": int(len(sel)), "metrics": res, "budget":
               {k: {kk: vv for kk, vv in v.items() if kk != "cfg"}
                for k, v in meta.items()}},
              open("tier2_results.json", "w"), indent=2, default=str)
    print("\nwrote tier2_results.json")


if __name__ == "__main__":
    main()
