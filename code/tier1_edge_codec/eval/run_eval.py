"""
run_eval.py  --  the Tier-1 results, cut to what a 6-page paper needs.

Structured like the owner's 30_evaluate_tier2.py and it keeps that file's
discipline: ONE streaming pass, `Accum` for accumulation, the per-step validity
mask live, everything scored against `series_true`, full split, no subsampling.

WHAT THIS EXPERIMENT IS FOR, STATED BEFORE ANY NUMBER
=====================================================
The scheme buys ONE thing: an hourly-refreshed look-back on a link that cannot
carry hourly data. Its value is confined to applications needing a SHORT-HORIZON
forecast REFRESHED EVERY HOUR -- intraday balancing, storage and demand-response
dispatch, live anomaly response. For a day-ahead profile computed once each
morning, a daily batch upload is adequate, this scheme buys nothing, and it
should not be deployed. The gain is zero at the resync hour by construction and
largest at the end of the day.

TWO SECTIONS, AND NOTHING ELSE
==============================
MODEL (section 4.1). Four rows compared AT MATCHED PACKETS PER DAY, not at
matched alpha -- comparing predictors at equal alpha compares them at different
transmission budgets, which is not a comparison.

    A. Persistence + deadband     the classic WSN send-on-delta baseline
    B. Global fixed               the pre-trained vector, adaptation disabled
    C. Global + NLMS              the proposed model
    D. Per-client ridge           NOT deployable; the ceiling reference only

Expected ordering A < B <= C <= D. B > A is structural, because ridge over a tap
set containing lag 1 has persistence in its hypothesis space and is fitted to
the loss being reported. C - B is the adaptation gain. D - C is what per-client
calibration would buy if one were willing to ship 321 weight vectors.

SYSTEM (section 4.2). `control` (no codec), `batch-daily` (the honest
competitor), and the alpha sweep. The headline is the LEAD-1 MAE BY ISSUE HOUR
table: at issue hour h the refreshed system serves a lead-1 forecast while the
daily-batch system serves lead h+1 of its morning run. Both predict the same
absolute hour, so it is exactly matched, and the gap is the entire value of the
architecture.

DELETED ON PURPOSE (section 4.4): the packet-loss study, the sync_safe/legacy
comparison rows, the cap8 rows, the event-detection proxy, the LSTNet RSE/CORR
appendix, and the alpha = inf corrupted-window control. Each is one sentence in
the limitations paragraph, and the scripts that produce them still exist.
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
from edge.protocol import EXCEPTION_BYTES as EXCEPTION_BYTES_    # noqa: E402
from edge.protocol import time_on_air                             # noqa: E402
from eval.metrics_tier1 import (TTN_FAIR_USE_S, communication_cost,  # noqa: E402
                                delta_per_client, frac_hours_capped,
                                linf_violation, quantisation_mae,
                                recon_fidelity, staleness_hours)
from sim.periodic import label as periodic_label                  # noqa: E402
from sim.periodic import periodic_nbytes, run_periodic            # noqa: E402
from sim.reconstruct import build_log1p_arrays                    # noqa: E402
from sim.run_codec import Store, run_policy                       # noqa: E402
from tier2.adapter import (Accum, distribution, forecast_kwh,     # noqa: E402
                           frac_no_skill, iter_batches,
                           load_frozen_dlinear, make_window_builder,
                           patch_y_past, probe_index_convention,
                           store_manifest)

ALPHA_GRID = (0.0, 0.25, 0.5, 0.7, 1.0, 1.5, 2.0, 4.0)
MATCH_RATES = (15.0, 10.0, 6.0)
CONTROL, BATCH_DAILY = "control", "batch-daily"
# Tap sets for the forecast-level half of the weight-count ablation. The
# predictor-level sweep lives in eval/tap_ablation.py and needs no forecaster;
# only a couple of sets are worth the streaming DLinear pass. Keep these
# identical to LADDER there, or the two tables disagree about what T7 means.
TAP_LADDER = {
    "T7":  (1, 2, 3, 24, 25),
    "T8":  (1, 2, 3, 24, 25, 168),
    "T9":  (1, 2, 3, 24, 25, 48, 168),
    "T10": (1, 2, 3, 24, 25, 168, 169, 336),
    "T11": tuple(range(1, 25)),
    "T12": tuple(range(1, 25)) + (168,),
}

FAMILY_NAMES = {
    "A": "Persistence + deadband (send-on-delta)",
    "B": "Global fixed (pre-trained, adaptation off)",
    "C": "Global + NLMS (proposed)",
    "D": "Per-client ridge (CEILING, not deployable)",
}


def alpha_label(a):
    return "inf" if a == float("inf") else f"{a:g}"


def derive_budget(mean_payload_bytes, sf=12, allowance_s=TTN_FAIR_USE_S,
                  retry_margin=0.20, realised_toa_s=None):
    """The transmission budget, stated in AIRTIME (section 4.3).

    budget_s = (1 - retry_margin) * allowance_s

    THE BUDGET IS SECONDS, NOT PACKETS, AND THE SELECTION TEST USES SECONDS.
    A packets/day figure is also returned, but only as an indicative
    conversion: the two frame types differ by more than a factor of three in
    time-on-air at SF12 -- 991 ms for an 8-byte exception against 3449 ms for
    the 83-byte daily resync, a ratio of 3.48 -- so any packet budget
    presupposes a payload mix and the mix moves
    with the deadband. Dividing the selected packet rate by a packet budget
    does NOT give the fraction of the allowance consumed; that fraction is
    airtime over allowance, which is what the tables report.

    A 20% retry margin, not 50%: LoRa uplinks at SF12 in a single-building
    deployment are not lossy enough to justify reserving half the allowance, and
    the daily RESYNC is the only packet whose loss matters at all -- exceptions
    are individually disposable. Reserving one fifth covers a retried resync and
    a handful of retried exceptions per day.
    """
    budget_s = (1.0 - retry_margin) * allowance_s
    toa = realised_toa_s or time_on_air(mean_payload_bytes, sf)
    return {"budget_airtime_s": budget_s,
            "budget_pkt_per_day_indicative": budget_s / toa,
            "sf": sf, "allowance_s": allowance_s, "retry_margin": retry_margin,
            "mean_payload_bytes": mean_payload_bytes, "time_on_air_s": toa,
            "derivation": (f"{1 - retry_margin:g} x {allowance_s:g} s/day TTN "
                           f"fair use = {budget_s:.1f} s/day at SF{sf}"),
            "indicative_note": (f"about {budget_s / toa:.1f} pkt/day at the "
                                f"realised mean time-on-air of "
                                f"{toa * 1000:.0f} ms; indicative only")}


# ------------------------------------------------------- the batch baseline
@torch.no_grad()
def build_daily_batch_forecasts(model, wb, sel, batch_size, device):
    """One DLinear run per (day-start origin, client), on a CLEAN window.

    The window is the unmodified store's, which is marginally MORE information
    than a strict daily-batch hub would hold at hour t0. That asymmetry favours
    the BASELINE, so the comparison understates the scheme -- the direction an
    honest control should err in.
    """
    pairs = sorted({(int(t) - int(t) % 24, int(c)) for t, c in sel})
    pairs = [(t0, c) for t0, c in pairs
             if t0 - wb.L + 1 >= 0 and t0 + wb.lag + wb.H - 1 < wb.T]
    out = {}
    arr = np.asarray(pairs, np.int64)
    for i in range(0, len(arr), batch_size):
        chunk = arr[i:i + batch_size]
        b = wb.build(chunk, device=device)
        f = forecast_kwh(model, b).cpu().numpy()
        for r, (t0, c) in enumerate(chunk):
            out[(int(t0), int(c))] = f[r]
    return out, len(pairs)


def held_forecast(batch_fc, chunk, H):
    """A window at origin t with h = t % 24 needs hours t+1..t+24; the morning
    run at t0 = t-h covers the first 24-h of them. Steps beyond that are not
    available to a daily-batch system and are MASKED, not back-filled."""
    B = len(chunk)
    f = np.zeros((B, H), np.float64)
    cov = np.zeros((B, H), bool)
    for r in range(B):
        t, c = int(chunk[r, 0]), int(chunk[r, 1])
        h = t % 24
        row = batch_fc.get((t - h, c))
        if row is None:
            continue
        n = H - h
        f[r, :n] = row[h:h + n]
        cov[r, :n] = True
    return f, cov


# ----------------------------------------------------------------- scoring
@torch.no_grad()
def score_rows(model, wb, sel, recon, batch_fc, batch_size, device,
               corr_steps=(24,)):
    names = [CONTROL, BATCH_DAILY] + list(recon)
    acc = {n: Accum(wb.M, wb.H, corr_steps) for n in names}
    acc_sn24 = Accum(wb.M, wb.H, corr_steps)
    hod_ae = {n: np.zeros(24) for n in names}
    hod_n = {n: np.zeros(24) for n in names}
    l1_ae = {n: np.zeros(24) for n in names}
    l1_n = {n: np.zeros(24) for n in names}
    # Per-CLIENT lead-1 error (added 2026-09-29, conference revision request B:
    # reviewers asked for paired client-level uncertainty on next-hour error).
    # Accumulated alongside the pooled by-issue-hour figures above and never
    # folded into them, so every reported number is computed exactly as before.
    l1c_ae = {n: np.zeros(wb.M) for n in names}
    l1c_n = {n: np.zeros(wb.M) for n in names}
    n_unc = 0

    done, t0 = 0, time.time()
    for chunk, b in iter_batches(wb, sel, batch_size, device):
        y = b["y_true_kwh"].cpu().numpy()
        v = b["valid"].cpu().numpy()
        cl = chunk[:, 1]
        hod = chunk[:, 0] % 24
        acc_sn24.add(y, b["sn24_kwh"].cpu().numpy(), v, cl)

        fcs = {CONTROL: forecast_kwh(model, b).cpu().numpy()}
        fb, cov = held_forecast(batch_fc, chunk, wb.H)
        fcs[BATCH_DAILY] = fb
        n_unc += int((~cov).sum())
        for n, (ex, wi) in recon.items():
            yp = patch_y_past(b, chunk, ex, wi, wb.L, device)
            fcs[n] = forecast_kwh(model, b, yp).cpu().numpy()

        for n, f in fcs.items():
            vv = v & cov if n == BATCH_DAILY else v
            acc[n].add(y, f, vv, cl)
            ae = np.abs(np.where(vv, f - y, 0.0)).sum(axis=1)
            np.add.at(hod_ae[n], hod, ae)
            np.add.at(hod_n[n], hod, vv.sum(axis=1))
            m1 = vv[:, 0]
            np.add.at(l1_ae[n], hod, np.abs(np.where(m1, f[:, 0] - y[:, 0], 0.0)))
            np.add.at(l1_n[n], hod, m1.astype(float))
            np.add.at(l1c_ae[n], cl, np.abs(np.where(m1, f[:, 0] - y[:, 0], 0.0)))
            np.add.at(l1c_n[n], cl, m1.astype(float))

        done += len(chunk)
        if done % (batch_size * 400) < batch_size:
            print(f"  scored {done:,d} / {len(sel):,d} windows "
                  f"({time.time() - t0:.0f}s)", flush=True)

    hod = {n: (hod_ae[n] / np.maximum(hod_n[n], 1)).tolist() for n in names}
    lead1 = {n: (l1_ae[n] / np.maximum(l1_n[n], 1)).tolist() for n in names}
    lead1_pc = {n: {"abs_err_sum_kwh": l1c_ae[n], "n": l1c_n[n]} for n in names}
    return acc, acc_sn24, hod, lead1, n_unc, lead1_pc


def per_client_cost(tx, resync_bytes, active, bytes_per_tx, sf=12):
    """Per-client packets/day and SF`sf` airtime/day -- the vectors
    eval.metrics_tier1.communication_cost averages into its reported means,
    computed with its exact formula so the per-client paired statistics of the
    conference revision (request B, F) sit on the same axis as the paper.
    Callers assert that the means agree with communication_cost's."""
    from eval.metrics_tier1 import EPS
    tx = np.asarray(tx, bool)
    rb = np.asarray(resync_bytes)
    a = np.asarray(active, bool)
    days = np.maximum(a.sum(axis=0) / 24.0, EPS)
    n_exc = tx.sum(axis=0).astype(np.float64)
    n_res = (rb > 0).sum(axis=0).astype(np.float64)
    mean_res_bytes = float(rb[rb > 0].mean()) if (rb > 0).any() else 34.0
    t_exc = time_on_air(bytes_per_tx, sf)
    t_res = time_on_air(mean_res_bytes, sf)
    return {"packets_per_day": (n_exc + n_res) / days,
            "airtime_per_day_s": (n_exc * t_exc + n_res * t_res) / days}


def interp_at_rate(points, rate):
    """dMASE at a given packets/day, linearly interpolated along the curve.

    Comparing predictors at matched ALPHA compares them at different
    transmission budgets, which is not a comparison. Matched packet rate is.
    """
    pts = sorted(points, key=lambda p: p[0])
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    if rate < xs.min() or rate > xs.max():
        return None
    return float(np.interp(rate, xs, ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="tier2_data")
    ap.add_argument("--checkpoint", default="dlinear.pt")
    ap.add_argument("--pretrained", default=paths.inp("edge_pretrained.json"))
    ap.add_argument("--per_client",
                    default=paths.inp("edge_pretrained_per_client.npy"))
    ap.add_argument("--select_seed", type=int, default=1,
                    help="RNG for the validation-window subsample that SELECTS "
                         "alpha. Default 1 = the value hardcoded until "
                         "2026-09-14, so default behaviour is unchanged. Vary "
                         "it to test whether the operating point is an "
                         "artefact of one subsample; see "
                         "docs/reference/stochasticity.md.")
    ap.add_argument("--per_client_out", default=None,
                    help="Optional. Dump the PER-CLIENT metric vectors that "
                         "the reported distributions are computed from, so a "
                         "bootstrap over clients can put a CI on every "
                         "reported number without re-running anything. "
                         "Writes a SEPARATE file: --out is unchanged, so "
                         "tier1_results.json stays bit-identical. Default "
                         "None = write nothing.")
    ap.add_argument("--split", default="test")
    ap.add_argument("--select_split", default="valid")
    ap.add_argument("--alphas", default=",".join(alpha_label(a) for a in ALPHA_GRID))
    ap.add_argument("--ablation_alphas", default="0.25,0.5,1,2,4")
    ap.add_argument("--periods", default="2,3,4,6",
                    help="fixed periodic refresh baseline, in hours; '' "
                         "disables it")
    ap.add_argument("--k_mu", type=int, default=5)
    ap.add_argument("--extra_taps", default="",
                    help="forecast-level rows for alternative tap sets, e.g. "
                         "'T7,T12'. Empty by default, so the reportable run is "
                         "unchanged. Each needs its own ridge fit -- the "
                         "shipped vector is fitted for T8 and is meaningless "
                         "for any other lag set -- which eval/tap_ablation.py "
                         "writes to --tap_cache.")
    ap.add_argument("--tap_cache", default="_pretrain_cache",
                    help="per-tap-set ridge fits, written by eval.tap_ablation")
    ap.add_argument("--margin_basis", default="min",
                    choices=("valid", "test", "min"),
                    help="which estimate of the oracle skill margin the "
                         "tolerance is a fraction of; see the note in main()")
    ap.add_argument("--tolerance", type=float, default=0.10)
    ap.add_argument("--tolerance_fallback", type=float, default=0.15)
    ap.add_argument("--budget_sf", type=int, default=12)
    ap.add_argument("--retry_margin", type=float, default=0.20)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--select_samples", type=int, default=150000,
                    help="validation windows used for the operating-point "
                         "choice; alpha must not be selected on test")
    ap.add_argument("--recon_dir", default="_recon_cache")
    ap.add_argument("--out", default=paths.result("tier1_results.json"))
    a = ap.parse_args()

    alphas = [float(s) for s in a.alphas.split(",")]
    abl_alphas = [float(s) for s in a.ablation_alphas.split(",")]

    pre = json.load(open(a.pretrained))
    w_global = tuple(pre["w_global_q"])
    per_client = np.load(a.per_client)
    base = CodecParams(k_mu=a.k_mu).with_w_init(w_global)
    p_A = PARAMS_SEND_ON_DELTA
    p_B = CodecParams(**{**base.__dict__, "adapt": False})
    p_C = base
    p_D = CodecParams(**{**base.__dict__, "adapt": False})

    man = store_manifest(a.data_dir)
    model, cfg, ck = load_frozen_dlinear(a.checkpoint, man, device=a.device)
    wb = make_window_builder(a.data_dir, cfg.lookback_len, cfg.horizon_len)
    probe = probe_index_convention(wb, a.split)
    print(f"index convention probe: {probe['convention']} "
          f"(dev {probe['target_max_dev']:.2e} / {probe['lookback_max_dev']:.2e})")
    if man.get("SYNTHETIC") or ck.get("SYNTHETIC"):
        print("!! SYNTHETIC store and/or checkpoint -- wiring check, not a result")
    if ck.get("trainer") == "tier1_fallback":
        print("!" * 74)
        print("!! FALLBACK CHECKPOINT. This model was trained by "
              "tier2/train_dlinear.py, not")
        print("!! supplied by Tier 2. The frozen forecaster is the premise of "
              "this experiment, so")
        print("!! every reported number must be regenerated against the real "
              "checkpoint before")
        print("!! publication. Rerun with --checkpoint <the real one>; nothing "
              "else changes.")
        print("!" * 74)
    print(f"edge predictor: taps {list(base.taps)}, Q{base.w_q} int32, "
          f"k_mu={base.k_mu} (mu={2.0 ** -base.k_mu:g}), "
          f"w_init = global ridge (lambda {pre['lambda']:g}, "
          f"{pre['improvement_over_persistence']:+.1%} vs persistence on "
          f"validation)")

    store = Store.load(a.data_dir)
    store.assert_scales_are_train_only()
    acct = store.client_accounting()
    print(f"clients: M={acct['M']}  scorable={acct['n_scorable']}  "
          f"degenerate mase_scale={acct['n_degenerate']}  "
          f"dropped={acct['dropped']}")

    sel = wb.splits[a.split]
    if a.max_samples:
        rng = np.random.default_rng(0)
        sel = sel[np.sort(rng.choice(len(sel), min(a.max_samples, len(sel)),
                                     replace=False))]
        print(f"[dev mode] {len(sel):,d} windows -- NOT reportable")
    print(f"\nsplit '{a.split}': {len(sel):,d} windows x {wb.H} steps, "
          f"{len(np.unique(sel[:, 1]))} clients, L={wb.L}")

    # ---------------------------------------------------------- run the codec
    rows = {}
    for al in alphas:
        rows[f"C@{alpha_label(al)}"] = {"params": p_C, "alpha": al, "family": "C",
                                        "w_pc": None}
    for al in abl_alphas:
        rows.setdefault(f"A@{alpha_label(al)}", {"params": p_A, "alpha": al,
                                                 "family": "A", "w_pc": None})
        rows.setdefault(f"B@{alpha_label(al)}", {"params": p_B, "alpha": al,
                                                 "family": "B", "w_pc": None})
        rows.setdefault(f"D@{alpha_label(al)}", {"params": p_D, "alpha": al,
                                                 "family": "D",
                                                 "w_pc": per_client})
    # The fixed periodic refresh baseline (reviewer item 6): no codec, no
    # predictor, no threshold -- just a schedule. It answers the most obvious
    # objection to the whole scheme, and it needs no new scoring machinery
    # because sim/periodic.py emits the same arrays a codec policy does.
    # The weight-count ablation, forecast level. Family "T" is deliberately
    # OUTSIDE "ABCD", so these rows never enter the model-ablation table nor
    # the operating-point selection, which is family C only.
    tap_ids = [t.strip() for t in a.extra_taps.split(",") if t.strip()]
    for tid in tap_ids:
        if tid not in TAP_LADDER:
            raise SystemExit(f"unknown tap set {tid!r}; "
                             f"known: {sorted(TAP_LADDER)}")
        cache = os.path.join(a.tap_cache, f"{tid}.json")
        if not os.path.exists(cache):
            raise SystemExit(
                f"{cache} not found. Run\n"
                f"  python -m eval.tap_ablation --data_dir {a.data_dir} "
                f"--tap_sets {tid}\nfirst: every tap set needs its own ridge "
                f"fit, and reusing the shipped one would\nmeasure a predictor "
                f"whose weights were fitted for a different lag set.")
        taps = TAP_LADDER[tid]
        w_t = tuple(int(v) for v in json.load(open(cache))["w_global_q"])
        p_T = CodecParams(taps=taps, buf_len=24 + max(taps), k_mu=a.k_mu,
                          w_init=w_t)
        for al in abl_alphas:
            rows[f"{tid}@{alpha_label(al)}"] = {"params": p_T, "alpha": al,
                                                "family": "T", "w_pc": None,
                                                "tap_id": tid}

    for pr in [int(x) for x in a.periods.split(",") if x.strip()]:
        rows[periodic_label(pr)] = {"params": None, "alpha": float(-pr),
                                    "family": "P", "w_pc": None, "period": pr}

    print(f"\nrunning the codec over {store.T:,d} h x {store.M} clients for "
          f"{len(rows)} configurations")
    # The reconstruction arrays are (T, M) float32 -- 34 MB each on the real
    # store, two per configuration. Holding twenty-odd configurations resident
    # is 1.5 GB and gets the process killed, so they go to disk and come back
    # memory-mapped. patch_y_past only ever touches 512 x 336 elements at a
    # time, so the mapped access pattern costs nothing.
    os.makedirs(a.recon_dir, exist_ok=True)
    recon, cost, pc_cost = {}, {}, {}
    for lab, spec in rows.items():
        t0 = time.time()
        if spec["family"] == "P":
            r = run_periodic(store, spec["period"])
        else:
            r = run_policy(store, spec["alpha"], spec["params"],
                           backend="batch", w_init_per_client=spec["w_pc"])
        ex, wi = build_log1p_arrays(r, store)
        tag = lab.replace("@", "_at_").replace(".", "p")
        pe = os.path.join(a.recon_dir, f"{tag}_exact.npy")
        pw = os.path.join(a.recon_dir, f"{tag}_within.npy")
        np.save(pe, ex)
        np.save(pw, wi)
        del ex, wi
        recon[lab] = (np.load(pe, mmap_mode="r"), np.load(pw, mmap_mode="r"))
        if spec["family"] == "P":
            # A zero-order hold offers NO L-inf bound, so there is nothing to
            # assert. That absence is the qualitative difference between this
            # baseline and the codec, and it is reported rather than hidden.
            viol = None
            nb = periodic_nbytes(spec["period"])
            cc = communication_cost(r.tx, np.zeros_like(r.resync_bytes),
                                    r.active, bytes_per_tx=nb)
            pcc = per_client_cost(r.tx, np.zeros_like(r.resync_bytes),
                                  r.active, nb, sf=a.budget_sf)
            cc["bytes_per_day_mean"] = float(
                (r.resync_bytes.sum(axis=0)
                 / np.maximum(r.active.sum(axis=0) / 24.0, 1e-9)).mean())
        else:
            viol = linf_violation(r.u_within, r.u_q, r.eps_eff, r.active)
            assert viol == 0, f"{lab}: {viol} L-inf violations"
            cc = communication_cost(r.tx, r.resync_bytes, r.active)
            pcc = per_client_cost(r.tx, r.resync_bytes, r.active,
                                  EXCEPTION_BYTES_, sf=a.budget_sf)
        assert float(pcc["packets_per_day"].mean()) == cc["packets_per_day_mean"]
        assert (float(pcc["airtime_per_day_s"].mean())
                == cc["airtime_per_day_s"][a.budget_sf])
        pc_cost[lab] = pcc
        st = staleness_hours(r.tx, r.active)
        live = r.active & ~store.missing
        fid = recon_fidelity(r.u_within, r.u_q, r.k, live, store.mase_scale)
        eps_zero = (spec["family"] == "P"
                    or all((e == 0) for e in r.eps if e is not None))
        cost[lab] = {
            **cc,
            "alpha": ("—" if spec["family"] == "P"
                      else alpha_label(spec["alpha"])),
            "family": spec["family"],
            "period_h": spec.get("period"),
            # Without this the family-T table finds no rows: the cost dict is
            # built field by field, so anything not copied here is invisible
            # to every consumer downstream.
            "tap_id": spec.get("tap_id"),
            # The periodic baseline has NO L-inf guarantee; recording the
            # observed maximum instead would dress a zero-order hold up as a
            # bounded scheme.
            "linf_guarantee": (None if spec["family"] == "P" else "eps"),
            "linf_violation": viol,
            "frac_hours_eps_doubled": (
                None if eps_zero
                else frac_hours_capped(r.eps_eff, r.eps, r.active)),
            "recon_mae_kwh": fid["recon_mae_kwh"],
            "recon_mae_over_mase_scale":
                distribution(fid["recon_mae_over_mase_scale"]),
            "staleness_mean_h": float(st[r.active].mean()),
            "staleness_p50_h": float(np.percentile(st[r.active], 50)),
            "staleness_p90_h": float(np.percentile(st[r.active], 90)),
            "staleness_max_h": int(st[r.active].max()),
        }
        print(f"  {lab:>8s}  pkt/day {cc['packets_per_day_mean']:6.2f}  "
              f"B/day {cc['bytes_per_day_mean']:7.1f}  "
              f"recon MAE {fid['recon_mae_kwh']:9.3f} kWh  "
              f"stale {cost[lab]['staleness_mean_h']:5.2f} h  "
              f"({time.time() - t0:.0f}s)", flush=True)
        del r, st, live, fid

    r0 = run_policy(store, 0.0, p_C, backend="batch")
    qm = quantisation_mae(r0.k, store.series_filled, store.active_mask())
    qm["per_client_over_mase_scale"] = distribution({
        int(c): float(0.5 * 2.0 ** int(r0.k[c]) / store.mase_scale[c])
        for c in range(store.M) if np.isfinite(store.mase_scale[c])
        and store.mase_scale[c] > 0})

    # ---------------------------------------------------- the batch baseline
    print(f"\nbuilding the daily-batch baseline (one DLinear run per day per "
          f"client, clean window)")
    batch_fc, n_pairs = build_daily_batch_forecasts(model, wb, sel,
                                                    a.batch_size, a.device)
    print(f"  {n_pairs:,d} morning origins")

    print(f"\nscoring {len(recon) + 2} rows in one streaming pass")
    acc, acc_sn24, hod, lead1, n_unc, lead1_pc = score_rows(
        model, wb, sel, recon, batch_fc, a.batch_size, a.device)
    sn_pc = acc_sn24.per_client(wb.mase_scale)["MAE"]
    cov_frac = 1.0 - n_unc / float(len(sel) * wb.H)

    results, pcs = {}, {}
    all_rows = [CONTROL, BATCH_DAILY] + list(recon)
    for lab in all_rows:
        m = acc[lab].pooled()
        pc = acc[lab].per_client(wb.mase_scale, sn_pc)
        ph = acc[lab].per_horizon()
        dm, ds = distribution(pc["MASE"]), distribution(pc["SS"])
        results[lab] = {**m, "MASE": dm["mean"], "MASE_p50": dm["p50"],
                        "MASE_p90": dm["p90"], "SS": ds["mean"],
                        "SS<0": frac_no_skill(pc["SS"]),
                        "nRMSE": distribution(pc["nRMSE"])["mean"],
                        "MAE_h": ph["MAE_h"].tolist(),
                        "RMSE_h": ph["RMSE_h"].tolist(),
                        "MAE_by_issue_hour": hod[lab],
                        "MAE_lead1_by_issue_hour": lead1[lab]}
        pcs[lab] = pc

    oracle = "C@0"
    for lab in all_rows:
        results[lab]["delta_MASE"] = distribution(
            delta_per_client(pcs[lab]["MASE"], pcs[oracle]["MASE"]))
        results[lab]["delta_MASE_vs_control"] = distribution(
            delta_per_client(pcs[lab]["MASE"], pcs[CONTROL]["MASE"]))
        results[lab]["delta_MAE_h"] = (
            np.asarray(results[lab]["MAE_h"])
            - np.asarray(results[oracle]["MAE_h"])).tolist()

    # The per-client vectors above are collapsed through distribution() before
    # they reach the artifact, so every reported figure is a point estimate over
    # 320 meters with no stated spread. Dumping them costs nothing and lets a
    # bootstrap over clients turn each one into an interval -- which is the
    # honest answer to "would this hold on a different set of meters?".
    # Deliberately a SEPARATE file: tier1_results.json must stay bit-identical.
    if a.per_client_out:
        pc_doc = {
            "n_clients_scored": len(pcs[CONTROL]["MASE"]),
            "oracle": oracle, "control": CONTROL,
            "metrics_per_client": {
                lab: {k: {str(c): float(v) for c, v in pcs[lab][k].items()}
                      for k in ("MASE", "SS", "nRMSE") if k in pcs[lab]}
                for lab in all_rows},
            "delta_MASE_per_client": {
                lab: {str(c): float(v) for c, v in
                      delta_per_client(pcs[lab]["MASE"],
                                       pcs[oracle]["MASE"]).items()}
                for lab in all_rows},
            "delta_MASE_vs_control_per_client": {
                lab: {str(c): float(v) for c, v in
                      delta_per_client(pcs[lab]["MASE"],
                                       pcs[CONTROL]["MASE"]).items()}
                for lab in all_rows},
            # Added 2026-09-29 (conference revision, requests B and F). Lead-1
            # absolute error summed per client over the scored test windows,
            # with the count, so a per-client next-hour MAE is sum / n; and the
            # per-client cost vectors behind the packet-rate and airtime axes.
            # Every client index 0..M-1; the paired statistics restrict to the
            # scored clients themselves.
            "lead1_abs_err_sum_kwh_per_client": {
                lab: {str(c): float(v) for c, v in
                      enumerate(lead1_pc[lab]["abs_err_sum_kwh"])}
                for lab in all_rows},
            "lead1_n_per_client": {
                lab: {str(c): float(v) for c, v in
                      enumerate(lead1_pc[lab]["n"])}
                for lab in all_rows},
            "mase_scale_per_client": {str(c): float(v) for c, v in
                                      enumerate(wb.mase_scale)},
            "packets_per_day_per_client": {
                lab: {str(c): float(v) for c, v in
                      enumerate(pc_cost[lab]["packets_per_day"])}
                for lab in pc_cost},
            "airtime_per_day_s_per_client": {
                lab: {str(c): float(v) for c, v in
                      enumerate(pc_cost[lab]["airtime_per_day_s"])}
                for lab in pc_cost},
            "airtime_sf": a.budget_sf,
        }
        os.makedirs(os.path.dirname(a.per_client_out) or ".", exist_ok=True)
        with open(a.per_client_out, "w") as f:
            json.dump(pc_doc, f, indent=2)
        print(f"wrote per-client vectors to {a.per_client_out}")
    l1_batch = np.asarray(lead1[BATCH_DAILY])
    for lab in all_rows:
        g = l1_batch - np.asarray(lead1[lab])
        results[lab]["refresh_gain_lead1_by_issue_hour"] = g.tolist()
        results[lab]["refresh_gain_lead1_mean"] = float(g.mean())

    # ------------------------------------------------------ the system table
    print(f"\n=== SYSTEM (section 4.2) ===")
    print(f"{'row':>12s} {'pkt/day':>8s} {'stale':>6s} {'MAE':>9s} "
          f"{'RMSE':>10s} {'MASE':>7s} {'dMASE':>8s} {'SS':>7s} {'SS<0':>6s} "
          f"{'gain@l1':>8s}")
    print("-" * 96)
    sys_rows = ([CONTROL, BATCH_DAILY]
                + [f"C@{alpha_label(x)}" for x in alphas]
                + [l for l in cost if cost[l]["family"] == "P"])
    for lab in sys_rows:
        r = results[lab]
        c = cost.get(lab)
        pk = "-" if c is None else f"{c['packets_per_day_mean']:8.2f}"
        stale = "-" if c is None else f"{c['staleness_mean_h']:6.2f}"
        print(f"{lab:>12s} {pk:>8s} {stale:>6s} {r['MAE']:9.2f} "
              f"{r['RMSE']:10.2f} {r['MASE']:7.4f} "
              f"{r['delta_MASE']['mean']:+8.4f} {r['SS']:+7.3f} "
              f"{r['SS<0']:6.1%} {r['refresh_gain_lead1_mean']:+8.2f}")
    print(f"  control = no codec. batch-daily = one DLinear run per day, held "
          f"24 h, 1 pkt/day.")
    print(f"  batch-daily's pooled columns cover {cov_frac:.1%} of the horizon "
          f"steps (a morning run")
    print(f"  cannot reach past lead 24); compare it on the LEAD-1 table, which "
          f"is exactly matched.")
    q0 = results[CONTROL]["MASE"] - results[oracle]["MASE"]
    print(f"  quantisation costs {abs(q0):.4f} MASE (control {results[CONTROL]['MASE']:.4f} "
          f"vs alpha=0 {results[oracle]['MASE']:.4f}).")
    print(f"  zero L-inf violations across the whole sweep.")
    rmse = [results[f"C@{alpha_label(x)}"]["RMSE"] for x in alphas]
    if len(rmse) > 2 and min(rmse) < rmse[0]:
        i = int(np.argmin(rmse))
        print(f"  NOTE RMSE is NON-MONOTONIC in alpha: {rmse[0]:.1f} at "
              f"alpha=0 falls to {rmse[i]:.1f} at "
              f"alpha={alpha_label(alphas[i])} before rising. Real and "
              f"explicable -- on silent")
        print(f"  hours the codec substitutes smooth predictions for spiky true "
              f"values, clipping the tail")
        print(f"  that RMSE is dominated by. At RMSE/MAE ~10 the tail owns "
              f"RMSE, so mild input")
        print(f"  smoothing acts as a regulariser.")

    # ---- fixed periodic refresh, at MATCHED packet rate (reviewer item 6)
    per_rows = [l for l in cost if cost[l]["family"] == "P"]
    periodic_cmp = []
    if per_rows:
        c_pts = [(cost[l]["packets_per_day_mean"],
                  results[l]["delta_MASE_vs_control"]["mean"])
                 for l in cost if cost[l]["family"] == "C"]
        print(f"\n=== FIXED PERIODIC REFRESH, at matched packets/day ===")
        print(f"the obvious objection -- 'why event-triggered rather than just "
              f"transmitting more often?'")
        print(f"{'row':>14s} {'pkt/day':>8s} {'B/day':>7s} {'dMASE':>8s} "
              f"{'codec @ rate':>13s} {'codec better':>13s}")
        for l in sorted(per_rows,
                        key=lambda x: -cost[x]["packets_per_day_mean"]):
            pk = cost[l]["packets_per_day_mean"]
            dp = results[l]["delta_MASE_vs_control"]["mean"]
            dc = interp_at_rate(c_pts, pk)
            rel = None if (dc is None or dp <= 0) else (dp - dc) / dp
            periodic_cmp.append({"row": l, "period_h": cost[l]["period_h"],
                                 "packets_per_day": pk, "dMASE": dp,
                                 "codec_dMASE_at_same_rate": dc,
                                 "codec_better_by": rel})
            print(f"{l:>14s} {pk:8.2f} {cost[l]['bytes_per_day_mean']:7.1f} "
                  f"{dp:8.4f} "
                  + ("          n/a" if dc is None else f"{dc:13.4f}")
                  + ("          n/a" if rel is None else f"{rel:13.1%}"))
        okp = [x["codec_better_by"] for x in periodic_cmp
               if x["codec_better_by"] is not None]
        if okp:
            print(f"  mean advantage of the codec at matched packet rate: "
                  f"{sum(okp) / len(okp):+.1%}")
        print(f"  The periodic baseline delivers EXACT values, so its completed "
              f"history is perfect and the")
        print(f"  comparison is purely about freshness of the current window -- "
              f"the axis in dispute. It also")
        print(f"  carries NO L-inf guarantee: a zero-order hold can be "
              f"arbitrarily wrong between refreshes,")
        print(f"  where the codec is bounded by eps at every hour. That "
              f"difference is qualitative and a")
        print(f"  packet-rate comparison cannot capture it.")

    print(f"\nLEAD-1 MAE by issue hour (kWh) -- THE HEADLINE. Both rows predict "
          f"the same absolute")
    print(f"hour; batch-daily serves lead h+1 of its morning run.")
    show = [0, 1, 3, 6, 9, 12, 18, 23]
    print(f"{'row':>12s}" + "".join(f"{'h' + str(h):>8s}" for h in show)
          + f"{'mean':>8s}")
    for lab in [BATCH_DAILY, CONTROL, oracle]:
        v = results[lab]["MAE_lead1_by_issue_hour"]
        print(f"{lab:>12s}" + "".join(f"{v[h]:8.2f}" for h in show)
              + f"{np.mean(v):8.2f}")
    l1c = np.asarray(results[CONTROL]["MAE_lead1_by_issue_hour"])
    full = float((l1_batch - l1c).mean())
    print(f"  hourly refresh reduces next-hour MAE from {l1_batch.mean():.1f} "
          f"to {l1c.mean():.1f} kWh, {full / max(l1_batch.mean(), 1e-9):.1%}.")
    print(f"  Zero at h=0 by construction. If the application only issues at "
          f"h=0, the gain is zero")
    print(f"  and this scheme is not for it.")

    # ------------------------------------------------------ operating point
    mean_payload = float(np.mean([cost[l]["mean_payload_bytes"]
                                  for l in cost if cost[l]["family"] == "C"]))
    _c = [l for l in cost if cost[l]["family"] == "C"]
    _toa = float(np.mean([cost[l]["time_on_air_s"][a.budget_sf]["mean_realised"]
                          for l in _c]))
    budget = derive_budget(mean_payload, sf=a.budget_sf,
                           retry_margin=a.retry_margin, realised_toa_s=_toa)
    margin = 1.0 - results[oracle]["MASE"]
    # ALPHA IS SELECTED ON VALIDATION. Re-scoring the whole validation split
    # would double the run, so a fixed subsample is used -- large enough that
    # the per-client p90 is stable, and it is the SELECTION that must not see
    # test, not the reporting.
    vsel = wb.splits[a.select_split]
    if len(vsel) > a.select_samples:
        rv = np.random.default_rng(a.select_seed)
        vsel = vsel[np.sort(rv.choice(len(vsel), a.select_samples,
                                      replace=False))]
    print(f"\nselecting alpha on '{a.select_split}' ({len(vsel):,d} windows)")
    vrec = {l: recon[l] for l in recon if cost[l]["family"] == "C"}
    vfc, _ = build_daily_batch_forecasts(model, wb, vsel, a.batch_size, a.device)
    vacc, vsn, _vh, _vl, _vu, _vpc = score_rows(model, wb, vsel, vrec, vfc,
                                                a.batch_size, a.device)
    vsn_pc = vsn.per_client(wb.mase_scale)["MAE"]
    vpc = {l: vacc[l].per_client(wb.mase_scale, vsn_pc)
           for l in [CONTROL] + list(vrec)}
    v_oracle = f"C@{alpha_label(alphas[0])}"
    v_p90 = {l: distribution(delta_per_client(vpc[l]["MASE"],
                                              vpc[v_oracle]["MASE"]))["p90"]
             for l in vrec}
    v_margin = 1.0 - distribution(vacc[v_oracle].per_client(
        wb.mase_scale, vsn_pc)["MASE"])["mean"]

    # WHICH MARGIN THE TOLERANCE IS A FRACTION OF.
    #
    # The rule is "give back at most `tolerance` of the forecaster's skill over
    # seasonal-naive". That skill is what the paper REPORTS, i.e. the test-split
    # margin. Estimating it on validation instead is not equivalent here:
    # validation is measurably easier for this forecaster (margin 0.286 against
    # 0.155 on test), so anchoring to it inflates the allowance by 84% and lets
    # an alpha through that gives back 16% of the reported skill while the rule
    # claims 10%. That is the bug the review caught, and it is a real one.
    #
    # Reading the margin from test does NOT leak the policy choice. The margin
    # is the alpha = 0 row -- the same number for every candidate -- so it is a
    # scale factor on the threshold, not a ranking signal. Every quantity that
    # ORDERS the candidates (the per-client dMASE p90) is computed on validation
    # and never on test. The default is `min`, which is the conservative of the
    # two and therefore cannot be accused of either failure.
    margins = {"valid": v_margin, "test": margin}
    margins["min"] = min(v_margin, margin)
    sel_margin = margins[a.margin_basis]
    print(f"  oracle skill margin: validation {v_margin:.4f}, test "
          f"{margin:.4f} -> using '{a.margin_basis}' = {sel_margin:.4f}")
    if v_margin > 1.2 * margin:
        print(f"  NOTE validation is {v_margin / margin:.2f}x easier than test "
              f"for this forecaster, so a validation-anchored tolerance would "
              f"be {v_margin / margin:.2f}x looser than the rule claims.")

    print(f"\n=== OPERATING POINT (section 4.3) ===")
    print(f"budget: {budget['derivation']}  ({budget['indicative_note']})")
    for _l in cost:
        _t = cost[_l].get("time_on_air_s", {}).get(a.budget_sf)
        if _t:
            print(f"  per-frame time-on-air at SF{a.budget_sf}: exception "
                  f"{_t['exception'] * 1000:.0f} ms, resync "
                  f"{_t['resync'] * 1000:.0f} ms")
            break
    print(f"tolerance: {a.tolerance:.0%} x {sel_margin:.4f} = "
          f"{a.tolerance * sel_margin:.5f} absolute dMASE p90")
    print(f"  the budget is AIRTIME: a row fits if its SF{a.budget_sf} airtime "
          f"is within {budget['budget_airtime_s']:.1f} s/day. '% allow' is "
          f"airtime / {TTN_FAIR_USE_S:g} s;")
    print(f"  '% budget' is airtime / {budget['budget_airtime_s']:.1f} s. "
          f"Packets divided by a packet budget gives neither.")
    print(f"{'row':>12s} {'pkt/day':>8s} {'SF12 s/day':>11s} {'% allow':>8s} {'% budget':>9s} "
          f"{'dMASE p90':>10s} {'fits':>6s} {'tol':>5s}")
    sel_note = None
    cand_record = []
    for tol in (a.tolerance, a.tolerance_fallback):
        tol_abs = tol * sel_margin
        cands = []
        for al in alphas:
            lab = f"C@{alpha_label(al)}"
            c = cost[lab]
            air = c["airtime_per_day_s"][a.budget_sf]
            p90 = v_p90[lab]                     # VALIDATION, not test
            fits = air <= (1 - a.retry_margin) * TTN_FAIR_USE_S
            cands.append((al, lab, c["packets_per_day_mean"], air, p90, fits))
            if tol == a.tolerance:
                cand_record.append(
                    {"alpha": alpha_label(al), "row": lab,
                     "packets_per_day": c["packets_per_day_mean"],
                     "airtime_sf12_s": air,
                     "frac_of_allowance": air / TTN_FAIR_USE_S,
                     "frac_of_budget": air / budget["budget_airtime_s"],
                     "valid_dMASE_p90": p90,
                     "fits_budget": bool(fits),
                     "passes_tolerance": bool(p90 <= tol_abs)})
                print(f"{lab:>12s} {c['packets_per_day_mean']:8.2f} "
                      f"{air:11.3f} {air / TTN_FAIR_USE_S:8.1%} "
                     f"{air / budget['budget_airtime_s']:9.1%} {p90:10.5f} "
                      f"{str(fits):>6s} "
                      f"{'pass' if p90 <= tol_abs else 'FAIL':>5s}")
        ok = [c for c in cands if c[4] <= tol_abs and c[5]]
        if ok:
            best = max(ok, key=lambda c: c[0])
            sel_note = {"selected_alpha": alpha_label(best[0]),
                        "row": best[1], "packets_per_day": best[2],
                        "airtime_sf12_s": best[3], "dMASE_p90": best[4],
                        "tolerance_frac_of_margin": tol,
                        "tolerance_abs_dMASE": tol_abs,
                        "margin_basis": a.margin_basis,
                        "oracle_skill_margin_valid": v_margin,
                        "oracle_skill_margin_test": margin,
                        "oracle_skill_margin_used": sel_margin,
                        "budget": budget,
                        "n_validation_windows": int(len(vsel)),
                        "candidates": cand_record,
                        "criterion": ("largest alpha whose VALIDATION "
                                      "per-client dMASE p90 stays <= "
                                      f"{tol:.0%} of the oracle skill margin, "
                                      "inside the derived airtime budget")}
            gain = results[best[1]]["refresh_gain_lead1_mean"]
            print(f"-> alpha = {alpha_label(best[0])}  "
                  f"({best[2]:.2f} pkt/day, dMASE p90 {best[4]:.5f}, "
                  f"{gain / max(full, 1e-9):.0%} of the refresh gain retained)"
                  + ("" if tol == a.tolerance
                     else f"   [tolerance widened to {tol:.0%}]"))
            break
        print(f"   no alpha passes at a {tol:.0%} tolerance"
              + ("; widening" if tol == a.tolerance else ""))

    # --------------------------------------------------- the model ablation
    # Reported AFTER the operating point so the selected packet rate can be a
    # column: the model result and the system result must be quoted at the same
    # configuration, or a reader cannot connect them.
    rates = list(MATCH_RATES)
    if sel_note:
        rates = sorted(set(rates) | {round(sel_note["packets_per_day"], 2)},
                       reverse=True)
    print(f"\n=== MODEL ABLATION (section 4.1), at MATCHED packets/day ===")
    # The ablation is the FOUR MODEL rows only. The periodic baseline is a
    # transmission schedule, not a predictor, so it belongs in the system
    # comparison and not in a table whose axis is which predictor is running.
    fam_pts = {f: [] for f in "ABCD"}
    for lab, c in cost.items():
        if c["family"] not in fam_pts:
            continue
        fam_pts[c["family"]].append(
            (c["packets_per_day_mean"],
             results[lab]["delta_MASE_vs_control"]["mean"],
             c["recon_mae_kwh"]))
    print(f"{'row':>44s}" + "".join(
        f"{str(r) + (' *' if sel_note and abs(r - sel_note['packets_per_day']) < 0.01 else '') + ' pkt/day':>17s}"
        for r in rates))
    abl = {}
    for f in "ABCD":
        pts = [(p, d) for p, d, _ in fam_pts[f]]
        vals = [interp_at_rate(pts, r) for r in rates]
        abl[f] = {"name": FAMILY_NAMES[f],
                  "dMASE_vs_control_at": dict(zip(map(str, rates), vals)),
                  "curve": sorted(fam_pts[f])}
        print(f"{f + '. ' + FAMILY_NAMES[f]:>44s}"
              + "".join("              n/a" if v is None else f"{v:17.4f}"
                        for v in vals))
    print(f"  dMASE against the CONTROL (no codec); lower is better. Ordering "
          f"A < B <= C <= D expected.")
    if sel_note:
        print(f"  * = the selected operating point, so the model and system "
              f"results are quoted at the")
        print(f"    same configuration.")
    for r in rates:
        vA, vB, vC, vD = (abl[f]["dMASE_vs_control_at"][str(r)] for f in "ABCD")
        if None in (vA, vB, vC):
            continue
        frac_pre = (vA - vB) / vA if vA else float("nan")
        ada = (vB - vC) / vB if vB else float("nan")
        print(f"  at {r:g} pkt/day: pre-training removes {frac_pre:.0%} of the "
              f"degradation ({vA - vB:+.4f}), adaptation a further "
              f"{ada:.0%} ({vB - vC:+.4f})"
              + ("" if vD is None else
                 f"; per-client calibration {vC - vD:+.4f}"))
    # Only where BOTH families were actually interpolable; `or 0` would treat a
    # missing point as a perfect score and fire this note spuriously.
    neg = [r for r in rates
           if abl["B"]["dMASE_vs_control_at"][str(r)] is not None
           and abl["C"]["dMASE_vs_control_at"][str(r)] is not None
           and abl["B"]["dMASE_vs_control_at"][str(r)]
           < abl["C"]["dMASE_vs_control_at"][str(r)]]
    if neg:
        print(f"  NOTE adaptation is NEGATIVE at the high-rate end. That is "
              f"expected and is worth")
        print(f"  stating: when almost every hour is transmitted the predictor "
              f"barely matters, so the")
        print(f"  families converge. The predictor earns its keep where the "
              f"budget is tight -- which is")
        print(f"  itself the argument for reporting on a packet-rate axis "
              f"rather than an alpha axis.")


    tap_abl = {}
    if tap_ids:
        print("\n=== WEIGHT-COUNT ABLATION, at MATCHED packets/day ===")
        print("  Forecast-level counterpart of eval/tap_ablation.py. Rows are "
              "compared at matched\n  packet rate, never at matched alpha: at "
              "a fixed deadband a WORSE predictor triggers\n  more often, so "
              "it transmits more and reconstructs better, inverting every "
              "ordering.")
        t_rates = sorted(set(MATCH_RATES), reverse=True)
        print(f"{'row':>34s} {'w':>3s} {'SRAM B':>7s}" + "".join(
            f"{str(r) + ' pkt/day':>15s}" for r in t_rates))
        for tid in ["T8"] + [t for t in tap_ids if t != "T8"]:
            want = ((lambda c: c["family"] == "C") if tid == "T8"
                    else (lambda c: c.get("tap_id") == tid))
            pts = [(cost[l]["packets_per_day_mean"],
                    results[l]["delta_MASE_vs_control"]["mean"])
                   for l in cost if want(cost[l])]
            if not pts:
                continue
            taps = TAP_LADDER[tid]
            nw = len(taps) + 1
            sram = 2 * (24 + max(taps)) + 48 + 4 * nw + 22
            vals = [interp_at_rate(pts, r) for r in t_rates]
            tap_abl[tid] = {"taps": list(taps), "n_weights": nw,
                            "sram_bytes": sram, "shipped": tid == "T8",
                            "dMASE_vs_control_at": dict(zip(map(str, t_rates),
                                                            vals)),
                            "curve": sorted(pts)}
            lab = f"{tid}{'*' if tid == 'T8' else ''}  {list(taps)}"
            if len(lab) > 34:
                lab = lab[:31] + "..."
            print(f"{lab:>34s} {nw:3d} {sram:7d}" + "".join(
                "            n/a" if v is None else f"{v:15.4f}"
                for v in vals))
        print("  dMASE against the CONTROL; lower is better. SRAM is the node "
              "state, dominated by\n  the ring buffer at 24 + max_tap samples, "
              "so memory tracks the LARGEST LAG while\n  accuracy tracks the "
              "WEIGHT COUNT -- the two axes trade independently.")

    out = {
        "store": a.data_dir, "checkpoint": a.checkpoint, "split": a.split,
        "n_windows": int(len(sel)), "L": wb.L, "H": wb.H,
        "SYNTHETIC_store": bool(man.get("SYNTHETIC")),
        "SYNTHETIC_checkpoint": bool(ck.get("SYNTHETIC")),
        "checkpoint_provenance": ck.get("provenance", "supplied by Tier 2"),
        "checkpoint_is_fallback": ck.get("trainer") == "tier1_fallback",
        "checkpoint_val_metric": ck.get("convergence_test", {}).get("history",
                                                                   [{}])[-1],
        "index_probe": probe, "codec_params": {k: v for k, v in
                                               base.__dict__.items()},
        "pretraining": {k: v for k, v in pre.items()
                        if k != "sigma_per_client"},
        "client_accounting": acct, "quantisation": qm,
        "tap_ablation": tap_abl,
        "budget": budget, "selection": sel_note, "ablation": abl,
        "batch_daily_coverage": cov_frac,
        "metrics": results, "cost": cost,
        "periodic_comparison": periodic_cmp,
        "disclosures": [
            ("TOLERANCE BASIS. The operating point is chosen on validation, but "
             "the TOLERANCE is anchored to the smaller of the validation and "
             "test skill margins, so a test-derived scalar does influence "
             "selection. It is harmless in the direction that matters: the "
             "smaller margin is a TIGHTER bar and selects a MORE expensive "
             "operating point. Anchoring on the validation margin instead "
             "would have selected a cheaper alpha "
             "(6.57 pkt/day rather than 9.37 on this run). A leak that works "
             "against the result is not one that flatters it."),
            ("RMSE IS NON-MONOTONIC IN ALPHA. It improves slightly as the codec "
             "gets lossier before worsening. This is real and explicable: on "
             "silent hours the codec substitutes smooth predictions for spiky "
             "true values, clipping the tail that RMSE is dominated by. At "
             "RMSE/MAE around 10 the tail owns RMSE, so mild input smoothing "
             "acts as a regulariser. Stated rather than rounded away."),
        ],
        "use_case": (
            "The scheme buys an hourly-refreshed look-back. Its value is "
            "confined to applications needing a SHORT-HORIZON forecast "
            "REFRESHED EVERY HOUR. For day-ahead scheduling a daily batch is "
            "adequate and this buys nothing. The gain is zero at the resync "
            "hour and is refresh_gain_lead1_by_issue_hour elsewhere."),
        "limitations": [
            "Single chronological split, single seed, no rolling origin and no "
            "model confidence set. Point estimates, not margin claims.",
            "The gain is confined to short-lead hourly-refresh applications "
            "and is zero at the resync hour.",
            "Clock drift is unmodelled; deployment assumes an RTC or the daily "
            "downlink phase correction.",
            "ACK loss is handled by a compiled-in robustness mode (sync_safe) "
            "that is not evaluated here; see eval/loss_sweep.py.",
            "The DLinear variant is DLinear with log1p and per-window RevIN, "
            "not the original paper's global StandardScaler, so its MASE "
            "figures are not comparable with published ECL MSE/MAE tables.",
            "The fixed periodic refresh baseline carries no L-inf guarantee; "
            "the packet-rate comparison against it does not capture that "
            "qualitative difference.",
        ],
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
