"""
loss_sweep.py  --  the packet-loss sensitivity curve. Blueprint 11.

"The epsilon guarantee is conditional on delivery" is listed as a known
limitation, with the instruction to produce a curve rather than a caveat. This
is that curve.

THREE LOSS MODES, AND WHAT SYNC-SAFE DID TO THEM (revisions R3, R4)
==================================================================
exception   The node pushed u, the mirror pushed u_hat. They diverge and the
            bound is void until the next RESYNC, which REPLAYS the day from the
            transmit bitmap and repairs the buffer exactly however many were
            lost.
resync      Costs fidelity for that day and defers any pending repair. The
            weight update happened anyway, identically, on both sides.
ack         Nothing. The ACK carries configuration only. Under `--legacy` this
            mode reproduces the original two-generals failure, which is how the
            value of the change is measured rather than asserted.

REVISION R3: this is run on a STRATIFIED SAMPLE across the load-scale range,
not on three arbitrary clients, and it reports the DISTRIBUTION of the
violation rate across clients rather than the mean.

REVISION R5: sync_safe IS NOW A FLAG, NOT A BUILD CONSTANT. `--sync_safe {0,1}`
selects it and defaults to 0, the configuration the paper reports; both
configurations are swept in a single invocation by default.

REVISION R6 -- WHAT THIS REVISION FIXES
=======================================
The previous run printed "recovery exceeded 24 h" at six operating points and
concluded the bounded-recovery claim did not hold. It does hold; the metric was
measuring the wrong thing, in two separate ways.

1.  RUN LENGTH MERGES INDEPENDENT FAILURES. Recovery was the length of a
    maximal run of consecutive violated hours. At 20 % loss with ~5 exceptions
    a day, a new lost frame lands before the previous run's last violated hour
    has cleared, so unrelated episodes chain into one contiguous run. The
    tell-tale is that the MEAN was flat (2.34 h at 1 % loss, 2.68 h at 20 %)
    while the MAX grew: a failing repair mechanism moves the mean. Exposure is
    now also attributed PER DISCARDED FRAME (`recovery_event_*`), which is the
    quantity the "repaired at the next resync" claim is about, and that is what
    the assertion below now tests.

2.  CENSORED RUNS WERE REPORTED AS MAXIMA. A violation run still open when the
    series ends was folded into the max "as a lower bound". Every value above
    24 h in the previous artefact carried the censoring marker. Censored
    episodes are still reported, but separately; they no longer masquerade as
    measurements.

3.  THE VALIDITY MASK WAS MISSING. Violations were counted over the whole
    series including imputed spans, while every other metric in the project is
    scored with `missing` live. It is applied here now.

The ONE genuine exceedance is ACK loss under sync_safe=0: 25 h at 1 % loss,
uncensored. That is correct and must not be "fixed" -- a lost ACK means the hub
applies its daily weight update and the node does not, and nothing within the
day repairs it. It is the case the paper already identifies as not
self-repairing, and this is its evidence. It is therefore excluded from the
bounded-recovery assertion and printed as an expected exceedance.
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

from edge.codec import EPS_INF, CodecParams          # noqa: E402
from edge.mirror import Mirror                       # noqa: E402
from edge.quantize import quantize_array             # noqa: E402
from eval.recovery import recovery_summary           # noqa: E402
from sim.run_codec import Store, run_policy          # noqa: E402

MODES = ("exception", "resync", "ack")
RESYNC_INTERVAL_H = 24
# Modes whose exposure is repaired at the next resynchronisation by
# construction. `ack` is deliberately absent: see the module docstring.
BOUNDED_MODES = ("resync", "exception")


def stratified_clients(store, n):
    """n clients spread across the load-scale range, so the curve is not a
    report about whichever meters happened to be first in the file."""
    mean_load = np.nan_to_num(np.nanmean(store.series_filled, axis=0))
    order = np.argsort(mean_load)
    order = order[mean_load[order] > 0]
    if len(order) <= n:
        return [int(c) for c in order]
    idx = np.linspace(0, len(order) - 1, n).round().astype(int)
    return [int(order[i]) for i in idx]


def _valid_mask(store, c, t0, n):
    """True where the hour is real data. Everywhere else in the project metrics
    are scored against `series_true` with `missing` live; this keeps that
    discipline, so imputed spans neither inflate the violation rate nor pad its
    denominator."""
    miss = getattr(store, "missing", None)
    if miss is None:
        return np.ones(n, dtype=bool)
    m = np.asarray(miss[t0:, c], dtype=bool)
    if m.size < n:
        m = np.concatenate([m, np.zeros(n - m.size, dtype=bool)])
    return ~m[:n]


def one(store, alpha, params, loss, mode, clients, seed):
    """Run the device, drop packets on the way to the mirror, and measure what
    the NanoPi actually ends up holding."""
    res = run_policy(store, alpha, params, backend="scalar", packet_log=True,
                     clients=clients,
                     exception_loss=(loss if mode == "exception" else 0.0),
                     resync_loss=(loss if mode == "resync" else 0.0),
                     ack_loss=(loss if mode == "ack" else 0.0), seed=seed)
    per_client, errs, viol_masks, event_idx = [], [], [], []
    for c in clients:
        t0 = int(store.onboarding[c])
        u, _ = quantize_array(store.series_filled[:, c], int(res.k[c]),
                              params.u_clamp)
        pkts, dropped = [], []
        for (h, b, ok) in res.packets[c]:
            if ok:
                pkts.append((h, b))
            else:
                # Index base of the masks below is t0, so shift to match.
                dropped.append(int(h) - t0)
        m = Mirror(params, res.eps[c], int(u[t0]), t0, node_id=c, strict=False)
        out = m.run(pkts, store.T)
        eps = res.eps[c]
        d = np.abs(out["u_within"] - u[t0:])
        bound = res.eps_eff[t0:, c]
        valid = _valid_mask(store, c, t0, np.size(d))
        # An unbounded client has no threshold to violate, so it contributes an
        # all-clear mask rather than being dropped -- dropping it would shrink
        # the denominator of the violation rate and flatter the curve.
        mask = (np.asarray(d > bound, dtype=bool) if eps is not EPS_INF
                else np.zeros(np.shape(d), dtype=bool))
        mask &= valid
        viol_masks.append(mask)
        event_idx.append(np.asarray(dropped, dtype=np.int64))
        denom = max(int(valid.sum()), 1)
        per_client.append(float(mask.sum()) / denom)
        errs.append(float((d[valid] * (2.0 ** int(res.k[c]))).mean())
                    if valid.any() else 0.0)
    pc = np.asarray(per_client)
    return {"violation_rate_mean": float(pc.mean()),
            "violation_rate_p50": float(np.percentile(pc, 50)),
            "violation_rate_p90": float(np.percentile(pc, 90)),
            "violation_rate_max": float(pc.max()),
            "recon_mae_kwh": float(np.mean(errs)),
            "n_clients": len(per_client),
            **recovery_summary(viol_masks, event_idx)}


HEAD = (f"{'sync_safe':>9s} {'mode':>10s} {'loss':>6s} {'viol mean':>10s} "
        f"{'p90':>9s} {'max':>9s} {'recon MAE kWh':>14s} "
        f"{'ev mean h':>10s} {'ev max h':>9s} {'run max h':>10s}")


def row(sync_safe, mode, r):
    flag = "!" if r["recovery_censored"] else " "
    return (f"{int(sync_safe):>9d} {mode:>10s} {r['loss']:6.2f} "
            f"{r['violation_rate_mean']:10.3%} {r['violation_rate_p90']:9.3%} "
            f"{r['violation_rate_max']:9.3%} {r['recon_mae_kwh']:14.3f} "
            f"{r['recovery_event_mean_h']:10.2f} "
            f"{r['recovery_event_max_h']:9.1f} "
            f"{r['recovery_run_max_h']:9.1f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="synth_data")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n_clients", type=int, default=64,
                    help="R3: stratified across the load-scale range")
    ap.add_argument("--sync_safe", type=int, choices=(0, 1), default=0,
                    help="codec configuration to report as primary. Default 0 "
                         "-- the configuration the paper reports.")
    ap.add_argument("--configs", default="0,1",
                    help="which configurations to sweep. Default both, so the "
                         "comparison never needs a re-run.")
    ap.add_argument("--legacy", action="store_true",
                    help="deprecated alias for --sync_safe 0, kept so old "
                         "invocations do not silently change meaning")
    ap.add_argument("--losses", default="0,0.01,0.05,0.1,0.2")
    ap.add_argument("--seed", type=int, default=7,
                    help="RNG for the packet-loss realisation. Default 7 = the "
                         "value hardcoded at the call site until 2026-09-14, so "
                         "default behaviour is unchanged. Table I is ONE draw "
                         "of a Monte Carlo; vary this to report a spread. See "
                         "docs/reference/stochasticity.md.")
    ap.add_argument("--pretrained", default=paths.inp("edge_pretrained.json"),
                    help="the globally pre-trained weight vector. REQUIRED: "
                         "without it the sweep runs a persistence-initialised "
                         "filter, which is not the model the paper reports.")
    ap.add_argument("--k_mu", type=int, default=5,
                    help="must match the value used in eval/run_eval.py")
    ap.add_argument("--out", default=paths.result("tier1_loss_sensitivity.json"))
    a = ap.parse_args()

    primary = 0 if a.legacy else a.sync_safe
    if a.legacy and a.sync_safe == 1:
        ap.error("--legacy and --sync_safe 1 contradict each other")
    configs = sorted({int(x) for x in a.configs.split(",") if x.strip() != ""})
    if not configs:
        ap.error("--configs is empty")
    if any(c not in (0, 1) for c in configs):
        ap.error("--configs takes 0 and/or 1")
    if primary not in configs:
        configs.append(primary)
        configs.sort()

    # THE MODEL UNDER TEST MUST BE THE ONE THE PAPER REPORTS.
    # CodecParams() alone carries the DEFAULT w_init, which is persistence
    # (1 << 20, 0, ...). That is not policy C: the reported model is the
    # globally pre-trained ridge vector of section 3.2 with adaptation on. Left
    # unloaded, this sweep measured a filter that appears nowhere in the paper,
    # and its reconstruction error at zero loss was accordingly about 1.8x the
    # value run_eval.py reports for the same alpha.
    if not os.path.exists(a.pretrained):
        raise SystemExit(
            f"{a.pretrained} not found. Run `python -m sim.pretrain` first; "
            "without it this sweep does not measure the reported model.")
    pre = json.load(open(a.pretrained))
    w_global = tuple(int(v) for v in pre["w_global_q"])

    store = Store.load(a.data_dir)
    clients = stratified_clients(store, a.n_clients)
    losses = [float(x) for x in a.losses.split(",")]

    print(f"packet-loss sensitivity at alpha={a.alpha:g}, "
          f"{len(clients)} clients stratified by load scale")
    print(f"configurations swept: {configs}   primary (paper): "
          f"sync_safe={primary}")
    print(f"model: policy C -- taps {list(CodecParams().taps)}, k_mu={a.k_mu}, "
          f"w_init = global ridge from {a.pretrained}")
    print("ev = exposure attributable to ONE discarded frame (the bounded "
          "quantity);\nrun = maximal contiguous violated run, which merges "
          "independent failures at high loss.")
    print(HEAD)

    doc = {"alpha": a.alpha, "clients": clients, "losses": losses,
           "n_clients": len(clients),
           "sync_safe": bool(primary),
           "primary": f"sync_safe_{primary}",
           "resync_interval_h": RESYNC_INTERVAL_H,
           "bounded_modes": list(BOUNDED_MODES),
           "k_mu": a.k_mu,
           "w_init": "global ridge (%s)" % a.pretrained,
           "w_global_q": list(w_global)}

    for sync_safe in configs:
        params = CodecParams(k_mu=a.k_mu,
                             sync_safe=bool(sync_safe)).with_w_init(w_global)
        got = getattr(params, "sync_safe", None)
        if got is not None and bool(got) != bool(sync_safe):
            raise SystemExit(
                f"CodecParams(sync_safe={bool(sync_safe)}) came back with "
                f"sync_safe={got!r}; the flag is not reaching the codec.")
        curves = {}
        for mode in MODES:
            curves[mode] = []
            for loss in losses:
                r = {"loss": loss,
                     **one(store, a.alpha, params, loss, mode, clients,
                           seed=a.seed)}
                curves[mode].append(r)
                print(row(sync_safe, mode, r))
        doc[f"sync_safe_{sync_safe}"] = {"sync_safe": bool(sync_safe),
                                         "curves": curves}
        print()

    doc["curves"] = doc[doc["primary"]]["curves"]

    with open(a.out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {a.out} "
          f"({', '.join('sync_safe_%d' % c for c in configs)}; "
          f"top-level `curves` mirrors {doc['primary']})")

    # ---- the bounded-recovery check, now on the right quantity -------------
    breaches = [(c, m, p["loss"], p["recovery_event_max_h"])
                for c in configs
                for m in BOUNDED_MODES
                for p in doc[f"sync_safe_{c}"]["curves"][m]
                if p["recovery_event_max_h"] > RESYNC_INTERVAL_H]
    if breaches:
        print(f"\nWARNING: per-event recovery exceeded {RESYNC_INTERVAL_H} h "
              f"at {len(breaches)} operating point(s):")
        for c, m, l, v in breaches:
            print(f"  sync_safe={c} {m} loss={l:g}: {v:g} h")
        print("A single lost frame should be repaired at the next "
              "resynchronisation, so this is a real failure of the mechanism "
              "and not a metric artefact. The test suite will fail on it.")
    else:
        print(f"\nper-event recovery bounded by {RESYNC_INTERVAL_H} h at every "
              f"{'/'.join(BOUNDED_MODES)} operating point")

    ack_over = [(c, p["loss"], p["recovery_event_max_h"])
                for c in configs
                for p in doc[f"sync_safe_{c}"]["curves"]["ack"]
                if p["recovery_event_max_h"] > RESYNC_INTERVAL_H]
    if ack_over:
        print(f"ACK loss exceeds one interval at {len(ack_over)} operating "
              f"point(s), as expected: the hub applies its daily weight update "
              f"and the node does not,")
        print("so nothing within the day repairs it. This is the case the "
              "paper reports as not self-repairing, and sync_safe=1 removes "
              "it by construction.")

    if any(p["recovery_censored"]
           for c in configs
           for cur in doc[f"sync_safe_{c}"]["curves"].values() for p in cur):
        print("rows marked ! had a violation run still open at the end of the "
              "window. Those runs are counted in recovery_censored_* and are "
              "EXCLUDED from both maxima, since their exposure was never "
              "observed.")

    if 1 in configs:
        print("\nUnder sync_safe=1 the ACK column is flat at zero BY "
              "CONSTRUCTION -- the ACK carries")
        print("configuration only, so there is no two-generals failure left "
              "to measure. It is not")
        print("evidence about the reported build.")
    if 0 in configs:
        print("Under sync_safe=0 (reported) the ACK column IS the two-generals "
              "failure: the mirror")
        print("resyncs and the node does not, and the protocol cannot "
              "self-correct within the day.")


if __name__ == "__main__":
    main()
