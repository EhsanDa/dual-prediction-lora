"""
protocol_matrix.py  --  what ONE lost frame does, for every node/gateway pairing,
simulated CAUSALLY.

WHY THIS MODULE EXISTS
======================
Between 2026-09-14 and 2026-09-28 this repository reached three different
conclusions about the same paragraph of the paper (Section IV-E, "A lost
resynchronisation is symmetric ...") and rewrote its documents each time. The
root of the disagreement was that the question was never posed precisely: *which
node behaviour* and *which gateway behaviour* is being asked about. This module
poses it precisely and answers it for every combination, in one place, so it
does not have to be argued again. docs/findings/2026-09-28-frame-loss-settled.md
is the write-up; tests/test_protocol_matrix.py pins the conclusions.

THE TWO THINGS THAT WERE CONFLATED
==================================
1. There are two NODE behaviours for the legacy (sync_safe = 0) arithmetic:

     gated     send RESYNC -> wait for the ACK -> on ACK run the 24 updates on the
               true day, on no ACK keep the old weights and the old buffer.
               This is the protocol the paper describes and reports, the one
               TIER1_EDGE_CODEC_BLUEPRINT.md section 5.1 specifies ("on no ACK
               after retries: keep the old weights, keep the old buffer. Both
               sides do the same, so they stay synchronised"), and the one
               sim/run_codec.py, export/gen_golden.py and firmware/test_host.c
               implement. Table I was produced with it.
     ungated   run the 24 updates unconditionally, THEN send the RESYNC carrying
               the post-update weights. This is what firmware/node/main.ino does
               (and did in the original import, commit 1d89f66) while compiled
               with SYNC_SAFE 0. It is the R4 / sync_safe scheduling applied to
               legacy arithmetic -- a combination nothing else models.

   eval/single_loss_exposure.py drops a RESYNC from the gateway's input after
   running the node as if every boundary were applied, so it measures the
   UNGATED node. Its "4 breaches in 2 474 probes" is correct for that node and
   says nothing about the gated protocol the paper reports.

2. sim/run_codec.py builds the RESYNC frame AFTER deciding whether the node
   applied the update -- a decision that, physically, depends on an ACK that has
   not arrived yet. The frame's weight field therefore "knows the future". That
   is harmless for everything the paper reports, because the legacy gateway
   never reads that field. It is NOT harmless for any gateway that adopts the
   weights (CodecParams.adopt_resync_weights), so this module builds every
   frame from the state the node holds at the moment it transmits.

THE GATEWAY BEHAVIOURS
======================
  legacy        on RESYNC: overwrite the day with truth and run the 24 updates
                from its OWN weights (edge/mirror.py, legacy branch).
  adopt_after   on RESYNC: as legacy, then replace its weights with the frame's.
                This is what CodecParams.adopt_resync_weights implements today.
                It is correct only if the frame carries POST-update weights,
                i.e. only for an ungated node.
  adopt_before  on RESYNC: replace its weights with the frame's, THEN run the 24
                updates. Correct for a gated node, whose frame can only carry
                PRE-update weights.
  sync_safe     edge/mirror.py's sync_safe branch, verbatim.

EVENTS (one per probe, on an otherwise lossless channel)
=======================================================
  none       nothing is lost
  exception  one EXCEPTION frame is lost on the uplink
  resync     one RESYNC frame is lost on the uplink (so no ACK is sent either)
  ack        the RESYNC arrives but its ACK is lost on the downlink

Reproduce the headline table (15 clients, the first 400 h of life, ~10 min):

    python -m eval.protocol_matrix --data_dir ../tier2_data
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import paths                                                      # noqa: E402

from edge.codec import EPS_INF, CodecParams, CodecState           # noqa: E402
from edge.quantize import quantize_array                          # noqa: E402

RESYNC_INTERVAL_H = 24

# (node, gateway) pairs worth asking about. The first is the paper's.
PAIRS = (
    ("gated", "legacy"),         # the protocol the paper reports
    ("gated", "adopt_before"),   # Option 1, in the only order a gated node allows
    ("gated", "adopt_after"),    # Option 1 as currently coded, on the paper's node
    ("ungated", "legacy"),       # firmware/node/main.ino against a legacy gateway
    ("ungated", "adopt_after"),  # main.ino against Option 1 as currently coded
    ("sync_safe", "sync_safe"),  # revision R4
)
EVENTS = ("none", "exception", "resync", "ack")


class Link:
    """One node and one gateway, advanced hour by hour.

    Deliberately NOT built on sim/run_codec.py or edge/mirror.py: those are
    what is being checked. It is built on edge/codec.py's CodecState, which is
    the specification both of them are validated against, and it is itself
    validated against them wherever they agree on semantics
    (tests/test_protocol_matrix.py).
    """

    def __init__(self, node_mode, gw_mode, w_init, k_mu, eps, u0, t0):
        ss = node_mode == "sync_safe"
        if ss != (gw_mode == "sync_safe"):
            raise ValueError("sync_safe arithmetic must be on both sides or "
                             "neither")
        p = CodecParams(k_mu=k_mu, sync_safe=ss).with_w_init(w_init)
        self.node_mode, self.gw_mode = node_mode, gw_mode
        self.node = CodecState(p, eps, u0=u0, t_start=t0)
        self.gw = CodecState(p, eps, u0=u0, t_start=t0)

    # ---------------------------------------------------------- day boundary
    def boundary(self, t, lose_resync=False, lose_ack=False):
        n, g = self.node, self.gw
        values = n.resync_payload()
        bitmap = n.resync_bitmap()

        # ---- node, in physical order
        if self.node_mode == "gated":
            w_frame = list(n.w)                     # what it holds when it sends
            acked = (not lose_resync) and (not lose_ack)
            if acked:
                n.day_boundary(values)
            else:
                n.new_day()
        elif self.node_mode == "ungated":
            n.day_boundary(values)                  # unconditional, then send
            w_frame = list(n.w)
        else:                                       # sync_safe
            n.day_boundary()
            w_frame = list(n.w)

        # ---- gateway, given what arrived
        got = not lose_resync
        if self.gw_mode == "sync_safe":             # edge/mirror.py, verbatim
            if got and bitmap:
                g.replay_day(list(values), list(bitmap))
            g.day_boundary()
            if got:
                g.w = [int(x) for x in w_frame]
                g.u_hat = g.predict()
                g.mark_day_start()
        elif not got:
            g.new_day()
        elif self.gw_mode == "legacy":
            g.day_boundary(values)
        elif self.gw_mode == "adopt_after":
            g.day_boundary(values)
            g.w = [int(x) for x in w_frame]
            g.u_hat = g.predict()
        elif self.gw_mode == "adopt_before":
            g.w = [int(x) for x in w_frame]
            g.day_boundary(values)                  # re-predicts internally
        else:
            raise ValueError(self.gw_mode)

    # ------------------------------------------------------------- one hour
    def hour(self, t, u, clamped=False, lose_exception=False):
        """Returns (gateway reconstruction, node's eps_eff, node transmitted)."""
        r = self.node.step(int(u), clamped=clamped)
        if r.tx and not lose_exception:
            rg = self.gw.apply_exception(r.e, clamped=r.clamped)
        else:
            rg = self.gw.apply_silent()
        return rg.u_recon, r.eps_eff, r.tx

    def in_sync(self):
        return self.node.sync_snapshot() == self.gw.sync_snapshot()

    def weight_gap(self):
        return max(abs(a - b) for a, b in zip(self.node.w, self.gw.w))


def run_link(link, u, clamped, valid, t_from, t_to, event=None):
    """Advance `link` over [t_from, t_to). `event` = (kind, hour) or None.

    Returns per-hour arrays relative to t_from: violation (gateway outside the
    bound the node held that hour), in_sync, weight gap.
    """
    n = t_to - t_from
    viol = np.zeros(n, bool)
    sync = np.zeros(n, bool)
    gap = np.zeros(n, np.int64)
    kind, at = event if event else (None, None)
    for i, t in enumerate(range(t_from, t_to)):
        if link.node.is_boundary(t):
            link.boundary(t, lose_resync=(kind == "resync" and t == at),
                          lose_ack=(kind == "ack" and t == at))
        rec, eps_eff, _ = link.hour(t, u[t], bool(clamped[t]),
                                    lose_exception=(kind == "exception"
                                                    and t == at))
        if eps_eff is not EPS_INF:
            viol[i] = abs(int(rec) - int(u[t])) > eps_eff and valid[t]
        sync[i] = link.in_sync()
        gap[i] = link.weight_gap()
    return viol, sync, gap


def probe_exposure(viol, h):
    """eval/single_loss_exposure.py's definition, kept for comparability:
    0 if the loss hour itself is clean, else hours to the first clean hour."""
    if not viol[h]:
        return 0
    nz = np.flatnonzero(~viol[h:h + 600])
    return int(nz[0]) if nz.size else -1


def probe_client(store, c, w_init, k_mu, alpha, pair, window, horizon,
                 events=("exception", "resync", "ack"), stride_exc=1,
                 boundary_stride=1, scales=None):
    """Every candidate event in the first `window` hours of client c's life,
    one per probe, for one (node, gateway) pair.

    window > 0: the commissioning mode -- every probe runs to
                t0 + window + horizon (the setting of the settlement's table).
    window = 0: the whole life -- every probe runs `horizon` hours past its
                own event, so a late event is followed as far as an early one.
    Strides are applied as candidates are met, in time order: stride 1 keeps
    every candidate, so the default selection is unchanged.
    """
    if scales is None:
        from sim.run_codec import choose_scales, eps_for_alpha
        k, _ = choose_scales(store)
        eps_all, _ = eps_for_alpha(store, k, alpha)
    else:
        k, eps_all = scales
    eps = eps_all[c]
    if eps is EPS_INF:
        return None
    params = CodecParams(k_mu=k_mu)
    u, cl = quantize_array(store.series_filled[:, c], int(k[c]), params.u_clamp)
    t0 = int(store.onboarding[c])
    whole = window <= 0
    t_end = store.T if whole else min(store.T, t0 + window + horizon)
    # Whole life: an event is probed only if its full follow-up fits in the
    # series. An event in the last days has no later RESYNC to be repaired
    # by, and would be counted as "never re-synchronised" for a reason that
    # is the end of the data, not the protocol (right-censoring).
    in_window = ((lambda age: age > 0 and t0 + age + horizon <= store.T)
                 if whole else (lambda age: 0 < age < window))
    valid = ~np.asarray(store.missing[:, c], bool)

    # Lossless pass, checkpointing the link at the START of every hour that
    # carries a candidate event -- before its boundary, if it has one -- so a
    # probe replays that hour from exactly the state the lossless run had.
    ckpt, cand = {}, []
    lossless = Link(*pair, w_init, k_mu, eps, int(u[t0]), t0)
    v0 = np.zeros(t_end - t0, bool)
    s0 = np.zeros(t_end - t0, bool)
    n_b = n_x = 0                      # boundaries / exceptions met in window
    for i, t in enumerate(range(t0, t_end)):
        age = t - t0
        is_b = lossless.node.is_boundary(t)
        inside = in_window(age)
        want_b = inside and is_b and n_b % boundary_stride == 0
        want_x = inside and "exception" in events and n_x % stride_exc == 0
        pre = copy.deepcopy(lossless) if (want_b or want_x) else None
        if inside and is_b:
            if want_b:
                for ev in ("resync", "ack"):
                    if ev in events:
                        cand.append((ev, t))
                        ckpt[t] = pre
            n_b += 1
        if is_b:
            lossless.boundary(t)
        rec, eps_eff, tx = lossless.hour(t, u[t], bool(cl[t]))
        if inside and tx and "exception" in events:
            if want_x:
                cand.append(("exception", t))
                ckpt[t] = pre
            n_x += 1
        if eps_eff is not EPS_INF:
            v0[i] = abs(int(rec) - int(u[t])) > eps_eff and valid[t]
        s0[i] = lossless.in_sync()
    exc = [e for e in cand if e[0] == "exception"]
    cand = [e for e in cand if e[0] != "exception"] + exc

    rows = [{"client": c, "t0": t0, "event": "none", "age_h": None,
             "lossless_violations": int(v0.sum()),
             "lossless_always_in_sync": bool(s0.all())}]
    for kind, t in cand:
        link = copy.deepcopy(ckpt[t])
        p_end = t_end if not whole else min(store.T, t + horizon)
        viol, sync, gap = run_link(link, u, cl, valid, t, p_end,
                                   event=(kind, t))
        h = 0
        out_of_sync = np.flatnonzero(~sync)
        rows.append({
            "client": c, "t0": t0, "event": kind, "age_h": t - t0,
            "probe_exposure_h": probe_exposure(viol, h),
            "violated_hours": int(viol[:600].sum()),
            "any_violation": bool(viol[:600].any()),
            "ever_out_of_sync": bool(out_of_sync.size),
            "back_in_sync_after_h": (None if not out_of_sync.size else
                                     (int(out_of_sync[-1]) + 1
                                      if out_of_sync[-1] + 1 < len(sync)
                                      else -1)),
            "weight_gap_at_end": int(gap[-1]),
        })
    return rows


SHARES = {
    "beyond_one_interval": lambda r: (r["probe_exposure_h"] > RESYNC_INTERVAL_H
                                      or r["probe_exposure_h"] < 0),
    "any_violation": lambda r: r["any_violation"],
    "never_back_in_sync": lambda r: r["back_in_sync_after_h"] == -1,
}


def share_ci(rows, kind, pred, n_boot=2000, ci=95, seed=0):
    """Share of `kind` probes satisfying `pred`, with a percentile bootstrap
    over CLIENTS (the quasi-independent units): each resample draws clients
    with replacement and pools their probes."""
    by_c = {}
    for r in rows:
        if r["event"] == kind:
            k, n = by_c.get(r["client"], (0, 0))
            by_c[r["client"]] = (k + bool(pred(r)), n + 1)
    if not by_c:
        return None
    ks = np.array([v[0] for v in by_c.values()], np.float64)
    ns = np.array([v[1] for v in by_c.values()], np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ks), size=(n_boot, len(ks)))
    draws = ks[idx].sum(1) / np.maximum(ns[idx].sum(1), 1)
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    return {"share": float(ks.sum() / ns.sum()), "k": int(ks.sum()),
            "n": int(ns.sum()), "clients": len(ks),
            "ci": [float(np.percentile(draws, lo)),
                   float(np.percentile(draws, hi))]}


def summarise(rows, n_boot=0):
    out = {}
    for kind in ("exception", "resync", "ack"):
        R = [r for r in rows if r["event"] == kind]
        if not R:
            continue
        pe = [r["probe_exposure_h"] for r in R]
        pe_ok = np.array([x for x in pe if x >= 0], np.float64)
        vh = np.array([r["violated_hours"] for r in R], np.float64)
        stuck = [r for r in R if r["back_in_sync_after_h"] == -1]
        out[kind] = {
            "probes": len(R),
            "breaches_over_24h": sum(1 for x in pe if x > RESYNC_INTERVAL_H
                                     or x < 0),
            "worst_probe_exposure_h": max(pe),
            "probe_exposure_p50_h": float(np.percentile(pe_ok, 50)),
            "probe_exposure_p90_h": float(np.percentile(pe_ok, 90)),
            "probes_with_any_violation": sum(r["any_violation"] for r in R),
            "max_violated_hours": max(r["violated_hours"] for r in R),
            "violated_hours_p50": float(np.percentile(vh, 50)),
            "violated_hours_p90": float(np.percentile(vh, 90)),
            "probes_ever_out_of_sync": sum(r["ever_out_of_sync"] for r in R),
            "probes_never_back_in_sync": len(stuck),
            "probes_weights_differ_at_end": sum(r["weight_gap_at_end"] > 0
                                                for r in R),
        }
        if n_boot:
            out[kind]["shares"] = {name: share_ci(rows, kind, f, n_boot)
                                   for name, f in SHARES.items()}
    base = [r for r in rows if r["event"] == "none"]
    out["lossless"] = {
        "clients": len(base),
        "violations": sum(r["lossless_violations"] for r in base),
        "clients_ever_out_of_sync": sum(not r["lossless_always_in_sync"]
                                        for r in base),
    }
    return out


# ------------------------------------------------------------- parallel runs
_G = {}                                 # fork-shared state for the workers


def _work(c):
    g = _G
    return probe_client(g["store"], c, g["w"], g["k_mu"], g["alpha"],
                        g["pair"], g["window"], g["horizon"],
                        stride_exc=g["stride_exc"],
                        boundary_stride=g["boundary_stride"],
                        scales=g["scales"])


def run_pair(store, clients, w, k_mu, alpha, pair, window, horizon,
             stride_exc, boundary_stride, scales, workers):
    _G.update(store=store, w=w, k_mu=k_mu, alpha=alpha, pair=pair,
              window=window, horizon=horizon, stride_exc=stride_exc,
              boundary_stride=boundary_stride, scales=scales)
    rows = []
    if workers <= 1:
        results = map(_work, clients)
    else:
        import multiprocessing as mp
        pool = mp.get_context("fork").Pool(workers)
        results = pool.imap(_work, clients, chunksize=1)
    for r in results:
        if r:
            rows += r
    if workers > 1:
        pool.close()
        pool.join()
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="../tier2_data")
    ap.add_argument("--pretrained", default=paths.inp("edge_pretrained.json"))
    ap.add_argument("--clients",
                    default="118,245,2,6,113,147,300,50,200,17,88,260,5,99,150",
                    help="default = the 15 clients eval/single_loss_exposure.py "
                         "probed, so the two are directly comparable. 'all' = "
                         "every client with a finite threshold.")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--k_mu", type=int, default=5)
    ap.add_argument("--window", type=int, default=400,
                    help="probe events in the first N hours of life; 0 = the "
                         "whole life, each probe followed --horizon hours")
    ap.add_argument("--horizon", type=int, default=800)
    ap.add_argument("--stride_exc", type=int, default=1)
    ap.add_argument("--boundary_stride", type=int, default=1,
                    help="probe every Nth day boundary (RESYNC and ACK)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="client-level bootstrap resamples for the shares; "
                         "0 = none")
    ap.add_argument("--pairs", default="all",
                    help="comma list of node:gateway, e.g. gated:legacy")
    ap.add_argument("--no_rows", action="store_true",
                    help="write the summaries only (population runs)")
    ap.add_argument("--out", default=paths.result("protocol_matrix.json"))
    a = ap.parse_args(argv)

    from sim.run_codec import Store, choose_scales, eps_for_alpha
    store = Store.load(a.data_dir)
    w = tuple(int(x) for x in json.load(open(a.pretrained))["w_global_q"])
    k, _ = choose_scales(store)
    eps_all, _ = eps_for_alpha(store, k, a.alpha)
    clients = ([c for c in range(store.M) if eps_all[c] is not EPS_INF]
               if a.clients == "all" else [int(x) for x in a.clients.split(",")])
    pairs = (PAIRS if a.pairs == "all" else
             tuple(tuple(p.split(":")) for p in a.pairs.split(",")))

    doc = {"alpha": a.alpha, "k_mu": a.k_mu, "window_h": a.window,
           "horizon_h": a.horizon, "stride_exc": a.stride_exc,
           "boundary_stride": a.boundary_stride, "bootstrap": a.bootstrap,
           "n_clients": len(clients), "clients": clients, "pairs": {}}
    print(f"{len(clients)} clients, window {a.window or 'whole life'}, "
          f"horizon {a.horizon} h, exception stride {a.stride_exc}, boundary "
          f"stride {a.boundary_stride}, {a.workers} worker(s)")
    print(f"{'node':>9s} {'gateway':>12s} {'event':>9s} {'probes':>6s} "
          f"{'>24h':>5s} {'worst':>6s} {'any viol':>8s} {'max vh':>6s} "
          f"{'desync':>6s} {'never re-sync':>13s} {'w differ @end':>13s}")
    for pair in pairs:
        t_start = time.time()
        rows = run_pair(store, clients, w, a.k_mu, a.alpha, pair, a.window,
                        a.horizon, a.stride_exc, a.boundary_stride,
                        (k, eps_all), a.workers)
        s = summarise(rows, a.bootstrap)
        key = f"{pair[0]}:{pair[1]}"
        doc["pairs"][key] = {"summary": s}
        if not a.no_rows:
            doc["pairs"][key]["rows"] = rows
        L = s["lossless"]
        print(f"{pair[0]:>9s} {pair[1]:>12s} {'none':>9s} {L['clients']:6d} "
              f"lossless violations {L['violations']}, clients ever out of "
              f"sync {L['clients_ever_out_of_sync']}")
        for kind in ("exception", "resync", "ack"):
            if kind not in s:
                continue
            x = s[kind]
            print(f"{'':>9s} {'':>12s} {kind:>9s} {x['probes']:6d} "
                  f"{x['breaches_over_24h']:5d} {x['worst_probe_exposure_h']:6d} "
                  f"{x['probes_with_any_violation']:8d} "
                  f"{x['max_violated_hours']:6d} "
                  f"{x['probes_ever_out_of_sync']:6d} "
                  f"{x['probes_never_back_in_sync']:13d} "
                  f"{x['probes_weights_differ_at_end']:13d}", flush=True)
        print(f"{'':>24s}({time.time() - t_start:.0f} s)", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(doc, f)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
