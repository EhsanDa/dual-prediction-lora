"""Does the node/gateway weight gap accumulate after lost ACKs? (2026-10-13)

In the paper's protocol (gated node, legacy gateway) a lost acknowledgement
leaves the gateway updated and the node not; legacy never exchanges weights,
so the two are never again byte-identical (docs/findings/2026-09-28-frame-
loss-settled.md). The author asked whether that gap then grows and piles up.

Both parties keep updating on the same 24 true values every day, so each
update pulls both weight vectors toward the same target; only fixed-point
rounding stops the gap reaching exactly zero. This measures it, over each
client's whole life, in two regimes:

  single  one lost ACK at the 10th day boundary, every other frame delivered;
  rate    every ACK lost independently with probability --ack_loss.

Per client: the largest weight gap (Q20 LSB, max over the 7 weights) in each
quarter of the client's life, the final gap, and the violated hours in each
quarter. No trend across quarters is the "does not accumulate" evidence.

    python -m eval.ack_drift --data_dir ../tier2_data
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

from edge.codec import CodecParams                                # noqa: E402
from edge.quantize import quantize_array                          # noqa: E402
from eval.protocol_matrix import EPS_INF, Link                    # noqa: E402
from sim.run_codec import Store, choose_scales, eps_for_alpha     # noqa: E402


def run_client(store, c, u, cl, eps, w, k_mu, mode, ack_loss, rng):
    t0 = int(store.onboarding[c])
    valid = ~np.asarray(store.missing[:, c], bool)
    L = Link("gated", "legacy", w, k_mu, eps, int(u[t0]), t0)
    span = store.T - t0
    nb, lost, gaps, gap_t = 0, 0, [], []
    viol_q = np.zeros(4, int)
    for t in range(t0, store.T):
        if L.node.is_boundary(t):
            nb += 1
            lose = (nb == 10) if mode == "single" else bool(rng.random() < ack_loss)
            lost += lose
            L.boundary(t, lose_ack=lose)
            gaps.append(L.weight_gap())
            gap_t.append(t - t0)
        rec, e, _ = L.hour(t, u[t], bool(cl[t]))
        if e is not EPS_INF and abs(int(rec) - int(u[t])) > e and valid[t]:
            viol_q[min(3, 4 * (t - t0) // span)] += 1
    g, gt = np.array(gaps), np.array(gap_t)
    q = [g[(gt >= i * span / 4) & (gt < (i + 1) * span / 4)] for i in range(4)]
    return {"client": int(c), "acks_lost": int(lost), "boundaries": int(nb),
            "gap_max_by_quarter_lsb": [int(x.max()) if x.size else 0 for x in q],
            "gap_final_lsb": int(g[-1]) if g.size else 0,
            "violated_hours_by_quarter": viol_q.tolist(),
            "in_sync_at_end": bool(L.in_sync())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../tier2_data")
    ap.add_argument("--pretrained", default=paths.inp("edge_pretrained.json"))
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--k_mu", type=int, default=5)
    ap.add_argument("--client_stride", type=int, default=16)
    ap.add_argument("--ack_loss", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=paths.result(os.path.join("revision",
                                                               "ack_drift.json")))
    a = ap.parse_args(argv)
    store = Store.load(a.data_dir)
    k, _ = choose_scales(store)
    eps_all, _ = eps_for_alpha(store, k, a.alpha)
    w = tuple(int(x) for x in json.load(open(a.pretrained))["w_global_q"])
    p = CodecParams(k_mu=a.k_mu)
    clients = [c for c in range(0, store.M, a.client_stride)
               if eps_all[c] is not EPS_INF]
    out = {"what": "weight gap and violations after lost ACKs, gated node + "
                   "legacy gateway, whole life", "alpha": a.alpha, "k_mu": a.k_mu,
           "ack_loss": a.ack_loss, "seed": a.seed, "clients": clients,
           "weight_lsb": "Q20: 1 LSB = 2**-20 of a weight", "single": [],
           "rate": []}
    rng = np.random.default_rng(a.seed)
    for c in clients:
        u, cl = quantize_array(store.series_filled[:, c], int(k[c]), p.u_clamp)
        for mode in ("single", "rate"):
            r = run_client(store, c, u, cl, eps_all[c], w, a.k_mu, mode,
                           a.ack_loss, rng)
            out[mode].append(r)
            print(mode, r, flush=True)
    for mode in ("single", "rate"):
        rows = out[mode]
        gq = np.array([r["gap_max_by_quarter_lsb"] for r in rows], float)
        vq = np.array([r["violated_hours_by_quarter"] for r in rows], float)
        out[f"{mode}_summary"] = {
            "median_gap_max_by_quarter_lsb": np.median(gq, axis=0).tolist(),
            "median_violated_hours_by_quarter": np.median(vq, axis=0).tolist(),
            "median_gap_final_lsb": float(np.median([r["gap_final_lsb"] for r in rows])),
            "clients_in_sync_at_end": int(sum(r["in_sync_at_end"] for r in rows)),
            "n_clients": len(rows)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({m: out[f"{m}_summary"] for m in ("single", "rate")}, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
