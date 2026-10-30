"""
periodic_predict.py  --  fixed periodic refresh WITH predictor imputation.
Conference revision, request F (2026-09-29).

The reviewers asked for the periodic comparison to be decomposed: WHEN to
transmit (event-triggered vs a fixed schedule) crossed with HOW the gateway
fills the hours it has not been told (the shared predictor vs a zero-order
hold). Three cells existed:

    event-triggered x predictor   the proposed scheme (policy C)
    event-triggered x ZOH         the persistence deadband (policy A)
    periodic        x ZOH         sim/periodic.py

This module is the fourth: the SAME schedule and the SAME packets as
sim/periodic.py -- every `period` hours the node sends the exact values since
the last refresh -- but between refreshes the gateway imputes with the
predictor of the proposed scheme instead of holding the last value.

THE GATEWAY'S PREDICTOR
=======================
edge/codec.py's CodecState with the reported configuration: taps
(1, 2, 3, 24, 25, 168), the globally pre-trained weights, NLMS at k_mu = 5 in
legacy mode, i.e. 24 updates per day on the TRUE day. The gateway can run that
update itself because a periodic refresh delivers every true value; the node
needs no predictor at all in this cell. On a refresh hour the delivered hours
overwrite the ring (they were predictions) and the prediction is recomputed;
on any other hour the ring takes the recursive prediction, so an hour k past
the last refresh is a k-step-ahead forecast from exact data.

The reconstruction is assembled for the forecaster with
sim/reconstruct.py::assemble_window_horizon and refresh_horizon(period): hours
up to the last refresh are exact, later hours are these predictions.
"""
import os

import numpy as np

from edge.codec import EPS_INF, CodecParams, CodecState
from sim.periodic import run_periodic
from sim.run_codec import PolicyResult

_G = {}


def _client_within(c):
    g = _G
    u = g["u"][:, c]
    t0 = int(g["t_start"][c])
    T, period, p = g["T"], g["period"], g["params"]
    out = np.zeros(T, np.int64)
    st = CodecState(p, EPS_INF, u0=int(u[t0]), t_start=t0)
    for t in range(t0, T):
        if st.is_boundary(t):
            # legacy: overwrite the last 24 slots with the true day and run the
            # 24 NLMS updates on it -- the gateway holds every true value.
            st.day_boundary([int(x) for x in u[t - 24:t]])
        if t % period == 0:
            # the refresh at t delivers (t - period, t]: the hours before t in
            # that span were predictions in the ring; make them exact.
            for j in range(period - 1):
                hour = t - 1 - j
                if hour >= t0:
                    st._set_hist(j, int(u[hour]))
            st.u_hat = st.predict()
            r = st.step(int(u[t]), force_tx=True)
        else:
            r = st.step(int(u[t]), force_tx=False)
        out[t] = r.u_recon
    return c, out


def run_periodic_predict(store, period: int, w_init, k_mu=5, workers=None):
    """Same schedule, packets, bytes and exact history as
    sim.periodic.run_periodic(store, period); u_within replaced by the
    gateway's predictor imputation. Returns a PolicyResult."""
    base = run_periodic(store, period)
    params = CodecParams(k_mu=k_mu).with_w_init(tuple(int(x) for x in w_init))
    _G.update(u=base.u_q, t_start=store.onboarding.astype(np.int64),
              T=store.T, period=period, params=params)
    within = np.zeros_like(base.u_within)
    clients = list(range(store.M))
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    if workers <= 1:
        results = map(_client_within, clients)
    else:
        import multiprocessing as mp
        pool = mp.get_context("fork").Pool(workers)
        results = pool.imap_unordered(_client_within, clients, chunksize=4)
    for c, col in results:
        within[:, c] = col
    if workers > 1:
        pool.close()
        pool.join()
    within = np.where(base.active, within, 0)
    notes = dict(base.notes)
    notes.update({"policy": "fixed periodic refresh, predictor imputation",
                  "imputation": "global pre-trained NLMS predictor, legacy "
                                "daily update on the true day"})
    return PolicyResult(
        alpha=base.alpha, k=base.k, eps=base.eps,
        eps_degenerate=base.eps_degenerate, u_exact=base.u_exact, u_q=base.u_q,
        u_within=within, e=np.where(base.active, base.u_q - within, 0),
        tx=base.tx, event=base.event, resynced=base.resynced,
        eps_eff=base.eps_eff, active=base.active, clamped=base.clamped,
        resync_bytes=base.resync_bytes, packets={}, notes=notes)


def label(period: int) -> str:
    return f"periodic-pred-{period}h"
