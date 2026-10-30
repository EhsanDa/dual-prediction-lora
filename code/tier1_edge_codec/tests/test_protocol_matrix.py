"""
test_protocol_matrix.py  --  the frame-loss question, pinned so it cannot be
argued a fourth time.

What a lost RESYNC or a lost ACK does depends on WHICH node behaviour and WHICH
gateway behaviour are paired. The repository reached three different answers
between 2026-09-14 and 2026-09-28 because it never said which pairing it was
measuring. docs/findings/2026-09-28-frame-loss-settled.md is the write-up; these
tests are its load-bearing claims. If one ever fails, the finding is wrong --
read it before changing either.

Two groups:

  * EQUIVALENCE. eval/protocol_matrix.Link reproduces the existing code paths
    exactly wherever they model the same thing -- sim/run_codec.py + edge/mirror.py
    for the gated protocol (Table I's path), and the old single-loss probe for the
    ungated node. So its verdicts on the other pairings can be trusted.
  * MECHANISM. The verdict for each pairing, on a small synthetic series.
"""
import numpy as np
import pytest

from edge.codec import CodecParams
from edge.mirror import Mirror
from edge.protocol import TYPE_RESYNC
from eval.protocol_matrix import Link, run_link
from sim.run_codec import _run_scalar, choose_scales, eps_for_alpha

W0 = (1 << 20, 0, 0, 0, 0, 0, 0)            # persistence start: weights move
EPS, T = 60, 24 * 30


def series(t0=0, seed=11):
    """Deterministic load with diurnal and weekly structure plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(t0 + T)
    y = (3000 + 900 * np.sin(2 * np.pi * t / 24) + 400 * np.sin(2 * np.pi * t / 168)
         + rng.normal(0, 120, t.size))
    return np.clip(np.round(y), 0, 32000).astype(np.int64)


def run(pair, event=None, t0=0, t_end=None):
    u = series(t0)
    t_end = t_end or len(u)
    link = Link(*pair, W0, 5, EPS, int(u[t0]), t0)
    valid = np.ones(len(u), bool)
    cl = np.zeros(len(u), bool)
    viol, sync, gap = run_link(link, u, cl, valid, t0, t_end, event=event)
    return viol, sync, gap


BOUNDARY = 24 * 4                            # a day boundary well after warm-up


# ------------------------------------------------------------ the mechanism --
def test_lossless_every_consistent_pairing_stays_in_sync():
    for pair in [("gated", "legacy"), ("gated", "adopt_before"),
                 ("ungated", "legacy"), ("ungated", "adopt_after"),
                 ("sync_safe", "sync_safe")]:
        viol, sync, _ = run(pair)
        assert sync.all() and not viol.any(), pair


def test_paper_protocol_lost_resync_is_symmetric():
    """The paper's sentence. Gated node, legacy gateway: no frame, no ACK,
    both keep the old weights and the old buffer -- in sync at every hour."""
    viol, sync, gap = run(("gated", "legacy"), event=("resync", BOUNDARY))
    assert sync.all() and not viol.any() and (gap == 0).all()


def test_paper_protocol_lost_ack_desynchronises_for_good():
    """The failure the reviewers name. The gateway updates, the node does not,
    and legacy never reads a weight field, so nothing brings them back."""
    viol, sync, gap = run(("gated", "legacy"), event=("ack", BOUNDARY))
    assert not sync[BOUNDARY:].any(), "should never re-synchronise"
    assert gap[-1] > 0


def test_adopt_before_repairs_a_lost_ack_but_not_within_one_interval():
    """Option 1 in the order a gated node allows: the frame carries the
    node's PRE-update weights, the gateway takes them and then updates.

    It repairs -- but NOT at the next resync. The un-ACKed day stays in the ring
    as reconstructions on the node and as truth on the gateway, and the next
    days' 24 NLMS updates read back as far as lag 168 into it, so each update
    re-opens a small gap until that day ages out: 24 + 168 hours plus the day in
    progress. Measured on this fixture: last weight difference at +191 h."""
    viol, sync, gap = run(("gated", "adopt_before"), event=("ack", BOUNDARY))
    nxt = BOUNDARY + 24
    assert gap[BOUNDARY] > 0
    assert 0 < gap[nxt] < gap[BOUNDARY], "shrinks at the next resync, not zero"
    settled = BOUNDARY + 24 + 168 + 24
    assert (gap[settled:] == 0).all() and sync[settled:].all()


def test_adopt_after_is_wrong_for_the_papers_node():
    """CodecParams.adopt_resync_weights as coded (update, THEN adopt) takes the
    gated node's pre-update weights and so undoes its own update every day --
    it breaks synchronisation on a LOSSLESS channel. It is only valid for an
    ungated node, whose frame carries post-update weights."""
    viol, sync, _ = run(("gated", "adopt_after"))
    assert not sync.all()


def test_ungated_node_lost_resync_desynchronises_for_good():
    """What the 2026-09-14 probe measured, and what firmware/node/main.ino
    does against a legacy gateway."""
    viol, sync, gap = run(("ungated", "legacy"), event=("resync", BOUNDARY))
    assert not sync[BOUNDARY:].any()
    assert gap[-1] > 0


def test_ungated_node_with_adopt_after_repairs_a_lost_resync():
    viol, sync, gap = run(("ungated", "adopt_after"), event=("resync", BOUNDARY))
    assert (gap[BOUNDARY + 24:] == 0).all()


@pytest.mark.parametrize("kind", ["exception", "resync", "ack"])
def test_sync_safe_recovers_from_any_single_loss(kind):
    ev_hour = BOUNDARY if kind != "exception" else BOUNDARY + 5
    viol, sync, gap = run(("sync_safe", "sync_safe"), event=(kind, ev_hour))
    assert (gap[BOUNDARY + 24:] == 0).all()
    assert sync[BOUNDARY + 24:].all()


@pytest.mark.parametrize("pair", [("gated", "legacy"), ("sync_safe", "sync_safe")])
def test_a_lost_exception_is_repaired_at_the_next_delivered_resync(pair):
    viol, sync, gap = run(pair, event=("exception", BOUNDARY + 5))
    assert sync[BOUNDARY + 24:].all() and (gap == 0).all()


def test_sync_safe_mirror_follows_a_node_commissioned_mid_day():
    """The replay snapshot must sit at a calendar-day start. Before the fix in
    CodecState.step, a meter commissioned at hour 160 (client 118 on ECL) left
    its sync_safe mirror on a lossless channel."""
    for t0 in (0, 5, 16, 23):
        _, sync, _ = run(("sync_safe", "sync_safe"), t0=t0)
        assert sync.all(), t0


# ------------------------------------------------------------ equivalence ----
SUBSET = [0, 2, 4]


def _old_path(store, c, p, applied_off=None, delivered_off=None, drop_log=None):
    k, _ = choose_scales(store)
    eps, _ = eps_for_alpha(store, k, 1.0)
    from edge.quantize import quantize_array
    M, nd = store.M, (store.T + 23) // 24
    u = np.zeros((store.T, M), np.int64)
    cl = np.zeros((store.T, M), bool)
    u[:, c], cl[:, c] = quantize_array(store.series_filled[:, c], int(k[c]),
                                       p.u_clamp)
    t0 = int(store.onboarding[c])
    u0 = np.zeros(M, np.int64)
    u0[c] = u[t0, c]
    dl = np.ones((nd, M), bool)
    ap = np.ones((nd, M), bool)
    if applied_off is not None:
        ap[applied_off, c] = False
    if delivered_off is not None:
        dl[delivered_off, c] = False
    o = _run_scalar(p, u, eps, u0, store.onboarding.astype(np.int64), store.T,
                    [c], dl, ap, None, cl)
    pk = [(h, b) for (h, b, ok) in o["packets"][c]
          if ok and not (h == drop_log and (b[0] >> 4) == TYPE_RESYNC)]
    m = Mirror(p, eps[c], int(u[t0, c]), t0, node_id=c, strict=False)
    return m.run(pk, store.T)["u_within"], u[:, c], cl[:, c], eps[c], t0


def _link_path(store, c, pair, u, cl, eps, t0, event):
    link = Link(*pair, CodecParams().w_init, 5, eps, int(u[t0]), t0)
    out = []
    for t in range(t0, store.T):
        if link.node.is_boundary(t):
            link.boundary(t, lose_resync=(event == ("resync", t)),
                          lose_ack=(event == ("ack", t)))
        out.append(link.hour(t, u[t], bool(cl[t]))[0])
    return np.asarray(out)


@pytest.mark.parametrize("case", ["lost_resync", "lost_ack", "probe"])
def test_link_reproduces_the_existing_code_paths(store, case):
    """Table I's path (run_codec resync/ack loss + Mirror) for the gated node,
    and the 2026-09-14 probe (node applies, frame dropped from the mirror's
    log) for the ungated node -- hour for hour."""
    p = CodecParams(k_mu=5)
    for c in SUBSET:
        t0 = int(store.onboarding[c])
        d = (t0 // 24) + 3
        t = d * 24
        if case == "lost_resync":
            old = _old_path(store, c, p, applied_off=d, delivered_off=d)
            pair, ev = ("gated", "legacy"), ("resync", t)
        elif case == "lost_ack":
            old = _old_path(store, c, p, applied_off=d)
            pair, ev = ("gated", "legacy"), ("ack", t)
        else:
            old = _old_path(store, c, p, drop_log=t)
            pair, ev = ("ungated", "legacy"), ("resync", t)
        rec, u, cl, eps, t0 = old
        if eps is None:
            continue
        assert np.array_equal(_link_path(store, c, pair, u, cl, eps, t0, ev),
                              rec), (case, c)
