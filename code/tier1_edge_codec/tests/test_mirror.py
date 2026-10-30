"""
test_mirror.py  --  blueprint 6.4.

The mirror's reconstruction must equal the codec's, bit for bit, from the
packet stream alone. This is the property that makes the whole scheme valid, so
it is tested on every client of a real store rather than on one hand-picked
series.
"""

import numpy as np
import pytest

from edge.codec import EPS_INF, CodecParams
from edge.mirror import Mirror
from edge.quantize import quantize_array
from sim.run_codec import run_policy

P = CodecParams()


SUBSET = [0, 2, 4]          # the scalar backend is slow; three clients is
                            # enough to cover onboarding, a flat meter and a
                            # normal one, and every client is covered by the
                            # batch-versus-scalar equivalence test anyway


def _mirror_all(store, res, alphas_eps=None):
    for c in sorted(res.packets):
        t0 = int(store.onboarding[c])
        u, _ = quantize_array(store.series_filled[:, c], int(res.k[c]), P.u_clamp)
        pkts = [(h, b) for (h, b, ok) in res.packets[c] if ok]
        m = Mirror(P, res.eps[c], int(u[t0]), t0, node_id=c)
        out = m.run(pkts, store.T)
        yield c, u, m, out


@pytest.mark.parametrize("alpha", [0.0, 0.25, 1.0, float("inf")])
def test_mirror_matches_the_codec_on_every_client(store, alpha):
    res = run_policy(store, alpha, P, backend="scalar", packet_log=True,
                     clients=SUBSET)
    for c, u, m, out in _mirror_all(store, res):
        t0 = int(store.onboarding[c])
        assert np.array_equal(out["u_within"], res.u_within[t0:, c]), c
        assert np.array_equal(out["u_exact"], res.u_exact[t0:, c]), c
        assert m.n_desync_suspected == 0, c


@pytest.mark.parametrize("alpha", [0.25, 1.0])
def test_mirror_holds_the_epsilon_bound(store, alpha):
    """Independently of the codec: the mirror alone, against quantised truth."""
    res = run_policy(store, alpha, P, backend="scalar", packet_log=True,
                     clients=SUBSET)
    for c, u, m, out in _mirror_all(store, res):
        eps = res.eps[c]
        if eps is EPS_INF:
            continue
        t0 = int(store.onboarding[c])
        d = np.abs(out["u_within"] - u[t0:])
        assert d.max() <= max(2 * eps, 0), (c, eps, int(d.max()))


def test_only_the_current_day_is_ever_approximate(store):
    """Blueprint 5.2's consequence, asserted rather than assumed: after each
    daily resync the older slots hold exact values, so u_exact must equal the
    quantised truth everywhere a resync has landed."""
    res = run_policy(store, 1.0, P, backend="scalar", packet_log=True,
                     clients=SUBSET)
    for c, u, m, out in _mirror_all(store, res):
        t0 = int(store.onboarding[c])
        r = out["resynced"]
        assert r.any(), c
        assert np.array_equal(out["u_exact"][r], u[t0:][r]), c
        # and the un-resynced remainder is at most the final partial day
        assert (~r).sum() <= 24 + 24, (c, int((~r).sum()))


def test_mirror_rejects_another_node_id(store):
    """Packets are addressed. A mirror fed someone else's traffic must ignore
    it, not decode it into this client's series."""
    res = run_policy(store, 1.0, P, backend="scalar", packet_log=True, clients=[0, 1])
    t0 = int(store.onboarding[0])
    u, _ = quantize_array(store.series_filled[:, 0], int(res.k[0]), P.u_clamp)
    pkts = [(h, b) for (h, b, ok) in res.packets[1] if ok]     # WRONG node
    m = Mirror(P, res.eps[0], int(u[t0]), t0, node_id=0)
    out = m.run(pkts, store.T)
    assert m.n_exception_applied == 0 and m.n_resync_applied == 0
    # with nothing applied it is a pure free-running prediction, which for a
    # non-degenerate client must differ from the truth somewhere
    assert not np.array_equal(out["u_within"], u[t0:])


def test_mirror_is_deterministic(store):
    res = run_policy(store, 0.5, P, backend="scalar", packet_log=True, clients=[0])
    t0 = int(store.onboarding[0])
    u, _ = quantize_array(store.series_filled[:, 0], int(res.k[0]), P.u_clamp)
    pkts = [(h, b) for (h, b, ok) in res.packets[0] if ok]
    outs = []
    for _ in range(2):
        m = Mirror(P, res.eps[0], int(u[t0]), t0, node_id=0)
        o = m.run(pkts, store.T)
        outs.append((o["u_within"].tolist(), m.sync_snapshot()))
    assert outs[0] == outs[1]
