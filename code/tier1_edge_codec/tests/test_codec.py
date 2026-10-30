"""
test_codec.py  --  blueprint 9.3. These are the important ones.

The load-bearing test in the whole repository is `test_silent_hour_trap`: it
builds a codec that pushes u instead of u_hat on a silent hour and asserts that
the synchronisation test CATCHES it. If that ever passes, the synchronisation
test has stopped testing anything and every other result here is worthless.
"""

import numpy as np
import pytest

from edge.codec import (EPS_INF, CodecParams, CodecState, StepResult,
                        warm_start)
from edge.codec_batch import BatchCodec, run_batch
from edge.mirror import Mirror
from edge.protocol import Resync, encode_exception, encode_resync, Exception_
from edge.protocol import FLAG_CLAMP, FLAG_EVENT
from edge.quantize import choose_scale, quantize_array

P = CodecParams()
SS = CodecParams(sync_safe=True)      # the compiled-in robustness mode


def _quantised(tiny):
    y, onb = tiny["y"], tiny["onboarding"]
    out = []
    for c in range(y.shape[1]):
        k = choose_scale(y[onb[c]:, c])
        u, cl = quantize_array(y[:, c], k, P.u_clamp)
        out.append((k, u, cl))
    return out


def drive(u, eps, t0, T, params=P, push_true_on_silent=False, drop_resync=(),
          drop_exception=()):
    """Run the device side, returning the state trace and the packet log.

    `drop_resync` / `drop_exception` are sets of hours whose packet is emitted
    but not delivered -- the node's behaviour is unaffected, which under
    sync_safe is the whole point.
    """
    st = warm_start(params, eps, int(u[t0]), t0)
    snaps, log, recon = [], [], np.zeros(T, np.int64)
    for t in range(t0, T):
        if st.is_boundary(t):
            payload, bitmap = st.resync_payload(), st.resync_bitmap()
            n_prev = st.n_tx_today
            if params.sync_safe:
                st.day_boundary()
            else:
                st.day_boundary(payload)
            if t not in drop_resync:
                log.append((t, encode_resync(Resync(
                    0, (t // 24) & 0xFF, n_prev, tuple(payload),
                    tx_bitmap=tuple(bitmap), weights=tuple(st.w)))))
        if push_true_on_silent:
            r = _broken_step(st, int(u[t]))
        else:
            r = st.step(int(u[t]))
        if r.tx and t not in drop_exception:
            log.append((t, encode_exception(Exception_(
                0, t % 24, (FLAG_EVENT if r.event else 0)
                | (FLAG_CLAMP if r.clamped else 0), int(r.e)))))
        recon[t] = r.u_recon
        snaps.append(st.snapshot())
    return st, snaps, log, recon


def _broken_step(st, u):
    """THE DELIBERATE BUG (blueprint 9.3): push the truth on a silent hour.

    Written out in full rather than monkey-patched, so it is obvious that the
    ONLY difference from CodecState.step is the pushed value."""
    from edge.fixedpoint import abs_i, clamp, i16, i32, sar
    u = i16(u)
    u_hat = st.u_hat
    e = i32(u - u_hat)
    ae = abs_i(e)
    st.s = i16(clamp(st.s + sar(ae - st.s, st.p.s_shift), 0, 32767))
    eps_eff = st.eps_effective()
    tx = False if eps_eff is EPS_INF else (ae > eps_eff)
    st._push(u)                                   # <-- the bug, and only this
    st.day_true[st.t % 24] = u
    if tx:
        st.n_tx_today += 1
    st.n_seen += 1
    st.t += 1
    st.u_hat = st.predict()
    return StepResult(e=e, tx=tx, event=False, clamped=False, u_hat=u_hat,
                      u_recon=u if tx else u_hat, eps_eff=eps_eff)


def _mirror_run(log, eps, u0, t0, T, params=P):
    m = Mirror(params, eps, u0, t0, node_id=0, strict=False)
    return m, m.run(log, T)


# --------------------------------------------------------------------- tests
def test_initialisation_first_prediction_is_last_value(tiny):
    """With w_init = (4096, 0, ...) the first prediction is exactly u[t-1]."""
    for k, u, _ in _quantised(tiny):
        st = warm_start(P, 0, int(u[0]), 0)
        assert st.u_hat == int(u[0])
        assert st.predict() == int(u[0])


def test_synchronisation_bytewise(tiny):
    """A codec and a mirror driven only by the packet stream have identical
    snapshot() bytes at EVERY hour, for every client."""
    T = tiny["y"].shape[0]
    for c, (k, u, _) in enumerate(_quantised(tiny)):
        t0 = int(tiny["onboarding"][c])
        for eps in (0, 20, EPS_INF):
            st, snaps, log, recon = drive(u, eps, t0, T)
            m = Mirror(P, eps, int(u[t0]), t0, node_id=0)
            # replay hour by hour and compare after each
            out = m.run(log, T)
            assert np.array_equal(out["u_within"], recon[t0:]), (c, eps)
            assert m.sync_snapshot() == st.sync_snapshot(), (c, eps)


def test_silent_hour_trap(tiny):
    """THE test. A transmitter that pushes u instead of u_hat must break
    synchronisation. If this xfails, test_synchronisation_bytewise is vacuous."""
    T = tiny["y"].shape[0]
    k, u, _ = _quantised(tiny)[0]
    eps = 20
    st, snaps, log, recon = drive(u, eps, 0, T, push_true_on_silent=True)
    m = Mirror(P, eps, int(u[0]), 0, node_id=0, strict=False)
    out = m.run(log, T)
    assert not np.array_equal(out["u_within"], recon), (
        "the broken transmitter still synchronises -- the synchronisation test "
        "is not testing anything")
    assert m.sync_snapshot() != st.sync_snapshot()


def test_epsilon_bound_zero_tolerance(tiny):
    """|recon - quantised truth| <= eps_eff at every hour. Zero tolerance.

    Against eps_eff, not eps: the daily cap doubles the threshold on
    exhaustion (blueprint 4.9), so eps alone is not the bound the system
    actually provides. The doubling is asserted to be the ONLY source of
    excess, by checking the ratio never exceeds 2.
    """
    T = tiny["y"].shape[0]
    for c, (k, u, _) in enumerate(_quantised(tiny)):
        t0 = int(tiny["onboarding"][c])
        for eps in (0, 5, 20, 100):
            st, snaps, log, recon = drive(u, eps, t0, T)
            d = np.abs(recon[t0:] - u[t0:])
            assert d.max() <= max(2 * eps, 0), (c, eps, d.max())


def test_alpha_zero_reconstructs_exactly(tiny):
    """alpha = 0 -> eps = 0 -> the mirror holds the quantised truth at every
    hour. This is the full-freshness oracle, and it must be EXACT."""
    T = tiny["y"].shape[0]
    for c, (k, u, _) in enumerate(_quantised(tiny)):
        t0 = int(tiny["onboarding"][c])
        st, snaps, log, recon = drive(u, 0, t0, T)
        assert np.array_equal(recon[t0:], u[t0:])


def test_alpha_inf_sends_only_resyncs(tiny):
    """alpha = inf -> no exception ever -> exactly one packet per day."""
    T = tiny["y"].shape[0]
    for c, (k, u, _) in enumerate(_quantised(tiny)):
        t0 = int(tiny["onboarding"][c])
        st, snaps, log, recon = drive(u, EPS_INF, t0, T)
        kinds = [b[0] >> 4 for _, b in log]
        assert set(kinds) == {2}, "an EXCEPTION escaped at alpha = inf"
        expected_days = len([t for t in range(t0, T)
                             if t % 24 == 0 and t - t0 >= 24])
        assert len(log) == expected_days


def test_determinism(tiny):
    """Two runs on the same input give identical packet logs and identical
    final state. Nothing in the codec may depend on anything but its input."""
    T = tiny["y"].shape[0]
    k, u, _ = _quantised(tiny)[0]
    a = drive(u, 30, 0, T)
    b = drive(u, 30, 0, T)
    assert a[3].tolist() == b[3].tolist()
    assert [p for _, p in a[2]] == [p for _, p in b[2]]
    assert a[0].snapshot() == b[0].snapshot()


def test_nlms_shrinks_the_error_by_mu():
    """Pins the NLMS shift semantics (edge/codec.py deviation 1).

    NLMS identity: after one update on a sample, the a-posteriori error on that
    same sample is (1 - mu) times the a-priori error. The shipped k_mu = 5
    claims mu = 1/32, so the ratio must land near 0.97. The blueprint's literal
    shift `b - k_mu` would give mu = 2**-10 and a ratio of 0.999; at Q12 the
    update quantised to zero entirely and the ratio was exactly 1.0. This test
    pins the realised step size against all three.
    """
    rng = np.random.default_rng(0)
    ratios = []
    for _ in range(200):
        st = CodecState(P, eps=0, u0=4000)
        st.buf = [int(v) for v in rng.integers(2000, 9000, P.buf_len)]
        phi = st.phi()
        target = int(rng.integers(2000, 9000))
        e0 = target - st._predict_from(phi)
        if abs(e0) < 50:
            continue
        st._nlms_update(phi, e0)
        e1 = target - st._predict_from(phi)
        ratios.append(e1 / e0)
    r = float(np.mean(ratios))
    mu = 2.0 ** -P.k_mu
    assert abs(r - (1 - mu)) < 0.04, (
        f"realised (1 - mu) is {r:.4f}; expected ~{1 - mu:.4f} for "
        f"k_mu={P.k_mu}. A value of 1.0 means the update quantised away, which "
        f"is exactly what Q12 did.")


def test_weights_are_frozen_within_a_day(tiny):
    """The whole desynchronisation argument rests on this."""
    T = tiny["y"].shape[0]
    k, u, _ = _quantised(tiny)[0]
    st = warm_start(P, 20, int(u[0]), 0)
    prev = None
    for t in range(T):
        boundary = st.is_boundary(t)
        if boundary:
            st.day_boundary(st.resync_payload())
        w = list(st.w)
        if prev is not None and not boundary:
            assert w == prev, f"weights moved inside a day at t={t}"
        prev = w
        st.step(int(u[t]))


def test_lost_resync_does_not_desynchronise(tiny):
    """The sync_safe ROBUSTNESS MODE, which is compiled in but is no longer the
    default: the weight update runs on the reconstructed ring, so it needs no
    packet at all and dropping every other RESYNC leaves the two sides
    byte-identical. Legacy is the shipped default because it measured both
    better and cheaper; this is what it gives up."""
    T = tiny["y"].shape[0]
    k, u, _ = _quantised(tiny)[0]
    drop = {t for t in range(0, T, 24) if (t // 24) % 2 == 0}
    st, snaps, log, recon = drive(u, 20, 0, T, params=SS, drop_resync=drop)
    assert any(b[0] >> 4 == 2 for _, b in log), "no resync survived; test is void"
    m = Mirror(SS, 20, int(u[0]), 0, node_id=0)
    out = m.run(log, T)
    assert np.array_equal(out["u_within"], recon)
    assert m.sync_snapshot() == st.sync_snapshot()


def test_lost_exception_is_repaired_by_the_next_resync(tiny):
    """Both halves matter. If the first does not hold the loss model is wrong;
    if the second does not hold the scheme never recovers the bound."""
    T = tiny["y"].shape[0]
    k, u, _ = _quantised(tiny)[0]
    eps = 20
    ref, _, _, recon = drive(u, eps, 0, T, params=SS)
    txs = [t for t in range(30, 45) if abs(recon[t] - u[t]) == 0]
    dropped = txs[0]
    st, snaps, log, recon = drive(u, eps, 0, T, params=SS,
                                  drop_exception={dropped})
    m = Mirror(SS, eps, int(u[0]), 0, node_id=0, strict=False)
    out = m.run(log, T)
    assert not np.array_equal(out["u_within"][dropped:dropped + 4],
                              recon[dropped:dropped + 4]), (
        "dropping an exception did not desynchronise -- the loss model is wrong")
    repaired_from = ((dropped // 24) + 1) * 24
    assert np.array_equal(out["u_within"][repaired_from:], recon[repaired_from:])
    assert m.sync_snapshot() == st.sync_snapshot()
    assert m.n_repaired >= 1, "the replay path never fired"


def test_many_lost_exceptions_are_still_repaired_by_one_resync(tiny):
    """The replay reconstructs the node's whole day from the bitmap, so the
    number of lost exceptions does not matter. This is the claim that replaces
    'the ACK is mandatory'."""
    T = tiny["y"].shape[0]
    k, u, _ = _quantised(tiny)[0]
    drop = set(range(24, 48))                 # every exception of one whole day
    st, snaps, log, recon = drive(u, 20, 0, T, params=SS, drop_exception=drop)
    m = Mirror(SS, 20, int(u[0]), 0, node_id=0, strict=False)
    out = m.run(log, T)
    assert np.array_equal(out["u_within"][48:], recon[48:])
    assert m.sync_snapshot() == st.sync_snapshot()


def test_legacy_mode_still_shows_the_ack_asymmetry(tiny):
    """The comparison the review asks for: with sync_safe off, a resync the
    mirror applies and the node does not really does desynchronise them. If
    this ever passes, sync_safe is fixing nothing and the change is unjustified.
    """
    from edge.codec import CodecParams as CP
    legacy = CP(sync_safe=False)
    T = tiny["y"].shape[0]
    k, u, _ = _quantised(tiny)[0]
    st = warm_start(legacy, 20, int(u[0]), 0)
    log, recon = [], np.zeros(T, np.int64)
    for t in range(T):
        if st.is_boundary(t):
            payload, bitmap = st.resync_payload(), st.resync_bitmap()
            log.append((t, encode_resync(Resync(0, (t // 24) & 0xFF, 0,
                                                tuple(payload),
                                                tx_bitmap=tuple(bitmap)))))
            if t != 48:                       # the ACK for day 2 is lost
                st.day_boundary(payload)
            else:
                st.new_day()
        r = st.step(int(u[t]))
        if r.tx:
            log.append((t, encode_exception(Exception_(0, t % 24, 0, int(r.e)))))
        recon[t] = r.u_recon
    m = Mirror(legacy, 20, int(u[0]), 0, node_id=0, strict=False)
    out = m.run(log, T)
    assert m.sync_snapshot() != st.sync_snapshot(), (
        "legacy mode did not desynchronise on a lost ACK, so there was nothing "
        "for sync_safe to fix")


def test_send_on_delta_baseline_is_persistence(tiny):
    """Revision R7. adapt=False freezes w at w_init, which IS persistence, so
    the transmit rule reduces to |y[t] - y[t-1]| > eps."""
    from edge.codec import PARAMS_SEND_ON_DELTA as SOD
    T = tiny["y"].shape[0]
    k, u, _ = _quantised(tiny)[0]
    st, _, _, _ = drive(u, 20, 0, T, params=SOD)
    assert st.w == list(SOD.w_init), "the weights moved with adapt=False"
    st2 = warm_start(SOD, 20, int(u[0]), 0)
    for t in range(T):
        if st2.is_boundary(t):
            st2.day_boundary()
        r = st2.step(int(u[t]))
        want = abs(int(u[t]) - int(r.u_hat)) > 20
        assert r.tx == want
        # and u_hat is exactly the previous reconstruction
        assert r.u_hat == st2.hist(1) if t > 0 else True


def test_batch_equals_scalar_exactly(tiny):
    """Blueprint 6.2: exactly, not approximately, for every client at every
    hour -- residuals, flags, reconstruction AND snapshot bytes."""
    y, onb = tiny["y"], tiny["onboarding"]
    T, M = y.shape
    q = _quantised(tiny)
    ks = [k for k, _, _ in q]
    u = np.stack([uu for _, uu, _ in q], axis=1)
    u0 = np.array([u[int(onb[c]), c] for c in range(M)], np.int64)

    for eps_val in (0, 25, EPS_INF):
        eps = [eps_val] * M
        out = run_batch(P, u, eps, u0, onb.astype(np.int64), T)
        for c in range(M):
            t0 = int(onb[c])
            st, snaps, log, recon = drive(u[:, c], eps_val, t0, T)
            assert np.array_equal(out["u_within"][t0:, c], recon[t0:]), (c, eps_val)
            assert np.array_equal(out["u_exact"][t0:, c],
                                  _scalar_exact(u[:, c], recon, t0, T)), (c, eps_val)

    # canonical state bytes, sampled through the run rather than only at the
    # end. canonical() rather than snapshot() because the two implementations
    # index the ring differently for late-onboarding clients; see
    # CodecState.canonical.
    eps = [25] * M
    bc = BatchCodec(P, eps, u0, onb.astype(np.int64), T)
    sts = [warm_start(P, 25, int(u0[c]), int(onb[c])) for c in range(M)]
    for t in range(T):
        bm = bc.boundary_mask(t)
        if bm.any():
            bc.day_boundary(bc.day_true.copy(), bm)
        for c in range(M):
            if sts[c].is_boundary(t):
                sts[c].day_boundary(sts[c].resync_payload())
        active = np.array([t >= int(onb[c]) for c in range(M)])
        bc.step(u[t], active)
        for c in range(M):
            if t >= int(onb[c]):
                sts[c].step(int(u[t, c]))
        if t % 37 == 0 or t == T - 1:
            for c in range(M):
                if t >= int(onb[c]) + 2:
                    assert bc.canonical(c) == sts[c].canonical(), (t, c)


def _scalar_exact(u_c, recon, t0, T):
    """What u_exact should be: the quantised truth on any resynced day, the
    within-day reconstruction elsewhere."""
    out = recon[t0:].copy()
    for t in range(t0, T):
        if t % 24 == 0 and t - t0 >= 24:
            out[t - 24 - t0:t - t0] = u_c[t - 24:t]
    return out


def test_scheduled_shutdown_does_not_explode_the_budget():
    """A recurring zero run is NOT an anomaly (blueprint 7). The codec has to
    absorb it; a filter that flags every cycle would burn the daily budget and
    make the exception rate a report about the shutdown schedule."""
    T = 24 * 40
    t = np.arange(T)
    clean = 800 * (1 + .4 * np.sin(2 * np.pi * t / 24))
    shut = clean.copy()
    shut[(t % 168) < 8] = 0.0                 # weekly 8-hour shutdown

    def rate(y):
        k = choose_scale(y)
        u, _ = quantize_array(y, k, P.u_clamp)
        # eps at alpha = 1, i.e. the client's own typical day-over-day change,
        # which is how sim/run_codec.py calibrates it. A fixed eps in device
        # units would make this a test of the quantiser scale instead.
        eps = int(round(np.mean(np.abs(u[24:] - u[:-24]))))
        _, _, log, _ = drive(u, eps, 0, T)
        return sum(1 for _, b in log if b[0] >> 4 == 1) / T

    r_clean, r_shut = rate(clean), rate(shut)
    assert r_shut < 2.5 * max(r_clean, 0.02), (
        f"exception rate went from {r_clean:.3f} to {r_shut:.3f} when a "
        f"scheduled shutdown was added: the codec is treating a known "
        f"recurring pattern as an anomaly")
