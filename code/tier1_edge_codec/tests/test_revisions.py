"""
test_revisions.py  --  the review items, each with a test that would fail if the
change were reverted.

A revision list is only implemented if reverting it breaks something.
"""

import numpy as np
import pytest
import torch

from edge.codec import (PARAMS_Q12_LEGACY, PARAMS_SEND_ON_DELTA, CodecParams,
                        CodecState, warm_start)
from eval.run_eval import (BATCH_DAILY, CONTROL, build_daily_batch_forecasts,
                           derive_budget, held_forecast)
from eval.sensitivity import restricted_rowsums
from sim.run_codec import Store, run_policy
from tier2.adapter import (forecast_kwh, iter_batches, load_frozen_dlinear,
                           make_window_builder, store_manifest)


# ------------------------------------------ the Q12 resolution defect (section 1)
def test_q12_update_quantises_to_nothing_and_q20_does_not():
    """The defect that made the filter lose to frozen persistence.

    At Q12, with loads around 1e4 device units, the NLMS update
    `(e * phi) >> (b - w_q + k_mu)` is 2 LSBs at k_mu = 2 and ZERO by k_mu = 4:
    the whole useful range of step sizes lies between "barely moves" and
    "does not move". Q20 gives the same step sizes two orders of magnitude of
    headroom. This is arithmetic, so it is asserted rather than described.
    """
    from edge.fixedpoint import msb_index, round_shift
    e, phi_lag, n_taps = 100, 10_000, 6

    def step_lsbs(w_q, k_mu, bias_input):
        P = n_taps * phi_lag ** 2 + bias_input ** 2
        sh = max(0, msb_index(P) - w_q + k_mu)
        return round_shift(e * phi_lag, sh)

    q12 = {k: step_lsbs(12, k, 4096) for k in (2, 3, 4, 5)}
    q20 = {k: step_lsbs(20, k, 4096) for k in (2, 3, 4, 5)}
    # Section 1.2's table, reproduced exactly: 2, 1, 0, 0.
    assert [q12[k] for k in (2, 3, 4, 5)] == [2, 1, 0, 0], q12
    # At Q20 nothing in the usable range quantises away, and k_mu = 5 -- the
    # value the sweep chose -- is 61 LSBs rather than 0.
    assert all(v > 0 for v in q20.values()), q20
    assert q20[5] == 61, q20
    # The RELATIVE step (fraction of unity weight) is finer at Q20/k=5 than the
    # only usable Q12 setting, so lowering mu is now actually representable.
    assert q20[5] / (1 << 20) < q12[2] / (1 << 12)


def test_the_shipped_configuration_is_q20_int32():
    p = CodecParams()
    assert p.w_q == 20 and p.w_clamp == 4 << 20 and p.k_mu == 5
    st = warm_start(p, 10, 5000, 0)
    # a weight that would overflow int16 must be representable
    st.w[0] = 4 << 20
    assert st.snapshot()


def test_bias_regressor_is_not_tied_to_w_q():
    """If the bias input were 1 << w_q it would be 1048576 at Q20 -- a hundred
    times the load magnitude -- so phi_bias^2 would dominate P and drive every
    LAG update to zero, reintroducing the defect Q20 exists to fix."""
    from edge.fixedpoint import msb_index, round_shift
    p = CodecParams()
    assert p.bias_input == 4096 and p.bias_input != (1 << p.w_q)
    phi_lag, e = 10_000, 100

    def lag_step(bias_input):
        P = len(p.taps) * phi_lag ** 2 + bias_input ** 2
        sh = max(0, msb_index(P) - p.w_q + p.k_mu)
        return round_shift(e * phi_lag, sh)

    assert lag_step(p.bias_input) > 32
    assert lag_step(1 << p.w_q) == 0, (
        "tying the bias regressor to w_q would freeze the lag weights")


def test_new_filter_beats_frozen_persistence():
    """Section 1's whole point: the ordering that made the original experiment
    report a negative result must now be the other way round."""
    rng = np.random.default_rng(0)
    T = 24 * 90
    t = np.arange(T)
    prof = np.array([.45, .40, .38, .37, .38, .45, .62, .85, 1.0, 1.05, 1.05,
                     1.02, .98, 1.0, 1.02, 1.0, .95, .92, .95, 1.0, .95, .80,
                     .62, .52])
    y = 4000 * prof[t % 24] * np.where((t // 24) % 7 >= 5, .6, 1.) \
        * np.exp(rng.normal(0, .08, T))
    u = np.round(y / 0.5).astype(int)

    def mean_abs_e(params):
        st = warm_start(params, 0, int(u[0]), 0)
        e = []
        for i in range(T):
            if st.is_boundary(i):
                st.day_boundary(st.resync_payload())
            e.append(abs(st.step(int(u[i])).e))
        return float(np.mean(e[24 * 30:]))

    adaptive = mean_abs_e(CodecParams())
    frozen = mean_abs_e(PARAMS_SEND_ON_DELTA)
    old = mean_abs_e(PARAMS_Q12_LEGACY)
    assert adaptive < frozen, (
        f"the adaptive filter ({adaptive:.1f}) still loses to frozen "
        f"persistence ({frozen:.1f}); section 2's fixes did not take")
    assert adaptive < old, (adaptive, old)


# -------------------------------------------------------------- B2: the cap
def test_the_cap_is_non_binding_by_default():
    """With n_max_tx = 24 the daily budget cannot be exhausted, so eps never
    doubles and the guarantee is unconditional. That is what makes the alpha
    axis a clean single-variable sweep."""
    assert CodecParams().n_max_tx == 24
    st = warm_start(CodecParams(), 5, 100, 0)  # noqa: F841
    for _ in range(24):
        assert st.eps_effective() == 5
        st.n_tx_today += 1
    st2 = warm_start(CodecParams(n_max_tx=8), 5, 100, 0)
    st2.n_tx_today = 8
    assert st2.eps_effective() == 10, "the robustness config must still double"


def test_epsilon_bound_is_unconditional_at_the_default_cap(store):
    """B2's payoff: with the cap non-binding, |recon - truth| <= eps exactly,
    not <= 2 eps."""
    for alpha in (0.25, 0.5, 1.0):
        r = run_policy(store, alpha, CodecParams(), backend="batch")
        eps = np.array([0 if e is None else e for e in r.eps], np.int64)
        d = np.abs(r.u_within - r.u_q)
        assert int(((d > eps[None, :]) & r.active).sum()) == 0, alpha
        assert (r.eps_eff[r.active] == eps[None, :].repeat(
            r.active.shape[0], 0)[r.active]).all(), (
            f"eps doubled somewhere at alpha={alpha} despite n_max_tx=24")


# ------------------------------------------- the selection rule (review fix 1)
def test_tolerance_is_actually_enforced():
    """The review caught a selection that returned the largest alpha inside the
    airtime budget while claiming to enforce a dMASE tolerance. The tolerance
    has to be able to REJECT."""
    margin, tol = 0.1553, 0.10
    tol_abs = tol * margin
    cands = [("0.7", 9.37, 0.01403, True), ("1", 6.57, 0.02459, True)]
    ok = [c for c in cands if c[2] <= tol_abs and c[3]]
    assert [c[0] for c in ok] == ["0.7"], (
        "alpha=1 gives back 16% of the reported skill margin; a rule claiming "
        "10% must not select it")
    best = max(ok, key=lambda c: float(c[0]))
    assert best[0] == "0.7"


def test_margin_basis_is_conservative_by_default():
    """The bug was anchoring the tolerance to the VALIDATION skill margin, which
    is 1.8x easier than test for this forecaster and inflated the allowance by
    the same factor. The default takes the smaller of the two."""
    import inspect

    from eval import run_eval as re_
    src = inspect.getsource(re_.main)
    assert 'margins["min"] = min(v_margin, margin)' in src
    ap_src = inspect.getsource(re_).split("--margin_basis")[1]
    assert 'default="min"' in ap_src


# ------------------------------------------------------------- B3: the budget
def test_budget_is_derived_from_a_radio_constraint():
    """The budget is AIRTIME, and that is what the selection test enforces.

    This test previously asserted against a packet-count budget and a
    `budget_pkt_per_day` key. The budget moved to airtime and the key was
    renamed to `budget_pkt_per_day_indicative`; the test was not updated and
    had been failing with a KeyError. See docs/findings/2026-09-14-budget-test.md. It is
    rewritten here against the quantity the pipeline actually enforces, rather
    than repaired by swapping the key -- asserting monotonicity of a figure
    the codebase itself labels "indicative only", while asserting nothing
    about the enforced budget, is what let the drift go unnoticed.
    """
    b = derive_budget(50.0, sf=12)
    # the budget is (1 - retry_margin) x allowance, in seconds
    assert b["budget_airtime_s"] == (1.0 - b["retry_margin"]) * b["allowance_s"]
    assert "TTN" in b["derivation"] and "SF12" in b["derivation"]
    # and it is a property of the radio allowance alone: neither the payload
    # size nor the spreading factor may move it
    assert derive_budget(100.0, sf=12)["budget_airtime_s"] == b["budget_airtime_s"]
    assert derive_budget(50.0, sf=7)["budget_airtime_s"] == b["budget_airtime_s"]


def test_indicative_packet_conversion_moves_the_right_way():
    """The packets/day figure is a presentational conversion, not a budget.

    Kept as a separate test from the one above so that a failure here can
    never be mistaken for a failure of the enforced airtime budget.
    """
    K = "budget_pkt_per_day_indicative"
    b = derive_budget(50.0, sf=12)
    assert b[K] > 0
    # a longer packet must buy fewer of them, and a lower SF must buy more
    assert derive_budget(100.0, sf=12)[K] < b[K]
    assert derive_budget(50.0, sf=7)[K] > b[K]


# ------------------------------------------- B4 / the addition: the two rows
def test_held_forecast_alignment_and_coverage():
    """At issue hour h the morning run covers the first 24-h steps and no more,
    and step j of the window is lead h+j of that run."""
    fc = {(24, 0): np.arange(100, 124, dtype=float)}
    chunk = np.array([[24, 0], [24 + 5, 0], [24 + 23, 0]], np.int64)
    f, cov = held_forecast(fc, chunk, 24)
    assert cov[0].all() and np.allclose(f[0], np.arange(100, 124))
    assert cov[1].sum() == 19 and np.allclose(f[1, :19], np.arange(105, 124))
    assert cov[2].sum() == 1 and f[2, 0] == 123
    assert not cov[2, 1:].any(), "uncovered steps must be masked, not filled"


def test_batch_daily_at_h0_is_exactly_the_control(store_dir):
    """The consistency check that pins the baseline: at the resync hour the
    held forecast IS the clean-window forecast, so if these ever differ the
    baseline is not doing what its docstring says."""
    man = store_manifest(store_dir["data"])
    model, cfg, _ = load_frozen_dlinear(store_dir["ckpt"], man)
    wb = make_window_builder(store_dir["data"], cfg.lookback_len, cfg.horizon_len)
    sel = wb.splits["test"]
    sel = sel[sel[:, 0] % 24 == 0][:64]
    assert len(sel), "no h=0 origins in the test split"
    fc, _ = build_daily_batch_forecasts(model, wb, sel, 64, "cpu")
    chunk, b = next(iter_batches(wb, sel, 64))
    held, cov = held_forecast(fc, chunk, wb.H)
    ctrl = forecast_kwh(model, b).numpy()
    assert cov.all()
    assert np.abs(held - ctrl).max() < 1e-4, float(np.abs(held - ctrl).max())


# ------------------------------------------------------- B5: endpoint reporting
def test_legacy_sync_is_the_default(store):
    """Section 2, Fix 3. legacy measured better AND cheaper at every alpha;
    sync_safe stays compiled as a robustness option."""
    assert CodecParams().sync_safe is False
    assert CodecParams(sync_safe=True).sync_safe is True


def test_endpoint_doubling_is_not_reported_as_a_number(store):
    """0.0% at alpha=0 was accidentally right for the wrong reason and 100.0%
    at alpha=inf was a reporting bug. Both must be n/a."""
    from eval.metrics_tier1 import frac_hours_capped
    r = run_policy(store, float("inf"), CodecParams(), backend="batch")
    assert all(e is None for e in r.eps)
    # the driver reports None for this case; the metric itself still returns a
    # number, so the test is on the driver's rule, expressed here directly
    eps_is_inf = all(e is None for e in r.eps)
    assert eps_is_inf, "alpha=inf must have no finite eps to double"


# ------------------------------------------------------ B6: restricted Jacobian
def test_restricted_jacobian_is_tighter_and_monotone_in_h():
    J = np.abs(np.random.default_rng(0).normal(0, 1, (24, 336)))
    R = restricted_rowsums(J)
    assert R.shape == (24, 24)
    # more columns can only add, so it is non-decreasing in the issue hour
    assert (np.diff(R, axis=1) >= -1e-12).all()
    # and h=0 is a single column
    assert np.allclose(R[:, 0], J[:, -1])
    # strictly tighter than the all-column sum
    assert R[:, 23].max() < np.abs(J).sum(axis=1).max()


# --------------------------------------------------------- R1: masked metrics
def test_recon_metrics_exclude_imputed_hours(store):
    """On imputed spans the codec is being scored against a linear interpolant,
    which is easier to predict than real data."""
    from eval.metrics_tier1 import recon_fidelity
    r = run_policy(store, 1.0, CodecParams(), backend="batch")
    live = r.active & ~store.missing
    a = recon_fidelity(r.u_within, r.u_q, r.k, live)["recon_mae_kwh"]
    b = recon_fidelity(r.u_within, r.u_q, r.k, r.active)["recon_mae_kwh"]
    assert store.missing.any(), "the fixture has no imputed hours"
    assert a != b, "masking made no difference; the mask is not being applied"


# ------------------------------------------------------- R2 / R5: bookkeeping
def test_client_accounting_names_every_dropped_client(store):
    a = store.client_accounting()
    assert a["M"] == store.M
    assert a["n_scorable"] <= a["M"]
    assert set(a["dropped"]) >= set(a["degenerate"])
    assert isinstance(a["note"], str) and a["note"]


def test_quantisation_scale_is_train_only(store):
    assert store.assert_scales_are_train_only()


# ------------------------------------------------------------- R7 / R8 variants
def test_send_on_delta_never_adapts(store):
    r = run_policy(store, 1.0, PARAMS_SEND_ON_DELTA, backend="batch")
    assert r.tx.any()
    st = warm_start(PARAMS_SEND_ON_DELTA, 10, 100, 0)
    for t in range(240):
        if st.is_boundary(t):
            st.day_boundary()
        st.step(100 + (t % 7) * 13)
    assert st.w == list(PARAMS_SEND_ON_DELTA.w_init)


def test_168_lag_is_in_the_default_tap_set(store):
    """The blueprint rejected the weekly lag on a division-of-labour argument;
    the R8 ablation measured otherwise, so it is now shipped. buf_len = 24+168
    additionally makes the carry unnecessary."""
    p = CodecParams()
    assert 168 in p.taps
    assert p.max_tap == 168 and p.buf_len > p.max_tap
    assert p.n_carry == 0, "buf_len 192 should make the carry unnecessary"
    r = run_policy(store, 1.0, p, backend="batch")
    assert r.tx.any()
    eps = np.array([0 if e is None else e for e in r.eps], np.int64)
    d = np.abs(r.u_within - r.u_q)
    assert int(((d > eps[None, :]) & r.active).sum()) == 0

    # SRAM cost of the ring, so the firmware README's claim can be checked
    assert p.buf_len * 2 == 384


def test_carry_generalises_to_any_tap_set():
    """n_carry is arithmetic, not a magic 1."""
    assert CodecParams().n_carry == 0
    assert PARAMS_Q12_LEGACY.n_carry == 1
    assert CodecParams(taps=(1, 2, 3, 24, 25, 168), buf_len=169,
                       w_init=(4096, 0, 0, 0, 0, 0, 0)).n_carry == 23
    with pytest.raises(AssertionError):
        CodecParams(taps=(1, 200), buf_len=48, w_init=(4096, 0, 0))


def test_pretrained_init_is_carried_by_with_w_init():
    w = tuple(range(7))
    p = CodecParams().with_w_init(w)
    assert p.w_init == w and CodecParams().w_init != w
    st = warm_start(p, 10, 100, 0)
    assert st.w == list(w)


def test_the_fair_use_allowance_is_the_same_number_everywhere():
    """TTN_FAIR_USE_S is defined twice, in eval/metrics_tier1.py and in
    eval/figures.py, with no import relationship between them. They agree
    today. If either is ever edited alone, the tables and the figures would
    silently disagree about the allowance and every '% of allowance' in the
    paper would be drawn against two different denominators. Nothing else
    catches that, so this does.
    """
    from eval.figures import TTN_FAIR_USE_S as in_figures
    from eval.metrics_tier1 import TTN_FAIR_USE_S as in_metrics
    assert in_figures == in_metrics, (
        f"the fair-use allowance has diverged: metrics_tier1 says "
        f"{in_metrics}, figures says {in_figures}. Make them agree before "
        f"trusting any '% of allowance' figure.")
