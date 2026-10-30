"""
test_revision.py  --  the machinery added for the conference revision
(2026-09-29): the horizon assembly and the periodic-predictor cell of the
factorial (request F), the per-client outputs of run_eval (request B), and the
population options of eval/protocol_matrix.py (request A).
"""
import json
import os
import subprocess
import sys

import numpy as np

from sim.reconstruct import (assemble_window, assemble_window_horizon,
                             day_start_horizon, refresh_horizon)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _arrays(T=2000, M=3, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((T, M)).astype(np.float32),
            rng.random((T, M)).astype(np.float32))


def test_horizon_assembly_with_the_day_rule_is_the_old_assembly():
    ex, wi = _arrays()
    L = 336
    t = np.arange(L, 1990, 7)
    c = t % 3
    old = assemble_window(t, c, ex, wi, L)
    new = assemble_window_horizon(t, c, ex, wi, L, day_start_horizon)
    assert np.array_equal(old, new)


def test_refresh_horizon_reads_every_delivered_hour_as_exact():
    ex, wi = _arrays()
    L, period = 48, 3
    t = np.array([1000, 1001, 1002, 1003])
    c = np.zeros_like(t)
    y = assemble_window_horizon(t, c, ex, wi, L, refresh_horizon(period))
    for row, tt in enumerate(t):
        last = tt - tt % period
        past = tt + np.arange(-L + 1, 1)
        want = np.where(past > last, wi[past, 0], ex[past, 0])
        assert np.array_equal(y[row], want)
    # and the day rule would have read the earlier refreshed hours from within
    y_day = assemble_window(t, c, ex, wi, L)
    assert not np.array_equal(y, y_day)


def test_periodic_predictor_is_exact_on_refresh_hours_and_keeps_the_schedule(store):
    from sim.periodic import run_periodic
    from sim.periodic_predict import run_periodic_predict
    w = (1 << 20, 0, 0, 0, 0, 0, 0)
    for period in (2, 3):
        base = run_periodic(store, period)
        pp = run_periodic_predict(store, period, w, workers=1)
        act = pp.active
        refresh = (np.arange(store.T) % period == 0)[:, None] & act
        assert np.array_equal(pp.u_within[refresh], pp.u_q[refresh])
        assert np.array_equal(pp.tx, base.tx)
        assert np.array_equal(pp.resync_bytes, base.resync_bytes)
        assert np.array_equal(pp.u_exact, base.u_exact)
        # between refreshes it imputes, and differs from a zero-order hold
        between = act & ~refresh
        assert not np.array_equal(pp.u_within[between], base.u_within[between])


def test_run_eval_per_client_outputs(store_dir, tmp_path):
    env = dict(os.environ, PYTHONPATH=REPO)
    pre, pc = tmp_path / "pre.json", tmp_path / "pre_pc.npy"
    r = subprocess.run([sys.executable, "-m", "sim.pretrain", "--data_dir",
                        store_dir["data"], "--out", str(pre),
                        "--per_client_out", str(pc)],
                       cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    out, pco = tmp_path / "res.json", tmp_path / "pc.json"
    r = subprocess.run([sys.executable, "-m", "eval.run_eval",
                        "--pretrained", str(pre), "--per_client", str(pc),
                        "--data_dir", store_dir["data"],
                        "--checkpoint", store_dir["ckpt"],
                        "--alphas", "0,1", "--ablation_alphas", "1",
                        "--max_samples", "400", "--select_samples", "400",
                        "--per_client_out", str(pco), "--out", str(out),
                        # never the shared _recon_cache/: a live real run
                        # memory-maps it (SIGBUS, 2026-09-29)
                        "--recon_dir", str(tmp_path / "recon")],
                       cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    res, d = json.loads(out.read_text()), json.loads(pco.read_text())
    for key in ("lead1_abs_err_sum_kwh_per_client", "lead1_n_per_client",
                "packets_per_day_per_client", "airtime_per_day_s_per_client",
                "mase_scale_per_client"):
        assert key in d, key
    for lab in ("C@0", "C@1"):
        v = np.array(list(d["packets_per_day_per_client"][lab].values()))
        assert np.isclose(v.mean(), res["cost"][lab]["packets_per_day_mean"],
                          rtol=0, atol=1e-12)
        n = np.array(list(d["lead1_n_per_client"][lab].values()))
        s = np.array(list(d["lead1_abs_err_sum_kwh_per_client"][lab].values()))
        assert n.sum() > 0 and (s >= 0).all()
    # the new outputs never reach the reported artifact
    assert "lead1_abs_err_sum_kwh_per_client" not in json.dumps(res)


def test_protocol_matrix_whole_life_and_strides(store):
    import json as _j
    from eval.protocol_matrix import probe_client, summarise
    from sim.run_codec import choose_scales, eps_for_alpha
    w = (1 << 20, 0, 0, 0, 0, 0, 0)
    k, _ = choose_scales(store)
    eps, _ = eps_for_alpha(store, k, 1.0)
    c = next(i for i in range(store.M) if eps[i] is not None and eps[i] > 0)
    full = probe_client(store, c, w, 5, 1.0, ("gated", "legacy"), 0, 48,
                        scales=(k, eps))
    thin = probe_client(store, c, w, 5, 1.0, ("gated", "legacy"), 0, 48,
                        stride_exc=5, boundary_stride=4, scales=(k, eps))
    kinds = lambda rows, kind: [r for r in rows if r["event"] == kind]
    assert len(kinds(thin, "resync")) == -(-len(kinds(full, "resync")) // 4)
    assert len(kinds(thin, "exception")) == -(-len(kinds(full, "exception")) // 5)
    # whole life is right-censored: every probe's follow-up fits in the series
    assert all(r["t0"] + r["age_h"] + 48 <= store.T for r in full
               if r["event"] != "none")
    # the paper's node: a lost RESYNC is harmless over the whole life too
    assert not any(r["any_violation"] for r in kinds(full, "resync"))
    s = summarise(full, n_boot=50)
    assert s["ack"]["shares"]["never_back_in_sync"]["n"] == len(kinds(full, "ack"))
    _j.dumps(s)
