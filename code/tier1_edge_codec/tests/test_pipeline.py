"""
test_pipeline.py  --  blueprint 9.4, plus the metric-integration checks.

One end-to-end run: generate -> run the codec at three alphas -> reconstruct ->
patch -> score -> write JSON. The central assertion is that alpha = 0 and
alpha = inf produce DIFFERENT forecasts. With a randomly-initialised checkpoint
the numbers are meaningless, but the difference proves patch_y_past reaches the
model -- which is the one thing a synthetic run can actually establish, and the
one thing that silently fails if the harness is wired wrong.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from edge.codec import CodecParams
from eval.metrics_tier1 import (communication_cost, frac_hours_capped,
                                linf_violation, proxy_events, recon_fidelity,
                                staleness_hours)
from sim.reconstruct import build_log1p_arrays
from sim.run_codec import run_policy
from tier2.adapter import (forecast_kwh, iter_batches, load_frozen_dlinear,
                           make_window_builder, patch_y_past,
                           probe_index_convention, store_manifest)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = CodecParams()


@pytest.fixture(scope="module")
def policies(store):
    out = {}
    for a in (0.0, 1.0, float("inf")):
        r = run_policy(store, a, P, backend="batch")
        out[a] = (r, *build_log1p_arrays(r, store))
    return out


def test_linf_bound_is_zero_at_every_alpha(policies):
    """An assertion, not a metric (blueprint 8.3)."""
    for a, (r, ex, wi) in policies.items():
        assert linf_violation(r.u_within, r.u_q, r.eps_eff, r.active) == 0, a


def test_endpoints_behave_as_the_two_baselines(policies, store):
    r0 = policies[0.0][0]
    rinf = policies[float("inf")][0]
    # alpha = 0: exact reconstruction everywhere the meter was live
    assert np.array_equal(r0.u_within[r0.active], r0.u_q[r0.active])
    # alpha = inf: not one exception, so exactly one packet per active day
    assert not rinf.tx.any()
    cc0 = communication_cost(r0.tx, r0.resync_bytes, r0.active)
    ccinf = communication_cost(rinf.tx, rinf.resync_bytes, rinf.active)
    assert ccinf["packets_per_day_mean"] < 1.05
    assert cc0["packets_per_day_mean"] > 10 * ccinf["packets_per_day_mean"]


def test_cost_and_error_move_in_opposite_directions(policies, store):
    """The rate-distortion curve must actually be a curve: more packets, less
    reconstruction error, monotonically."""
    rows = []
    for a in (0.0, 1.0, float("inf")):
        r = policies[a][0]
        cc = communication_cost(r.tx, r.resync_bytes, r.active)
        fid = recon_fidelity(r.u_within, r.u_q, r.k, r.active, store.mase_scale)
        rows.append((cc["packets_per_day_mean"], fid["recon_mae_kwh"]))
    pkts = [p for p, _ in rows]
    errs = [e for _, e in rows]
    assert pkts[0] > pkts[1] > pkts[2], pkts
    assert errs[0] < errs[1] < errs[2], errs


def test_staleness_ordering(policies):
    st0 = staleness_hours(policies[0.0][0].tx, policies[0.0][0].active)
    stinf = staleness_hours(policies[float("inf")][0].tx,
                            policies[float("inf")][0].active)
    a0 = policies[0.0][0].active
    ainf = policies[float("inf")][0].active
    assert st0[a0].mean() < stinf[ainf].mean()
    # daily batching is exactly (t % 24) + 1 for every live hour
    T = stinf.shape[0]
    c = 0
    live = np.nonzero(ainf[:, c])[0]
    live = live[live >= (live.min() // 24 + 2) * 24]
    assert np.array_equal(stinf[live, c], live % 24 + 1)


def test_patch_reaches_the_model_and_alpha_matters(store_dir, store, policies):
    """THE pipeline assertion (blueprint 9.4)."""
    man = store_manifest(store_dir["data"])
    model, cfg, ck = load_frozen_dlinear(store_dir["ckpt"], man)
    wb = make_window_builder(store_dir["data"], cfg.lookback_len, cfg.horizon_len)
    probe_index_convention(wb)
    sel = wb.splits["test"][:256]
    chunk, b = next(iter_batches(wb, sel, 256))

    fc = {}
    for a, (r, ex, wi) in policies.items():
        yp = patch_y_past(b, chunk, ex, wi, wb.L)
        fc[a] = forecast_kwh(model, b, yp).numpy()

    d = np.abs(fc[0.0] - fc[float("inf")])
    assert d.max() > 0, (
        "alpha=0 and alpha=inf produce identical forecasts: patch_y_past is "
        "not reaching the model and every row of the results table is the "
        "same experiment")
    # the oracle's window must be the store's own, to within quantisation
    yp0 = patch_y_past(b, chunk, policies[0.0][1], policies[0.0][2], wb.L)
    assert float((yp0 - b["y_past"]).abs().max()) < 5e-3
    # and the ordering must be sensible: alpha=1 sits between the endpoints
    d1 = np.abs(fc[1.0] - fc[0.0]).mean()
    dinf = np.abs(fc[float("inf")] - fc[0.0]).mean()
    assert d1 <= dinf, (d1, dinf)


def test_only_y_past_is_touched(store_dir, store, policies):
    """Targets, masks, baselines and the MASE scale must be identical across
    policies. If they are not, the comparison is not controlled."""
    man = store_manifest(store_dir["data"])
    _, cfg, _ = load_frozen_dlinear(store_dir["ckpt"], man)
    wb = make_window_builder(store_dir["data"], cfg.lookback_len, cfg.horizon_len)
    chunk, b = next(iter_batches(wb, wb.splits["test"][:128], 128))
    ref = {k: v.clone() for k, v in b.items()}
    for a, (r, ex, wi) in policies.items():
        patch_y_past(b, chunk, ex, wi, wb.L)
    for k, v in b.items():
        assert torch.equal(v, ref[k]), f"patching mutated the batch key '{k}'"


def test_proxy_events_are_labelled_and_plausible(store):
    pe, thr = proxy_events(store.series_true, store.missing, store.train_end)
    assert pe.dtype == bool
    frac = pe.sum() / max(np.isfinite(store.series_true).sum(), 1)
    assert frac < 0.1, f"proxy fires on {frac:.1%} of hours; it is not an event"


def test_end_to_end_cli(store_dir, tmp_path):
    """Runs the real entry points as a user would, and checks the JSON they
    write. A pipeline that only works when driven from inside pytest is not a
    pipeline."""
    env = dict(os.environ, PYTHONPATH=REPO)
    out = tmp_path / "tier1_results.json"
    pre = tmp_path / "pre.json"
    pc = tmp_path / "pre_pc.npy"
    rp = subprocess.run(
        [sys.executable, "-m", "sim.pretrain", "--data_dir", store_dir["data"],
         "--out", str(pre), "--per_client_out", str(pc)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    assert rp.returncode == 0, rp.stdout[-2000:] + rp.stderr[-2000:]
    r = subprocess.run(
        [sys.executable, "-m", "eval.run_eval",
         "--pretrained", str(pre), "--per_client", str(pc),
         "--data_dir", store_dir["data"], "--checkpoint", store_dir["ckpt"],
         "--alphas", "0,1", "--ablation_alphas", "1",
         "--max_samples", "400", "--select_samples", "400",
         # a private spill directory: the default, _recon_cache/, is the one
         # a real run memory-maps, and overwriting its files under a live run
         # kills it with SIGBUS (found 2026-09-29).
         "--recon_dir", str(tmp_path / "recon"),
         "--out", str(out)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "NOT reportable" in r.stdout
    assert "SYNTHETIC" in r.stdout

    res = json.loads(out.read_text())
    assert res["SYNTHETIC_store"] and res["SYNTHETIC_checkpoint"]
    for lab in ("C@0", "C@1"):
        assert res["cost"][lab]["linf_violation"] == 0
        assert "delta_MASE" in res["metrics"][lab]
    assert res["cost"]["C@0"]["packets_per_day_mean"] > \
        res["cost"]["C@1"]["packets_per_day_mean"]
    # the two reference rows must be present and labelled
    for lab in ("control", "batch-daily"):
        assert lab in res["metrics"], lab
    assert res["use_case"].startswith("The scheme buys")
    assert 0.0 < res["batch_daily_coverage"] <= 1.0
    # the four-row ablation must be there, named
    for fam in "ABCD":
        assert fam in res["ablation"] and res["ablation"][fam]["name"]
    # alpha = 0 has no meaningful doubling rate to report
    assert res["cost"]["C@0"]["frac_hours_eps_doubled"] is None
    # the deleted material must NOT be in the main results
    assert "event_proxy" not in res["cost"]["C@1"]
    assert not any(l.endswith("inf") for l in res["metrics"])
    assert res["limitations"] and len(res["limitations"]) >= 4

    # the checkpoint must be labelled as the fallback it is
    assert res["checkpoint_is_fallback"] is True
    assert "NOT the frozen model" in res["checkpoint_provenance"]
    assert "FALLBACK CHECKPOINT" in r.stdout

    # RESULTS.md must be generated from the JSON, never typed
    rt = tmp_path / "RESULTS.md"
    r3 = subprocess.run(
        [sys.executable, "-m", "eval.report_tables", "--results", str(out),
         "--sensitivity", str(tmp_path / "nope.json"), "--out", str(rt)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert r3.returncode == 0, r3.stdout + r3.stderr
    md = rt.read_text()
    assert "Do not edit by hand" in md
    assert "CHECKPOINT IS A FALLBACK" in md
    for lab in ("control", "batch-daily"):
        assert lab in md
    # the headline must quote the DELIVERABLE, not the no-codec bound, whenever
    # an operating point exists
    if res.get("selection"):
        assert "no-codec upper bound" in md

    # and the figures script must consume that JSON without hand-editing
    figs = tmp_path / "figs"
    r2 = subprocess.run(
        [sys.executable, "-m", "eval.figures", "--results", str(out),
         "--sensitivity", str(tmp_path / "nope.json"), "--outdir", str(figs)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    for name in ("fig1_lead1_by_issue_hour.png", "fig2_rate_utility.png",
                 "fig3_model_ablation.png", "fig4_rate_distortion.png",
                 "fig5_airtime_sf12.png", "table_system.csv",
                 "table_ablation.csv", "table_freshness.csv"):
        assert (figs / name).exists(), name


def test_metrics_tier1_self_test_passes():
    """The module's own __main__ suite, run as a subprocess so a failure shows
    up here rather than only when someone remembers to run it by hand."""
    env = dict(os.environ, PYTHONPATH=REPO)
    r = subprocess.run([sys.executable, "-m", "eval.metrics_tier1"],
                       cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "checks passed" in r.stdout
    assert "FAIL" not in r.stdout


def test_vendored_tier2_files_are_untouched():
    """Blueprint 6.6: no edits to any *_tier2.py the owner supplied. The
    checksums were taken when they were vendored."""
    import hashlib
    ref = {}
    with open(os.path.join(REPO, "vendor", "CHECKSUMS.md5")) as f:
        for line in f:
            h, name = line.split()
            ref[os.path.basename(name)] = h
    for name, h in ref.items():
        p = os.path.join(REPO, "vendor", name)
        got = hashlib.md5(open(p, "rb").read()).hexdigest()
        assert got == h, f"{name} has been modified -- it must not be"
