"""
adapter.py  --  the ONLY file that touches the Tier-2 API. Blueprint 6.6.

Everything of ours that needs `WindowBuilder`, `Accum`, `TiDEConfig` or the
checkpoint format goes through here. If the owner's signatures change, this
file changes and nothing else does.

WHY patch_y_past IS THE WHOLE HARNESS
====================================
DLinear reads ONLY y_past. Verify it in the owner's models_tier2.py: the
signature is forward(y_past, x_past, x_fut, s, w=None) and the body touches
nothing but y_past. And because RevIN computes mu and sigma from that same
window internally, replacing y_past is not just sufficient -- it is complete.
The reconstruction error then propagates through the normalisation statistics
AND through the inversion that uses them, which is blueprint 3(b) and is the
difference between a real measurement and a flattering one.

Everything else in the batch comes from the untouched store: targets, the
per-step validity mask, the baselines, the MASE scale. So they are byte
identical across policies and the ONLY thing that varies between rows of the
results table is DLinear's input. That is the experiment, stated as code.

NO PER-POLICY DATA DIRECTORIES. The store is read once. There is no symlink
farm, no regenerated manifest, and no copy of any *_tier2.py file.
"""

import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(REPO, "vendor")
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

from evalaccum_tier2 import Accum                      # noqa: E402,F401
from metrics_tier2 import (distribution, frac_no_skill,  # noqa: E402,F401
                           mase_scale, per_client, per_horizon, pooled_metrics)
from models_tier2 import TiDEConfig, build             # noqa: E402
from windows_tier2 import WindowBuilder                # noqa: E402


class IndexConventionError(AssertionError):
    pass


def make_window_builder(data_dir, lookback=336, horizon=24):
    return WindowBuilder(data_dir, lookback=lookback, horizon_len=horizon)


def load_frozen_dlinear(ckpt_path, manifest, lookback=336, horizon=24,
                        device="cpu"):
    """Load exactly as 30_evaluate_tier2.py does, then assert the contract.

    `device` places the model. It must agree with the device the batches are
    built on (eval/run_eval.py threads the same --device to both).

    Blueprint 2.2: raise on mismatch, do not warn. A checkpoint trained under a
    different feature spec, look-back or horizon is not a model that can be
    evaluated on this store -- it is a different experiment wearing the same
    filename.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ck["model"] == "dlinear", (
        f"Tier 1 shadows a frozen DLinear; this checkpoint is '{ck['model']}'. "
        f"Blueprint 2.3 puts every other model row out of scope.")
    cfg = TiDEConfig(**ck["cfg"])
    assert cfg.lookback_len == lookback, (
        f"checkpoint L={cfg.lookback_len}, expected {lookback}")
    assert cfg.horizon_len == horizon, (
        f"checkpoint H={cfg.horizon_len}, expected {horizon}")
    assert ck["feature_spec_hash"] == manifest["feature_spec_hash"], (
        f"checkpoint feature spec {ck['feature_spec_hash'][:16]} != store "
        f"{manifest['feature_spec_hash'][:16]}. Retrain or repreprocess.")
    model = build("dlinear", cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    # The checkpoint is deliberately loaded map_location="cpu" above -- a
    # checkpoint must not carry its training device into an evaluation. The
    # model is placed here instead, matching vendor/30_evaluate_tier2.py:48
    # (`m.eval().to(device)`).
    #
    # Until this line existed, --device only reached the INPUT batches
    # (iter_batches -> wb.build(chunk, device=...) and patch_y_past), so any
    # device other than cpu put the tensors on the GPU and left the weights on
    # the CPU, and the first matmul raised a device mismatch. `--device cuda`
    # had never run.
    #
    # `.to("cpu")` on a CPU model is a no-op, so the default path is
    # bit-identical -- PLAYBOOK rule 4. tests/test_device.py asserts it.
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, ck


def probe_index_convention(wb, split="test", n=64):
    """Blueprint 6.6. `sel` rows are (index, client); the document does not say
    whether `index` is the window start or the forecast origin. Determine it
    empirically and FAIL LOUDLY, at startup, rather than documenting a guess.

    The assumed convention is ORIGIN: y_true_kwh[b, h] == series_true[t + 1 + h, c]
    and y_past is series_log1p[t - L + 1 .. t, c]. Both are checked.
    """
    sel = wb.splits[split][:n]
    b = wb.build(sel)
    t, c = sel[:, 0], sel[:, 1]

    fut = t[:, None] + wb.lag + np.arange(wb.H)[None, :]
    ref = np.nan_to_num(wb.true[fut, c[:, None]])
    got = b["y_true_kwh"].numpy()
    # RELATIVE tolerance. The store is float64 and the batch arrives as
    # float32, so a CORRECT match still differs by ~1e-4 on a 1000 kWh meter.
    # An absolute 1e-6 test would reject the right convention for a rounding
    # reason, which is a worse failure than not testing at all. A one-hour
    # shift moves these by a whole hourly load, orders of magnitude above this
    # floor, so the test still separates the two conventions cleanly --
    # tests/test_contract.py asserts it fires on a deliberately shifted index.
    d_target = float(np.abs(ref - got).max()) / max(float(np.abs(ref).max()), 1.0)

    past = t[:, None] + np.arange(-wb.L + 1, 1)[None, :]
    ref_past = wb.series[past, c[:, None]]
    d_past = (float(np.abs(ref_past - b["y_past"].numpy()).max())
              / max(float(np.abs(ref_past).max()), 1.0))

    if not (d_target < 1e-5 and d_past < 1e-5):
        raise IndexConventionError(
            f"sel[:, 0] is NOT the forecast origin under the assumed "
            f"convention (target dev {d_target:.3e}, look-back dev "
            f"{d_past:.3e}). Every window this harness patches would be "
            f"shifted, so nothing downstream is trustworthy. Fix the "
            f"convention here, in tier2/adapter.py, and nowhere else.")
    return {"convention": "sel[:,0] is the forecast origin t; targets are "
                          f"y[t+{wb.lag} .. t+{wb.lag + wb.H - 1}]",
            "target_max_dev": d_target, "lookback_max_dev": d_past,
            "n_checked": int(len(sel))}


def iter_batches(wb, sel, batch_size=512, device="cpu"):
    """Stream (chunk, batch) pairs. `chunk` is the (B, 2) index array, which
    the caller needs in order to know which windows to patch."""
    for i in range(0, len(sel), batch_size):
        chunk = sel[i:i + batch_size]
        yield chunk, wb.build(chunk, device=device)


def patch_y_past(batch, chunk, y_exact_log1p, y_within_log1p, L, device="cpu"):
    """Replace y_past with the policy's reconstructed look-back. Returns a
    tensor; the caller substitutes it into the batch dict.

    THIS IS THE CRUX (blueprint 6.6). Nothing else in the batch is touched.
    """
    from sim.reconstruct import assemble_window
    y = assemble_window(chunk[:, 0], chunk[:, 1], y_exact_log1p, y_within_log1p, L)
    return torch.as_tensor(np.ascontiguousarray(y), dtype=torch.float32,
                           device=device)


@torch.no_grad()
def forecast_kwh(model, batch, y_past=None):
    """Run the frozen DLinear and invert to kWh through its own to_kwh, so the
    RevIN statistics used on the way out are the ones computed from the
    (possibly reconstructed) window on the way in. Blueprint 3(b)."""
    yp = batch["y_past"] if y_past is None else y_past
    yh, stats = model(yp, batch["x_past"], batch["x_fut"], batch["s"], batch["w"])
    return model.to_kwh(yh, stats)


def store_manifest(data_dir):
    with open(os.path.join(data_dir, "manifest.json")) as f:
        return json.load(f)
