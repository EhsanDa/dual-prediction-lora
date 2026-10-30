"""
test_device.py  --  the loader must place the model, and placing it on the CPU
must change nothing.

Background. `eval/run_eval.py` has parsed a `--device` flag since it was
written, but the flag only ever reached the INPUT batches: `iter_batches` ->
`wb.build(chunk, device=...)` and `patch_y_past`. `load_frozen_dlinear` loaded
the checkpoint `map_location="cpu"` and never moved the module, so any device
other than cpu put the tensors on the GPU and left the weights on the CPU, and
the first matmul raised a device mismatch. `--device cuda` had never run.

Two things are asserted here, and the first matters more than the second:

  1. The default is still cpu and a cpu forward pass is BIT-IDENTICAL, byte for
     byte, to one taken through a loader that does no placement at all. This is
     PLAYBOOK rule 4 -- the fix must not move a single float in the reported
     pipeline -- and it is asserted against a reference computed in-process
     rather than a hardcoded hash, so it stays true if the fixture changes.

  2. The model actually lands where it was asked to. The cuda half skips when
     there is no GPU, which is every machine this project has run on so far.
"""

import hashlib

import numpy as np
import pytest
import torch

from tier2.adapter import (forecast_kwh, load_frozen_dlinear,
                           make_window_builder, store_manifest)


def _fingerprint(model, wb, n=256):
    sel = np.asarray(wb.splits["test"][:n], np.int64)
    dev = str(next(model.parameters()).device)
    b = wb.build(sel, device=dev)
    with torch.no_grad():
        f = forecast_kwh(model, b).cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest(), f


def test_default_device_is_cpu(store_dir):
    man = store_manifest(store_dir["data"])
    model, _, _ = load_frozen_dlinear(store_dir["ckpt"], man)
    assert next(model.parameters()).device.type == "cpu"


def test_cpu_placement_is_bit_identical(store_dir):
    """The regression that guards the reported numbers."""
    man = store_manifest(store_dir["data"])
    model, cfg, _ = load_frozen_dlinear(store_dir["ckpt"], man)
    wb = make_window_builder(store_dir["data"], cfg.lookback_len,
                             cfg.horizon_len)

    h_default, f_default = _fingerprint(model, wb)

    # Explicit device="cpu" must be indistinguishable from the default...
    model_cpu, _, _ = load_frozen_dlinear(store_dir["ckpt"], man,
                                          device="cpu")
    h_cpu, _ = _fingerprint(model_cpu, wb)
    assert h_cpu == h_default

    # ...and both must equal the pre-fix behaviour, which is reconstructed
    # here by stripping the placement back out: a model built and loaded with
    # no .to() at all. Byte equality, not allclose.
    ck = torch.load(store_dir["ckpt"], map_location="cpu", weights_only=False)
    from vendor.models_tier2 import TiDEConfig, build
    bare = build("dlinear", TiDEConfig(**ck["cfg"]))
    bare.load_state_dict(ck["state_dict"])
    bare.eval()
    h_bare, f_bare = _fingerprint(bare, wb)
    assert h_bare == h_default
    assert f_bare.tobytes() == f_default.tobytes()


def test_state_dict_is_unchanged_by_placement(store_dir):
    man = store_manifest(store_dir["data"])
    a, _, _ = load_frozen_dlinear(store_dir["ckpt"], man)
    b, _, _ = load_frozen_dlinear(store_dir["ckpt"], man, device="cpu")
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert sa[k].cpu().numpy().tobytes() == sb[k].cpu().numpy().tobytes(), k


def test_weights_are_frozen_regardless_of_device(store_dir):
    """Placement must not undo the requires_grad_(False) the loader promises."""
    man = store_manifest(store_dir["data"])
    model, _, _ = load_frozen_dlinear(store_dir["ckpt"], man, device="cpu")
    assert not any(p.requires_grad for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA on this box")
def test_cuda_placement_actually_runs(store_dir):
    """The thing that used to raise a device mismatch.

    Deliberately NOT an equality assertion against the CPU result: a GPU
    changes float summation order, and whether that difference is tolerable is
    a question for docs/status/reproduction.md's acceptance bands, not for a
    unit test. This asserts only that the path executes and produces finite
    numbers of the right shape.
    """
    man = store_manifest(store_dir["data"])
    model, cfg, _ = load_frozen_dlinear(store_dir["ckpt"], man, device="cuda")
    assert next(model.parameters()).device.type == "cuda"
    wb = make_window_builder(store_dir["data"], cfg.lookback_len,
                             cfg.horizon_len)
    _, f = _fingerprint(model, wb)
    assert f.shape[1] == cfg.horizon_len
    assert np.isfinite(f).all()
