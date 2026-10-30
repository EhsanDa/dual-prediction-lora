"""
test_contract.py  --  blueprint 9.1.

Everything here is about the boundary with the owner's code. If these pass, the
harness is talking to the real Tier-2 modules on their own terms.
"""

import json
import os

import numpy as np
import pytest
import torch

from tier2.adapter import (IndexConventionError, load_frozen_dlinear,
                           make_window_builder, probe_index_convention,
                           store_manifest)

REQUIRED_FILES = {
    "series_log1p.npy": np.float32,
    "series_true.npy": np.float64,
    "series_filled.npy": np.float32,
    "missing.npy": np.bool_,
    "calendar.npy": np.float32,
    "static.npy": np.float32,
    "onboarding.npy": np.int64,
}

MANIFEST_KEYS = ("train_end", "valid_end", "T", "M", "revin_sigma_floor",
                 "revin_eps", "norm_window", "feature_spec_hash",
                 "calendar_features")


def test_store_has_every_file_with_the_right_dtype(store_dir):
    d = store_dir["data"]
    for name, dt in REQUIRED_FILES.items():
        p = os.path.join(d, name)
        assert os.path.exists(p), name
        a = np.load(p)
        assert a.dtype == dt, (name, a.dtype, dt)


def test_store_shapes_agree_with_the_manifest(store_dir):
    man = store_manifest(store_dir["data"])
    T, M = man["T"], man["M"]
    for name in ("series_log1p.npy", "series_true.npy", "series_filled.npy",
                 "missing.npy"):
        assert np.load(os.path.join(store_dir["data"], name)).shape == (T, M), name
    assert np.load(os.path.join(store_dir["data"], "calendar.npy")).shape == (T, 12)
    assert np.load(os.path.join(store_dir["data"], "static.npy")).shape[0] == M
    assert np.load(os.path.join(store_dir["data"], "onboarding.npy")).shape == (M,)


def test_manifest_has_the_keys_the_harness_reads(store_dir):
    man = store_manifest(store_dir["data"])
    for k in MANIFEST_KEYS:
        assert k in man, k
    assert 0 < man["train_end"] < man["valid_end"] < man["T"]
    assert man["revin_sigma_floor"] == 0.05
    assert man["revin_eps"] == 1e-5
    assert man["norm_window"] == 336


def test_series_true_is_nan_exactly_where_missing(store_dir):
    """C5. If these ever disagree, the metrics are being scored against
    interpolated values somewhere."""
    d = store_dir["data"]
    true = np.load(os.path.join(d, "series_true.npy"))
    miss = np.load(os.path.join(d, "missing.npy"))
    assert np.array_equal(np.isnan(true), miss)
    filled = np.load(os.path.join(d, "series_filled.npy"))
    assert np.isfinite(filled).all(), "series_filled must never be NaN"


def test_log1p_is_consistent_with_filled(store_dir):
    d = store_dir["data"]
    filled = np.load(os.path.join(d, "series_filled.npy"))
    log1p = np.load(os.path.join(d, "series_log1p.npy"))
    ref = np.log1p(np.clip(filled, 0, None)).astype(np.float32)
    assert np.abs(ref - log1p).max() < 1e-6


def test_onboarding_marks_exact_zeros_before_it(store_dir):
    d = store_dir["data"]
    filled = np.load(os.path.join(d, "series_filled.npy"))
    onb = np.load(os.path.join(d, "onboarding.npy"))
    for c in range(filled.shape[1]):
        if onb[c] > 0:
            assert np.all(filled[:onb[c], c] == 0.0), c


def test_window_builder_accepts_the_store_unmodified(store_dir):
    wb = make_window_builder(store_dir["data"])
    counts = wb.counts()
    assert counts["train"] > 0 and counts["test"] > 0
    b = wb.build(wb.splits["test"][:16])
    assert b["y_past"].shape[1] == wb.L
    assert b["y_true_kwh"].shape[1] == wb.H
    assert b["valid"].dtype == torch.bool


def test_checkpoint_contract_is_enforced(store_dir, tmp_path):
    man = store_manifest(store_dir["data"])
    model, cfg, ck = load_frozen_dlinear(store_dir["ckpt"], man)
    assert cfg.lookback_len == 336 and cfg.horizon_len == 24
    assert ck["model"] == "dlinear"
    assert not any(p.requires_grad for p in model.parameters()), (
        "the Tier-2 model is FROZEN; a trainable parameter here means the "
        "harness could be adapting the thing it is supposed to be measuring")

    # a mismatched feature-spec hash must RAISE, not warn
    bad = dict(ck)
    bad["feature_spec_hash"] = "0" * 64
    p = tmp_path / "bad.pt"
    torch.save(bad, p)
    with pytest.raises(AssertionError):
        load_frozen_dlinear(str(p), man)

    # so must the wrong model name and the wrong look-back
    bad2 = dict(ck)
    bad2["model"] = "tide_lite"
    p2 = tmp_path / "bad2.pt"
    torch.save(bad2, p2)
    with pytest.raises(AssertionError):
        load_frozen_dlinear(str(p2), man)


def test_index_probe_passes_and_fires_on_a_wrong_convention(store_dir):
    """Blueprint 9.1: the probe must FAIL on a deliberately wrong convention.
    A probe that cannot fail is a comment."""
    wb = make_window_builder(store_dir["data"])
    out = probe_index_convention(wb)
    assert out["lookback_max_dev"] < 1e-5

    class Shifted:
        """WindowBuilder with sel shifted by one hour: the classic
        start-of-window versus origin confusion."""

        def __init__(self, wb):
            self._wb = wb
            self.splits = {k: v.copy() for k, v in wb.splits.items()}
            for v in self.splits.values():
                v[:, 0] -= 1

        def __getattr__(self, n):
            return getattr(self._wb, n)

        def build(self, sel, device="cpu"):
            s = sel.copy()
            s[:, 0] += 1
            return self._wb.build(s, device=device)

    with pytest.raises(IndexConventionError):
        probe_index_convention(Shifted(wb))


def test_feature_spec_hash_recipe_matches_the_preprocessor(store_dir):
    """The synthetic manifest must hash the same keys, the same way, as the
    owner's 10_tier2_preprocess.py -- otherwise a real checkpoint would be
    rejected against a real store for a reason invented here."""
    import hashlib
    man = store_manifest(store_dir["data"])
    h = hashlib.sha256(json.dumps(
        {k: man[k] for k in ("feature_spec_version", "dynamic_features",
                             "window_features", "static_features",
                             "target_transform", "revin_sigma_floor",
                             "revin_eps", "norm_window")},
        sort_keys=True).encode()).hexdigest()
    assert h == man["feature_spec_hash"]


def test_synthetic_store_is_labelled_as_such(store_dir):
    """Nothing built on synthetic data may be mistakable for a result."""
    man = store_manifest(store_dir["data"])
    assert man.get("SYNTHETIC") is True
    ck = torch.load(store_dir["ckpt"], map_location="cpu", weights_only=False)
    assert ck.get("SYNTHETIC") is True
