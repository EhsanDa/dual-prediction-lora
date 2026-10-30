"""
conftest.py  --  fixtures.

Two distinct sizes, on purpose (blueprint 7):

  `tiny`  T = 240, M = 3, hand-checkable. Three clients: a normal one, an
          exactly constant one, and one commissioned at hour 48. Used by the
          codec, mirror and protocol tests, where a failure has to be readable
          by eye.
  `store` T = 8760, M = 6, a real schema-valid store with a manifest, a
          checkpoint and WindowBuilder-legal splits. T cannot be shrunk much:
          WindowBuilder purges index_L + H = 744 hours from the tail of train
          and valid, so a non-empty VALIDATION split needs
          0.8T - 744 > 0.7T, i.e. T > 7440. A 2400-hour store silently
          produces zero train and zero valid windows, which is exactly the
          kind of thing that makes a green test suite meaningless.

Both are built once per session into a temp directory. Nothing in the suite
touches a real store or a real checkpoint, because there are none.
"""

import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    from sim.synth import make_tiny_fixture
    d = tmp_path_factory.mktemp("tiny")
    y, onboarding = make_tiny_fixture(str(d))
    return {"y": y, "onboarding": onboarding, "dir": str(d)}


@pytest.fixture(scope="session")
def store_dir(tmp_path_factory):
    from sim.synth import make_synthetic_checkpoint, make_synthetic_store
    d = tmp_path_factory.mktemp("store")
    spec = make_synthetic_store(str(d / "data"), T=8760, M=6, seed=3,
                                verbose=False)
    make_synthetic_checkpoint(str(d / "ck.pt"), spec, seed=3, verbose=False)
    return {"data": str(d / "data"), "ckpt": str(d / "ck.pt"), "spec": spec}


@pytest.fixture(scope="session")
def store(store_dir):
    from sim.run_codec import Store
    return Store.load(store_dir["data"])


def quantised(y, k, params):
    from edge.quantize import quantize_array
    u, cl = quantize_array(y, k, params.u_clamp)
    return u, cl
