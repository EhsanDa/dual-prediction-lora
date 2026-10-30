"""
test_node_protocol.py  --  firmware/node/node_protocol.c vs edge/protocol.py.

The C encoders on the node and the Python decoders on the gateway are two
implementations of one wire format, and nothing but agreement makes the system
work. edge_codec.c has golden vectors for exactly this reason; this is the
same argument applied to the packets.

The C is compiled into a shared object and driven through ctypes, so the test
exercises the ACTUAL source the firmware is built from rather than a
transcription of it. If gcc is missing the test skips rather than passing
vacuously.

Note this runs the C on x86-64, so it proves the two implementations agree on
the format. It does not prove the AVR build agrees -- that is what the link
test on real hardware does (firmware/node/README.md), and it is why the resync
frame is sent and decoded over the air rather than only compared here.
"""
import ctypes
import os
import shutil
import subprocess
import tempfile

import pytest

from edge import protocol as P

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NODE = os.path.join(REPO, "firmware", "node")


@pytest.fixture(scope="module")
def lib():
    if shutil.which("gcc") is None:
        pytest.skip("gcc not available; the C side cannot be built")
    tmp = tempfile.mkdtemp(prefix="node_proto_")
    so = os.path.join(tmp, "libnodeproto.so")
    src = os.path.join(NODE, "node_protocol.c")
    cmd = ["gcc", "-std=c99", "-O2", "-fPIC", "-shared",
           "-I", os.path.join(REPO, "firmware"), "-I", NODE, src, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"could not build node_protocol.c:\n{r.stderr}"

    h = ctypes.CDLL(so)
    h.np_crc16.restype = ctypes.c_uint16
    h.np_crc16.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint8]
    h.np_crc8.restype = ctypes.c_uint8
    h.np_crc8.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint8]
    h.encode_exception.restype = ctypes.c_uint8
    h.encode_exception.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint8,
                                   ctypes.c_uint16, ctypes.c_uint8,
                                   ctypes.c_uint8, ctypes.c_int16]
    h.encode_resync.restype = ctypes.c_uint8
    h.encode_resync.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint8,
                                ctypes.c_uint16, ctypes.c_uint8, ctypes.c_uint8,
                                ctypes.POINTER(ctypes.c_int16), ctypes.c_uint32,
                                ctypes.POINTER(ctypes.c_int32)]
    return h


def _buf(n=256):
    return (ctypes.c_uint8 * n)()


def _bytes(b, n):
    return bytes(bytearray(b[:n]))


# ---------------------------------------------------------------- CRCs
CRC_CASES = [b"", b"\x00", b"\xff", b"123456789", bytes(range(32)),
             b"\x12\x34\x56\x78\x9a\xbc"]


@pytest.mark.parametrize("data", CRC_CASES)
def test_crc16_matches_python(lib, data):
    buf = (ctypes.c_uint8 * max(len(data), 1))(*data) if data else (ctypes.c_uint8 * 1)()
    assert lib.np_crc16(buf, len(data)) == P.crc16(data), (
        "CRC-16 disagrees; the gateway would reject every frame the node sends")


@pytest.mark.parametrize("data", CRC_CASES)
def test_crc8_matches_python(lib, data):
    buf = (ctypes.c_uint8 * max(len(data), 1))(*data) if data else (ctypes.c_uint8 * 1)()
    assert lib.np_crc8(buf, len(data)) == P.crc8(data)


# ----------------------------------------------------------- exception
# e is deliberately taken to both int16 extremes: the residual is the one field
# that can legitimately be large and negative.
EXC_CASES = [(0, 0, 0, 0), (1, 7, P.FLAG_EVENT, -1234), (65535, 23, P.FLAG_CLAMP, 32767),
             (513, 5, P.FLAG_EVENT | P.FLAG_CLAMP, -32768), (7, 31, 0, 1)]


@pytest.mark.parametrize("node_id,slot,flags,e", EXC_CASES)
def test_exception_frame_is_byte_identical(lib, node_id, slot, flags, e):
    out = _buf()
    n = lib.encode_exception(out, 255, node_id, slot, flags, e)
    got = _bytes(out, n)
    want = P.encode_exception(P.Exception_(node_id=node_id, hour_slot=slot,
                                           flags=flags, e=e))
    assert got == want, f"C {got.hex()} != Python {want.hex()}"
    # and the Python decoder must accept what the C produced
    back = P.decode_exception(got)
    assert (back.node_id, back.hour_slot, back.flags, back.e) == (node_id, slot, flags, e)


# -------------------------------------------------------------- resync
RESYNC_CASES = [
    # flat: every delta zero, the smallest possible payload
    [3000] * 24,
    # the shape a real day has
    [2500 + i * 37 for i in range(24)],
    # large alternating deltas: multi-byte varints, both signs
    [3000 + (1200 if i % 2 else -1200) for i in range(24)],
    # the int16 extremes, which is where zigzag and the buffer bound are tested
    [-32000 if i % 2 else 32000 for i in range(24)],
]


@pytest.mark.parametrize("vals", RESYNC_CASES)
def test_resync_frame_is_byte_identical(lib, vals):
    from edge.codec import CodecParams
    n_w = CodecParams().n_w
    weights = [1000 * (i + 1) for i in range(n_w)]
    bitmap = 0x00A5A5

    out = _buf()
    cv = (ctypes.c_int16 * 24)(*vals)
    cw = (ctypes.c_int32 * n_w)(*weights)
    n = lib.encode_resync(out, 255, 513, 3, 11, cv, bitmap, cw)
    assert n > 0, "the C encoder refused a frame that fits"
    got = _bytes(out, n)

    want = P.encode_resync(P.Resync(
        node_id=513, day_seq=3, n_exceptions=11, values=tuple(vals),
        status=0, tx_bitmap=tuple(bool(bitmap >> b & 1) for b in range(24)),
        weights=tuple(weights)))
    assert got == want, f"C {got.hex()} != Python {want.hex()}"

    back = P.decode_resync(got)
    assert list(back.values) == vals
    assert list(back.weights) == weights


def test_the_resync_buffer_bound_is_not_optimistic(lib):
    """main.ino sizes its packet buffer from RESYNC_MAX_BYTES.

    It used to declare `uint8_t pkt[80]`, which is smaller than the frames this
    encoder actually produces -- never caught, because the file had never been
    compiled. The bound must hold for the worst case the format allows, which
    is alternating int16 extremes: every delta becomes a three-byte varint.
    """
    from edge.codec import CodecParams
    n_w = CodecParams().n_w
    bound = 6 + 2 + 23 * 3 + 1 + 3 + 1 + 4 * n_w + 2

    worst = 0
    for vals in RESYNC_CASES:
        frame = P.encode_resync(P.Resync(
            node_id=0, day_seq=0, n_exceptions=0, values=tuple(vals),
            tx_bitmap=tuple([False] * 24), weights=tuple([0] * n_w)))
        worst = max(worst, len(frame))
    assert worst <= bound, (
        f"a {worst}-byte frame exceeds RESYNC_MAX_BYTES ({bound})")
    assert worst > 80, (
        "the worst case no longer exceeds 80 B, so the bug this test pins "
        "would not reproduce -- check the fixture, not the bound")


def test_encoder_refuses_a_buffer_that_is_too_small(lib):
    """Returning 0 rather than overrunning is the whole point of the cap."""
    from edge.codec import CodecParams
    n_w = CodecParams().n_w
    out = _buf()
    cv = (ctypes.c_int16 * 24)(*([-32000 if i % 2 else 32000 for i in range(24)]))
    cw = (ctypes.c_int32 * n_w)(*([0] * n_w))
    assert lib.encode_resync(out, 80, 0, 0, 0, cv, 0, cw) == 0, (
        "the encoder wrote a worst-case frame into an 80-byte buffer")
