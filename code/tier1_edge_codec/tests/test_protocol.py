"""test_protocol.py  --  packet formats and airtime."""

import numpy as np
import pytest

from edge.protocol import (DOWNLINK_BYTES, EXCEPTION_BYTES, CrcError, Downlink,
                           Exception_, FLAG_CLAMP, FLAG_EVENT, Resync, crc8,
                           crc16, decode_downlink, decode_exception,
                           decode_resync, encode_downlink, encode_exception,
                           encode_resync, time_on_air)
from sim.run_codec import resync_nbytes


def test_exception_round_trip_and_size():
    for e in (0, 1, -1, 32767, -32768, 1234):
        p = Exception_(node_id=513, hour_slot=23, flags=FLAG_EVENT | FLAG_CLAMP, e=e)
        b = encode_exception(p)
        assert len(b) == EXCEPTION_BYTES == 8
        q = decode_exception(b)
        assert (q.node_id, q.hour_slot, q.flags, q.e) == (513, 23, p.flags, e)
        assert q.event and q.clamp


def test_resync_round_trip_over_awkward_payloads():
    rng = np.random.default_rng(0)
    cases = [tuple(range(24)),
             tuple([0] * 24),
             tuple([32000] * 24),
             tuple(rng.integers(0, 12000, 24).tolist()),
             tuple(np.linspace(0, 30000, 24).astype(int).tolist()),
             tuple((-1) ** np.arange(24) * 9000)]
    for vals in cases:
        p = Resync(node_id=7, day_seq=200, n_exceptions=5,
                   values=tuple(int(v) for v in vals), status=3)
        b = encode_resync(p)
        q = decode_resync(b)
        assert q.values == p.values, vals
        assert (q.node_id, q.day_seq, q.n_exceptions, q.status) == (7, 200, 5, 3)


def test_resync_size_model_matches_the_encoder():
    """sim/ counts bytes with resync_nbytes for speed. If that drifts from the
    encoder, every airtime and duty-cycle number silently drifts with it."""
    rng = np.random.default_rng(1)
    for _ in range(300):
        vals = tuple(int(v) for v in rng.integers(0, 20000, 24))
        bm = tuple(bool(b) for b in rng.integers(0, 2, 24))
        w = tuple(int(v) for v in rng.integers(-(4 << 20), 4 << 20, 7))
        p = Resync(node_id=1, day_seq=1, n_exceptions=0, values=vals,
                   tx_bitmap=bm, weights=w)
        assert len(encode_resync(p)) == resync_nbytes(vals, len(w)), vals


def test_resync_size_measured_not_assumed():
    """MEASURED, and it does not match the blueprint's budget.

    Blueprint 5.3 budgets "~24-28" bytes for the 24 values, i.e. ~34 for the
    packet. That assumes each delta fits one varint byte, which needs |delta|
    below 64 device units. At the shipped target_max = 12000 a typical meter
    swings hundreds of device units per hour, so most deltas take two bytes and
    the packet lands near 50. Nothing downstream is affected -- every airtime
    and duty-cycle figure is computed from real encoded lengths, never from the
    budget -- but the prose figure should be corrected to ~50 bytes.

    A coarse series (small deltas) still hits the budgeted size, which is the
    other half of the measurement.

    Revision R4 then added the repair fields (3 bitmap + 1 length + the weight
    vector), and the final revision widened the weights to int32 and added a
    seventh tap, so they are 28 bytes: a typical packet is ~85. That is the price of making
    synchronisation unconditional, it is paid once a day, and the airtime table
    reports it from real encoded lengths.
    """
    t = np.arange(24)
    swingy = tuple((6000 + 800 * np.sin(2 * np.pi * t / 24)).astype(int).tolist())
    n_swingy = resync_nbytes(swingy)
    assert 75 <= n_swingy <= 95, n_swingy

    flat = tuple((100 + (t % 3)).tolist())        # deltas of 0, 1, -2
    assert resync_nbytes(flat) <= 70, resync_nbytes(flat)

    # and the size model still agrees with the encoder in both regimes
    for vals in (swingy, flat):
        p = Resync(node_id=1, day_seq=0, n_exceptions=0, values=vals,
                   tx_bitmap=tuple([False] * 24), weights=(0,) * 7)
        assert len(encode_resync(p)) == resync_nbytes(vals)


def test_downlink_round_trip():
    p = Downlink(ack_seq=17, eps=0xFFFF, command=2, hour_correction=-120)
    b = encode_downlink(p)
    assert len(b) == DOWNLINK_BYTES == 8
    q = decode_downlink(b)
    assert (q.ack_seq, q.eps, q.command, q.hour_correction) == (17, 0xFFFF, 2, -120)


def test_crc_catches_single_bit_flips():
    p = encode_exception(Exception_(1, 3, 0, 500))
    for byte in range(len(p) - 2):
        for bit in range(8):
            bad = bytearray(p)
            bad[byte] ^= 1 << bit
            with pytest.raises((CrcError, AssertionError)):
                decode_exception(bytes(bad))
    d = encode_downlink(Downlink(1, 100))
    bad = bytearray(d)
    bad[2] ^= 0x01
    with pytest.raises(CrcError):
        decode_downlink(bytes(bad))


def test_crc_reference_vectors():
    """CRC-16/CCITT-FALSE and CRC-8/ATM against their published check values,
    so the C port has something to agree WITH rather than just agreeing with a
    second copy of the same possibly-wrong loop."""
    assert crc16(b"123456789") == 0x29B1
    assert crc8(b"123456789") == 0xF4


def test_time_on_air_is_computed_and_ordered():
    """Semtech's formula. Airtime must rise with SF and with payload, and the
    low-data-rate optimisation must actually engage at SF11/12."""
    a7 = time_on_air(8, 7)
    a12 = time_on_air(8, 12)
    assert a12 > 20 * a7
    assert time_on_air(34, 9) > time_on_air(8, 9)
    on = time_on_air(20, 12, low_dr_opt=True)
    off = time_on_air(20, 12, low_dr_opt=False)
    assert on >= off, "LDR optimisation should not shorten the packet"
    assert time_on_air(20, 11) == time_on_air(20, 11, low_dr_opt=True), (
        "LDR must default to ON at SF11/125 kHz")


def test_time_on_air_against_hand_computed_case():
    """SF7, BW125, CR 4/5, 8-symbol preamble, explicit header, CRC on,
    13-byte payload. Symbol time is 2**7/125000 = 1.024 ms; preamble is
    12.25 symbols; payload symbols = 8 + ceil((8*13 - 4*7 + 28 + 16)/(4*7))*5
    = 8 + ceil(120/28)*5 = 8 + 5*5 = 33. Total 45.25 symbols = 46.336 ms."""
    assert abs(time_on_air(13, 7) - 0.046336) < 1e-6


def test_exception_is_much_cheaper_than_a_raw_value_packet():
    """The residual is the payload precisely because it is small; this pins the
    packet at 8 bytes so a future field addition has to be deliberate."""
    assert EXCEPTION_BYTES == 8
    assert time_on_air(EXCEPTION_BYTES, 12) < 1.0
