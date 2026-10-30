"""
protocol.py  --  packet formats and LoRa airtime. Blueprint 5.3 and 5.4.

Raw point-to-point LoRa, no LoRaWAN. Little-endian throughout. CRC covers
everything preceding it.

WHY THE RESIDUAL AND NEVER THE RAW VALUE
========================================
`e` is an order of magnitude smaller than `u`, and the NanoPi recovers
u = u_hat + e EXACTLY because u_hat is bit-identical on both sides. So every
exception packet is simultaneously a data packet and a continuous check on
synchronisation: if the two predictors had drifted, the reconstructed value
would be wrong, and it would be wrong in a way the daily RESYNC exposes
immediately. Sending the raw value would be a byte cheaper in the worst case
and would throw that property away.

AIRTIME IS COMPUTED, NEVER TABULATED
====================================
`time_on_air` implements the Semtech formula, including the low-data-rate
optimisation that is mandatory at SF11/SF12 on 125 kHz. Copying numbers out of
an online calculator would make the duty-cycle and channel-occupancy results in
eval/ unauditable, and those results are the scalability argument.
"""

import struct
from dataclasses import dataclass

PROTO_VERSION = 1

TYPE_EXCEPTION = 0x1
TYPE_RESYNC = 0x2
TYPE_DOWNLINK = 0x3

FLAG_EVENT = 0x1
FLAG_CLAMP = 0x2
FLAG_RESERVED = 0x4

CMD_NONE = 0
CMD_FULL_DUMP = 1
CMD_SET_N_MAX_TX = 2
CMD_CLOCK_CORRECTION = 3

EXCEPTION_BYTES = 8
DOWNLINK_BYTES = 8


# ----------------------------------------------------------------------- CRC
def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE. Chosen because it is three lines on AVR and the
    NanoPi side is stdlib-free too; the exact polynomial matters only in that
    both ends agree, and both ends come from this function."""
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def crc8(data: bytes) -> int:
    """CRC-8/ATM (poly 0x07), for the 8-byte downlink where 16 bits of CRC on a
    7-byte payload is not a sensible ratio."""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


class CrcError(ValueError):
    pass


# ------------------------------------------------------------------ packets
@dataclass
class Exception_:
    node_id: int
    hour_slot: int          # 0..23, 5 bits
    flags: int              # EVENT | CLAMP
    e: int                  # int16 residual

    @property
    def event(self):
        return bool(self.flags & FLAG_EVENT)

    @property
    def clamp(self):
        return bool(self.flags & FLAG_CLAMP)


@dataclass
class Resync:
    """The daily packet. Revision R4 added the last two fields.

    `tx_bitmap` and `weights` are what turn the RESYNC from a fidelity packet
    into a REPAIR packet. With them the hub can replay the node's day exactly
    (true value where the node transmitted, its own prediction where it did
    not) and then adopt the node's weights, so a single delivered RESYNC
    restores the shared state no matter what was lost in between -- and no ACK
    is needed for correctness at all. They cost 3 + 2*n_w bytes once a day.
    """
    node_id: int
    day_seq: int            # day counter mod 256
    n_exceptions: int       # exceptions the NODE sent this day
    values: tuple           # 24 TRUE device-unit values, oldest first
    status: int = 0         # bit0 clamp seen, bit1 event seen, bit2 eps doubled
    tx_bitmap: tuple = ()   # 24 bools: did the node transmit that hour
    weights: tuple = ()     # the node's post-update weight vector


@dataclass
class Downlink:
    ack_seq: int
    eps: int                # uint16; 0xFFFF means "no exception threshold"
    command: int = CMD_NONE
    hour_correction: int = 0    # signed, in seconds


def encode_exception(p: Exception_) -> bytes:
    assert 0 <= p.hour_slot < 32 and 0 <= p.flags < 8
    body = struct.pack("<BHBh", (TYPE_EXCEPTION << 4) | PROTO_VERSION, p.node_id,
                       (p.hour_slot & 0x1F) | ((p.flags & 0x7) << 5), p.e)
    return body + struct.pack("<H", crc16(body))


def decode_exception(b: bytes) -> Exception_:
    assert len(b) == EXCEPTION_BYTES, f"exception packet is {EXCEPTION_BYTES} bytes"
    body, crc = b[:-2], struct.unpack("<H", b[-2:])[0]
    if crc != crc16(body):
        raise CrcError("exception CRC mismatch")
    tv, node, hf, e = struct.unpack("<BHBh", body)
    assert tv >> 4 == TYPE_EXCEPTION and tv & 0xF == PROTO_VERSION
    return Exception_(node_id=node, hour_slot=hf & 0x1F, flags=(hf >> 5) & 0x7, e=e)


def _zigzag(v: int) -> int:
    return (v << 1) ^ (v >> 31)


def _unzigzag(z: int) -> int:
    return (z >> 1) ^ -(z & 1)


def _varint(z: int) -> bytes:
    out = bytearray()
    while True:
        b = z & 0x7F
        z >>= 7
        out.append(b | (0x80 if z else 0))
        if not z:
            return bytes(out)


def _read_varint(b: bytes, i: int):
    z, sh = 0, 0
    while True:
        v = b[i]
        i += 1
        z |= (v & 0x7F) << sh
        if not v & 0x80:
            return z, i
        sh += 7


def encode_resync(p: Resync) -> bytes:
    """Delta-coded: first value in full, then zigzag varint differences.

    Blueprint 5.3 budgets ~24-28 bytes for the 24 values. Hourly load deltas
    are small in device units, so most differences fit one byte; a day with a
    large step costs two. The encoding is length-prefixed so a decoder never
    has to guess, and the packet size is REPORTED per packet rather than
    assumed -- eval/ sums actual encoded lengths for the airtime figures.
    """
    assert len(p.values) == 24
    payload = bytearray(struct.pack("<h", int(p.values[0])))
    for a, b in zip(p.values, p.values[1:]):
        payload += _varint(_zigzag(int(b) - int(a)))
    bm = 0
    for j, v in enumerate(p.tx_bitmap):
        if v:
            bm |= 1 << j
    tail = bytes([bm & 0xFF, (bm >> 8) & 0xFF, (bm >> 16) & 0xFF,
                  len(p.weights)])
    tail += b"".join(struct.pack("<i", int(w)) for w in p.weights)
    body = (struct.pack("<BHBBB", (TYPE_RESYNC << 4) | PROTO_VERSION, p.node_id,
                        p.day_seq & 0xFF, min(p.n_exceptions, 255), len(payload))
            + bytes(payload) + struct.pack("<B", p.status) + tail)
    return body + struct.pack("<H", crc16(body))


def decode_resync(b: bytes) -> Resync:
    body, crc = b[:-2], struct.unpack("<H", b[-2:])[0]
    if crc != crc16(body):
        raise CrcError("resync CRC mismatch")
    tv, node, day, nexc, plen = struct.unpack("<BHBBB", body[:6])
    assert tv >> 4 == TYPE_RESYNC and tv & 0xF == PROTO_VERSION
    payload = body[6:6 + plen]
    v0 = struct.unpack("<h", payload[:2])[0]
    vals, i, cur = [v0], 2, v0
    while len(vals) < 24:
        z, i = _read_varint(payload, i)
        cur += _unzigzag(z)
        vals.append(cur)
    j = 6 + plen
    status = body[j]
    bm = body[j + 1] | (body[j + 2] << 8) | (body[j + 3] << 16)
    nw = body[j + 4]
    ws = tuple(struct.unpack("<i", body[j + 5 + 4 * i:j + 9 + 4 * i])[0]
               for i in range(nw))
    return Resync(node_id=node, day_seq=day, n_exceptions=nexc,
                  values=tuple(vals), status=status,
                  tx_bitmap=tuple(bool(bm >> b & 1) for b in range(24)),
                  weights=ws)


def encode_downlink(p: Downlink) -> bytes:
    body = struct.pack("<BBHBh", (TYPE_DOWNLINK << 4) | PROTO_VERSION,
                       p.ack_seq & 0xFF, p.eps & 0xFFFF, p.command,
                       int(p.hour_correction))
    return body + struct.pack("<B", crc8(body))


def decode_downlink(b: bytes) -> Downlink:
    assert len(b) == DOWNLINK_BYTES
    body, crc = b[:-1], b[-1]
    if crc != crc8(body):
        raise CrcError("downlink CRC mismatch")
    tv, ack, eps, cmd, corr = struct.unpack("<BBHBh", body)
    assert tv >> 4 == TYPE_DOWNLINK and tv & 0xF == PROTO_VERSION
    return Downlink(ack_seq=ack, eps=eps, command=cmd, hour_correction=corr)


# ------------------------------------------------------------------ airtime
def time_on_air(payload_bytes: int, sf: int, bw: float = 125e3, cr: int = 1,
                preamble: int = 8, explicit_header: bool = True,
                crc_on: bool = True, low_dr_opt=None) -> float:
    """LoRa time on air in seconds. Semtech SX1276 datasheet section 4.1.1.6.

    cr is the coding-rate index 1..4, meaning 4/(4+cr).
    low_dr_opt defaults to the mandatory setting: enabled for SF11/SF12 at
    125 kHz, where the symbol time exceeds 16 ms and the datasheet requires it.
    """
    assert 6 <= sf <= 12 and 1 <= cr <= 4
    de = ((sf >= 11 and bw <= 125e3) if low_dr_opt is None else low_dr_opt)
    t_sym = (2.0 ** sf) / bw
    t_preamble = (preamble + 4.25) * t_sym
    num = (8 * payload_bytes - 4 * sf + 28 + (16 if crc_on else 0)
           - (0 if explicit_header else 20))
    den = 4 * (sf - (2 if de else 0))
    n_payload = 8 + max(0, -(-num // den)) * (cr + 4)   # ceil division
    return t_preamble + n_payload * t_sym


def duty_cycle(airtime_s: float, window_s: float = 3600.0) -> float:
    return airtime_s / window_s


def aloha_collision_prob(n_nodes: int, packets_per_node_per_day: float,
                         airtime_s: float) -> float:
    """Pure-ALOHA collision probability at the given offered load.

    G = offered traffic in packets per vulnerable period (2 x airtime).
    P(collision) = 1 - exp(-2G_packets). Pure ALOHA, not slotted: nothing in
    this system is time-synchronised at packet granularity, and pretending
    otherwise would flatter the scalability numbers.
    """
    rate = n_nodes * packets_per_node_per_day / 86400.0     # packets per second
    return 1.0 - float(pow(2.718281828459045, -2.0 * rate * airtime_s))


def channel_occupancy(n_nodes: int, packets_per_node_per_day: float,
                      mean_payload_bytes: float, sf: int) -> float:
    """Fraction of wall-clock time the shared channel is busy."""
    return (n_nodes * packets_per_node_per_day
            * time_on_air(mean_payload_bytes, sf) / 86400.0)
