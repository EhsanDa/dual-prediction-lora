/* node_protocol.c -- transliteration of edge/protocol.py.
 *
 * Every function here has a named counterpart in the Python, and the two are
 * checked against each other by tests/test_node_protocol.py, which feeds the
 * same inputs to both and compares bytes. That test is the reason this file
 * can be trusted; reading it is not enough, for the same reason edge_codec.c
 * has golden vectors.
 */
#include <string.h>
#include "node_protocol.h"

/* CRC-16/CCITT-FALSE. Three lines on AVR, which is why it was chosen; the
 * polynomial matters only in that both ends agree, and both ends come from
 * edge/protocol.py::crc16. */
uint16_t np_crc16(const uint8_t *data, uint8_t n)
{
    uint16_t crc = 0xFFFF;
    uint8_t i, b;
    for (i = 0; i < n; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (b = 0; b < 8; b++)
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021)
                                 : (uint16_t)(crc << 1);
    }
    return crc;
}

/* CRC-8/ATM. 16 bits of CRC on a 7-byte downlink is not a sensible ratio. */
uint8_t np_crc8(const uint8_t *data, uint8_t n)
{
    uint8_t crc = 0, i, b;
    for (i = 0; i < n; i++) {
        crc ^= data[i];
        for (b = 0; b < 8; b++)
            crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07)
                               : (uint8_t)(crc << 1);
    }
    return crc;
}

uint8_t encode_exception(uint8_t *out, uint8_t cap, uint16_t node_id,
                         uint8_t hour_slot, uint8_t flags, int16_t e)
{
    uint16_t crc;
    if (cap < EXCEPTION_BYTES) return 0;
    out[0] = (uint8_t)((TYPE_EXCEPTION << 4) | PROTO_VERSION);
    out[1] = (uint8_t)(node_id & 0xFF);
    out[2] = (uint8_t)(node_id >> 8);
    out[3] = (uint8_t)((hour_slot & 0x1F) | ((flags & 0x07) << 5));
    out[4] = (uint8_t)((uint16_t)e & 0xFF);
    out[5] = (uint8_t)((uint16_t)e >> 8);
    crc = np_crc16(out, 6);
    out[6] = (uint8_t)(crc & 0xFF);
    out[7] = (uint8_t)(crc >> 8);
    return EXCEPTION_BYTES;
}

/* zigzag, over 32 bits, exactly as edge/protocol.py::_zigzag. The shift of a
 * negative value is implementation-defined in C89 but arithmetic on every
 * compiler this targets; avr-gcc is one, and edge_codec.c relies on the same
 * property and documents it. */
static uint32_t zigzag(int32_t v)
{
    return ((uint32_t)v << 1) ^ (uint32_t)(v >> 31);
}

static uint8_t put_varint(uint8_t *out, uint32_t z)
{
    uint8_t n = 0;
    for (;;) {
        uint8_t b = (uint8_t)(z & 0x7F);
        z >>= 7;
        out[n++] = (uint8_t)(b | (z ? 0x80 : 0x00));
        if (!z) return n;
    }
}

uint8_t encode_resync(uint8_t *out, uint8_t cap, uint16_t node_id,
                      uint8_t day_seq, uint8_t n_exceptions,
                      const int16_t *values24, uint32_t tx_bitmap,
                      const int32_t *weights)
{
    uint8_t  payload[2 + 23 * 3];
    uint8_t  plen = 0, n = 0, i;
    uint16_t crc;

    /* first value in full, then zigzag varint differences */
    payload[plen++] = (uint8_t)((uint16_t)values24[0] & 0xFF);
    payload[plen++] = (uint8_t)((uint16_t)values24[0] >> 8);
    for (i = 1; i < 24; i++)
        plen = (uint8_t)(plen + put_varint(payload + plen,
                         zigzag((int32_t)values24[i] - (int32_t)values24[i - 1])));

    if ((uint16_t)cap < (uint16_t)(6 + plen + 1 + 3 + 1 + 4 * N_W + 2)) return 0;

    out[n++] = (uint8_t)((TYPE_RESYNC << 4) | PROTO_VERSION);
    out[n++] = (uint8_t)(node_id & 0xFF);
    out[n++] = (uint8_t)(node_id >> 8);
    out[n++] = day_seq;
    out[n++] = n_exceptions;                 /* the caller clamps to 255 */
    out[n++] = plen;
    memcpy(out + n, payload, plen);
    n = (uint8_t)(n + plen);

    out[n++] = 0;                            /* status */
    out[n++] = (uint8_t)(tx_bitmap & 0xFF);
    out[n++] = (uint8_t)((tx_bitmap >> 8) & 0xFF);
    out[n++] = (uint8_t)((tx_bitmap >> 16) & 0xFF);
    out[n++] = (uint8_t)N_W;
    for (i = 0; i < N_W; i++) {
        uint32_t w = (uint32_t)weights[i];
        out[n++] = (uint8_t)(w & 0xFF);
        out[n++] = (uint8_t)((w >> 8) & 0xFF);
        out[n++] = (uint8_t)((w >> 16) & 0xFF);
        out[n++] = (uint8_t)((w >> 24) & 0xFF);
    }

    crc = np_crc16(out, n);
    out[n++] = (uint8_t)(crc & 0xFF);
    out[n++] = (uint8_t)(crc >> 8);
    return n;
}

bool decode_downlink(const uint8_t *in, uint8_t n, downlink_t *out)
{
    if (n != DOWNLINK_BYTES) return false;
    if (np_crc8(in, (uint8_t)(DOWNLINK_BYTES - 1)) != in[DOWNLINK_BYTES - 1])
        return false;
    if ((in[0] >> 4) != TYPE_DOWNLINK || (in[0] & 0x0F) != PROTO_VERSION)
        return false;
    out->ack_seq         = in[1];
    out->eps             = (uint16_t)(in[2] | ((uint16_t)in[3] << 8));
    out->command         = in[4];
    out->hour_correction = (int16_t)(in[5] | ((uint16_t)in[6] << 8));
    return true;
}
