/* node_protocol.h -- the wire formats, C side.
 *
 * A LINE-BY-LINE TRANSLITERATION of edge/protocol.py, which is the
 * specification -- the same relationship edge_codec.c has to edge/codec.py.
 * Little-endian throughout; the CRC covers everything preceding it.
 */
#ifndef NODE_PROTOCOL_H
#define NODE_PROTOCOL_H

#include <stdint.h>
#include <stdbool.h>
#include "../edge_config.h"

#define PROTO_VERSION    1
#define TYPE_EXCEPTION   0x1
#define TYPE_RESYNC      0x2
#define TYPE_DOWNLINK    0x3

#define FLAG_EVENT       0x1
#define FLAG_CLAMP       0x2

#define CMD_NONE             0
#define CMD_FULL_DUMP        1
#define CMD_SET_N_MAX_TX     2
#define CMD_CLOCK_CORRECTION 3

#define EXCEPTION_BYTES  8
#define DOWNLINK_BYTES   8

/* THE RESYNC BUFFER SIZE, DERIVED RATHER THAN GUESSED.
 *
 * main.ino used to declare `uint8_t pkt[80]`, which is too small and was never
 * caught because the file had never been compiled, let alone run. The frame is:
 *
 *     header      6   type|ver, node_id:2, day_seq, n_exc, payload_len
 *     payload         2 for values[0], then 23 zigzag varints
 *     status      1
 *     tx_bitmap   3
 *     n_w         1
 *     weights     4 * N_W
 *     crc         2
 *
 * A zigzag varint of an int16 delta is at most 3 bytes, so the payload's
 * ceiling is 2 + 23*3 = 71. Total worst case is 6 + 71 + 1 + 3 + 1 + 4*N_W + 2.
 * Typical is ~83 B, which is the figure Table II quotes for airtime.
 */
#define RESYNC_MAX_BYTES (6u + 2u + 23u * 3u + 1u + 3u + 1u + 4u * N_W + 2u)

uint16_t np_crc16(const uint8_t *data, uint8_t n);   /* CRC-16/CCITT-FALSE */
uint8_t  np_crc8(const uint8_t *data, uint8_t n);    /* CRC-8/ATM, poly 0x07 */

/* Returns the encoded length, or 0 if it would not fit in `cap`. */
uint8_t encode_exception(uint8_t *out, uint8_t cap, uint16_t node_id,
                         uint8_t hour_slot, uint8_t flags, int16_t e);

uint8_t encode_resync(uint8_t *out, uint8_t cap, uint16_t node_id,
                      uint8_t day_seq, uint8_t n_exceptions,
                      const int16_t *values24, uint32_t tx_bitmap,
                      const int32_t *weights);

typedef struct {
    uint8_t  ack_seq;
    uint16_t eps;
    uint8_t  command;
    int16_t  hour_correction;
} downlink_t;

bool decode_downlink(const uint8_t *in, uint8_t n, downlink_t *out);

#endif /* NODE_PROTOCOL_H */
