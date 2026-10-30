/* edge_codec.h -- Tier-1 dual-prediction codec, AVR-targeted.
 *
 * A LINE-BY-LINE TRANSLITERATION of edge/codec.py, which is the specification.
 * Where the two could differ they are made to agree explicitly, and
 * export/gen_golden.py + test_host.c prove it on recorded vectors rather than
 * by inspection. This matters more here than in a normal port: a C codec that
 * is merely CLOSE to the Python one does not degrade gracefully, it
 * desynchronises from the NanoPi and the reconstruction becomes silently
 * wrong.
 *
 * No floating point on the data path. No division (the ATmega328P has no
 * hardware divide). No dynamic allocation.
 */
#ifndef EDGE_CODEC_H
#define EDGE_CODEC_H

#include <stdint.h>
#include <stdbool.h>
#include "edge_config.h"

#define SNAPSHOT_BYTES (2 * BUF_LEN + 2 + 4 * N_W + 2 + 2 + 2 + 1 + 1 + 4 + 4 + 2)

typedef struct {
    int16_t  buf[BUF_LEN];      /* ring of RECONSTRUCTED values            96 B */
    int16_t  day_true[24];      /* TRUE values of the current day          48 B */
    int32_t  w[N_W];            /* Q20 weights, int32 -- see edge_config.h  28 B */
    int16_t  carry[N_CARRY_MAX];/* taps the ring cannot reach               2 B */
    uint32_t day_tx;            /* transmit bitmap for the current day      4 B */
    uint32_t t;                 /* absolute hour of the NEXT sample         4 B */
    uint32_t n_seen;            /* samples consumed since commissioning     4 B */
    uint32_t t_start;                                                   /* 4 B */
    int16_t  u_hat;             /* prediction for the current hour          2 B */
    int16_t  s;                 /* EWMA of |e|                              2 B */
    uint16_t eps;               /* EPS_NEVER means alpha = inf              2 B */
    uint8_t  head;                                                      /* 1 B */
    uint8_t  n_tx_today;                                                /* 1 B */
} codec_t;                      /* measured figure is in firmware/README */

typedef struct {
    int16_t e;
    int16_t u_hat;
    int16_t u_recon;
    uint16_t eps_eff;
    bool tx;
    bool event;
    bool clamped;
} step_result_t;

void    codec_init(codec_t *st, uint16_t eps, int16_t u0, uint32_t t_start);
int16_t codec_predict(const codec_t *st);
void    codec_step(codec_t *st, int16_t u, bool clamped, step_result_t *out);
bool    codec_is_boundary(const codec_t *st, uint32_t t);
/* In sync_safe mode (SYNC_SAFE 1) the update runs on the RECONSTRUCTED ring and
 * `true_24` is ignored, so it may be NULL: no packet and no ACK are involved.
 * In legacy mode (SYNC_SAFE 0, which is what edge_config.h SHIPS -- it measured
 * both better and cheaper at every alpha, see firmware/README.md) it must be
 * the previous day's true values. See edge/codec.py. */
void    codec_day_boundary(codec_t *st, const int16_t *true_24);
void    codec_new_day(codec_t *st);
void    codec_resync_payload(const codec_t *st, int16_t *out24);
uint32_t codec_resync_bitmap(const codec_t *st);
void    codec_snapshot(const codec_t *st, uint8_t *out /* SNAPSHOT_BYTES */);

/* exposed for the host tests */
int32_t ec_sar32(int32_t x, uint8_t n);
int64_t ec_sar64(int64_t x, uint8_t n);
int32_t ec_round_shift(int64_t x, uint8_t n);
uint8_t ec_msb_index(uint64_t x);
int32_t ec_clamp(int32_t x, int32_t lo, int32_t hi);

#endif /* EDGE_CODEC_H */
