/* main.ino -- the Arduino Nano application. Blueprint 10.3.
 *
 * 2026-09-28: this file COMPILES AND LINKS. Until then it was a skeleton whose
 * seven hardware symbols had no implementation -- and two of them, encode_resync
 * and apply_downlink, had no DECLARATION either, so it failed at compile before
 * the linker was ever reached. The implementations are in node_hw.c,
 * node_protocol.c and sx127x.c; the declarations now arrive with the headers
 * below. A buffer that was too small for its own protocol is also fixed; see
 * daily_boundary().
 *
 * SCHEDULING, SLEEP, METER READ AND RADIO ONLY. Every piece of arithmetic
 * lives in edge_codec.c, which knows nothing about hardware and is therefore
 * testable on the host against the Python reference (firmware/test_host.c).
 * Keep it that way: the moment a codec decision migrates into this file, the
 * equivalence test stops covering it.
 *
 * RADIO: raw point-to-point LoRa over a thin SX127x driver, NOT LoRaWAN. An
 * LMIC stack needs roughly 1.5-2 KB of SRAM and ~30 KB of Flash and does not
 * fit alongside the application on a 2 KB / 32 KB part. There is one gateway
 * per building, so LoRaWAN's join, addressing and network-server machinery buy
 * nothing here. This constraint invalidates a lot of otherwise-plausible
 * designs and belongs in the write-up.
 *
 * CLOCK: the Nano's resonator drifts on the order of minutes per day, and the
 * epsilon guarantee depends on both sides agreeing which hour a sample belongs
 * to. Samples are therefore indexed by SEQUENCE NUMBER, not by local time, and
 * the phase is corrected in the daily downlink. A DS3231 RTC is recommended for
 * deployment. This is unmodelled in the simulation and is a recorded
 * limitation, not an oversight.
 */

#include <stdint.h>
#include <stdbool.h>
extern "C" {
#include "../edge_codec.h"
#include "node_config.h"
#include "node_protocol.h"   /* encode_exception, encode_resync, the CRCs */
#include "sx127x.h"
}

/* ---- application state ------------------------------------------------- */
static codec_t   g_codec;
static uint32_t  g_seq = 0;          /* hours since commissioning */
static uint8_t   g_day_seq = 0;

/* ---- hardware shims, implemented in node_hw.c against sx127x.c ----------
 * These five are declared here and defined there. encode_resync and
 * apply_downlink used to be called below with no declaration anywhere, which
 * is what stopped this file compiling; they now come from node_protocol.h and
 * the declaration just below it. */
extern "C" {
extern float    meter_read_kwh(void);            /* replays the dataset trace */
extern bool     radio_send(const uint8_t *p, uint8_t n);
extern bool     radio_wait_downlink(uint8_t *p, uint8_t n, uint16_t timeout_ms);
extern void     sleep_until_next_hour(void);
extern uint16_t crc16_hw(const uint8_t *p, uint8_t n);
extern void     apply_downlink(const uint8_t *dl, codec_t *st);

/* Bench instrumentation, empty in a deployed build. It reads the hour's
 * result; it never computes one. */
extern void     bench_hour(uint32_t seq, const step_result_t *r);

/* The threshold the node commissions with. In a deployed build this is
 * edge_config.h's EPS_INIT, the per-meter constant gen_header.py emits. It is
 * a function rather than the macro because the bench replays a trace recorded
 * at a DIFFERENT eps, and because the paper's own design has eps retuned over
 * the downlink -- so the commissioning value was never something this file
 * should hard-code. See node_hw.c. */
extern uint16_t commissioning_eps(void);
}

/* Quantisation is the ONE place a float is allowed (blueprint 4.1): it is the
 * boundary between the meter and the codec. Round half away from zero, to match
 * C's round() and therefore the Python reference. */
static int16_t quantise(float kwh, bool *clamped)
{
    float scaled;
    long v;
    if (!(kwh == kwh)) kwh = 0.0f;                 /* NaN guard */
#if SCALE_K >= 0
    scaled = kwh / (float)(1UL << SCALE_K);
#else
    scaled = kwh * (float)(1UL << (-SCALE_K));
#endif
    v = (long)(scaled >= 0 ? scaled + 0.5f : scaled - 0.5f);
    *clamped = false;
    if (v >  U_CLAMP) { v =  U_CLAMP; *clamped = true; }
    if (v < -U_CLAMP) { v = -U_CLAMP; *clamped = true; }
    return (int16_t)v;
}

/* The frame is built by node_protocol.c's encoder, not open-coded here. This
 * file used to lay the eight bytes out by hand, which was correct but was a
 * SECOND implementation of a format whose first one is edge/protocol.py --
 * exactly the arrangement that drifts silently. One encoder, checked against
 * the Python by tests/test_node_protocol.py. */
static void send_exception(const step_result_t *r)
{
    uint8_t p[EXCEPTION_BYTES];
    uint8_t n = encode_exception(p, sizeof(p), NODE_ID,
                                 (uint8_t)(g_seq % 24u),
                                 (uint8_t)((r->event   ? FLAG_EVENT : 0)
                                         | (r->clamped ? FLAG_CLAMP : 0)),
                                 r->e);
    radio_send(p, n);
    /* No downlink window for exceptions: the node must not sit in RX, and the
     * daily resync is what repairs any loss. */
}

/* The daily boundary. THE ORDER MATTERS, and revision R4 changed it.
 *
 * The weight update runs FIRST and UNCONDITIONALLY -- it operates on the
 * reconstructed ring, which the hub also holds, so no packet and no ACK are
 * involved. The RESYNC is emitted AFTER, so it carries the POST-update weights
 * along with the day's true values and the transmit bitmap. With those three
 * the hub can replay the node's day exactly and adopt its weights, so a single
 * delivered RESYNC repairs everything however much was lost.
 *
 * There is consequently NO ACK REQUIREMENT and no RX window on the critical
 * path. The downlink is opportunistic and carries configuration only: a new
 * eps, a clock correction, a command. If it never arrives, nothing breaks. That
 * matters for power as much as for correctness -- an RX window per day is
 * affordable, one per exception was not. */
static void daily_boundary(void)
{
    int16_t payload[24];
    /* WAS `uint8_t pkt[80]`, AND 80 IS TOO SMALL. The resync frame is header 6
     * + payload (2 + 23 zigzag varints) + status 1 + bitmap 3 + n_w 1 + 4*N_W
     * weight bytes + CRC 2. That is ~83 B typically -- the very figure Table II
     * quotes for its airtime -- and up to RESYNC_MAX_BYTES worst case. The old
     * buffer would have been overrun on the first day boundary. It was never
     * caught because this file had never been compiled, let alone run.
     * The size is now derived from the protocol in node_protocol.h. */
    uint8_t pkt[RESYNC_MAX_BYTES], dl[DOWNLINK_BYTES];
    uint8_t n, attempt;
    uint8_t n_tx_prev = g_codec.n_tx_today;
    uint32_t bitmap;

    codec_resync_payload(&g_codec, payload);
    bitmap = codec_resync_bitmap(&g_codec);
    codec_day_boundary(&g_codec, payload);      /* unconditional */

    /* edge/protocol.py clamps n_exceptions with min(.., 255); here it is
     * already a uint8_t, because N_MAX_TX is 24 and codec_t.n_tx_today is one
     * byte, so the clamp cannot fire and is left out rather than written as
     * dead code. */
    n = encode_resync(pkt, sizeof(pkt), NODE_ID, g_day_seq, n_tx_prev,
                      payload, bitmap, g_codec.w);
    /* 0 means it did not fit. Sending a truncated resync is worse than sending
     * none: the hub would take a CRC failure for a lost frame, and the repair
     * path would look like loss rather than a bug. */
    if (n)
        for (attempt = 0; attempt < 3; attempt++)
            if (radio_send(pkt, n)) break;

    /* Opportunistic, not required. */
    if (radio_wait_downlink(dl, sizeof(dl), 1200))
        apply_downlink(dl, &g_codec);           /* new eps, commands, clock */
    g_day_seq++;
}

void setup(void)
{
    bool clamped;
    int16_t u0 = quantise(meter_read_kwh(), &clamped);
    codec_init(&g_codec, commissioning_eps(), u0, 0);
    g_seq = 0;
}

void loop(void)
{
    step_result_t r;
    bool clamped;
    int16_t u;

    /* blueprint 5.1, in order */
    if (codec_is_boundary(&g_codec, g_seq)) daily_boundary();    /* 4 */
    u = quantise(meter_read_kwh(), &clamped);                    /* 1 */
    codec_step(&g_codec, u, clamped, &r);                        /* 2, 3, 5 */
    if (r.tx) send_exception(&r);
    bench_hour(g_seq, &r);
    g_seq++;
    sleep_until_next_hour();                                     /* 6 */
}
