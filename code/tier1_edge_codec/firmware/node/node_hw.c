/* node_hw.c -- the hardware layer main.ino has always declared and never had.
 *
 * These are the seven symbols `main.ino` references. Until 2026-09-28 none of
 * them existed, so the sketch did not link -- and two of them were not even
 * declared, so it did not compile. They live here, and ONLY here: firmware/hil/
 * implements none of them, and `make -C ../hil guard` fails the build if it
 * ever does.
 *
 * NO CODEC ARITHMETIC IN THIS FILE. That is the same invariant main.ino keeps,
 * for the same reason: every number the node produces must come out of
 * edge_codec.c, which is the transliteration the golden vectors cover.
 *
 * THE METER. There is no meter attached and there is not meant to be -- this
 * project's input is dataset data. `meter_read_kwh` replays the recorded trace
 * in meter_trace.h, which is GOLDEN_U verbatim. Because SCALE_K is negative,
 * kWh = u * 2^SCALE_K is an exact binary fraction and main.ino's `quantise()`
 * returns the original integer, so the node's outputs must equal the golden
 * ones. That is what node_host.c checks. No figure derived from this stand-in
 * is a measurement of a meter, and nothing here should ever be quoted as one.
 */
#include <stdint.h>
#include <stdbool.h>
#include <avr/pgmspace.h>

#include "../edge_codec.h"
#include "node_config.h"
#include "node_protocol.h"
#include "sx127x.h"
#include "meter_trace.h"
#include "../hil/hil_uart.h"

/* ---- bench instrumentation ---------------------------------------------
 * Counters the host reads back. They are the bench's, not the application's. */
uint16_t g_hw_hour       = 0;    /* index into the replayed trace */
uint16_t g_hw_tx_frames  = 0;
uint32_t g_hw_tx_bytes   = 0;
uint32_t g_hw_air_ms     = 0;
uint8_t  g_hw_radio_ok   = 0;    /* RegVersion, 0 until sx_init runs */
uint8_t  g_hw_radio_live = 0;    /* 0 = count frames only, do not key the PA */

/* ---- the meter that is not a meter --------------------------------------
 *
 * THE COMMISSIONING READ DOES NOT ADVANCE THE TRACE, and that is not a fudge.
 * edge/codec.py:269 uses u0 only to PRIME the ring and day_true -- it is a
 * prior, not an observation -- and test_host.c accordingly seeds with
 * GOLDEN_U[0] and then steps on GOLDEN_U[0] as well. main.ino's setup() takes
 * a commissioning reading for that prior, which on a real meter an hour before
 * the first scheduled sample is a genuinely different reading.
 *
 * Replaying a recorded trace, the faithful mapping is that the commissioning
 * read and hour 0 are the same hour. Letting setup() consume trace[0] instead
 * shifts every output by one hour, which is exactly what it did the first time
 * this ran: the node's hour t reproduced the reference's hour t+1.
 *
 * So the first call returns trace[0] without advancing. main.ino is unchanged;
 * this is the bench's stand-in behaving like the fixture it is replaying. */
static bool g_commissioned = false;

float meter_read_kwh(void)
{
    int32_t u;
    if (g_hw_hour >= METER_TRACE_N) g_hw_hour = METER_TRACE_N - 1;
    /* int32 and PRE-CLAMP: one hour carries a 40 000 000-unit spike so that
     * main.ino's quantise() actually exercises its clamp. Shipping the clipped
     * value would leave that branch dead on the bench. */
    u = (int32_t)pgm_read_dword(&METER_TRACE[g_hw_hour]);
    if (g_commissioned) g_hw_hour++;
    else                g_commissioned = true;
    /* SCALE_K is negative, so this is an exact binary fraction and quantise()
     * inverts it without loss. */
#if SCALE_K >= 0
    return (float)u * (float)(1UL << SCALE_K);
#else
    return (float)u / (float)(1UL << (-SCALE_K));
#endif
    /* Exact in both directions: SCALE_K is a power of two and every value here
     * is an integer well inside float32's 24-bit mantissa, so quantise()
     * recovers u unchanged. The kWh figure itself is NOT physically meaningful
     * -- the trace's own scale is k=-2 while edge_config.h ships SCALE_K=-6 --
     * and nothing should ever quote it as an energy reading. Only the integer
     * round trip matters, and that is exact. */
}

/* THE COMMISSIONING THRESHOLD.
 *
 * edge_config.h ships EPS_INIT = 502, which gen_header.py computed for
 * synth_data client 0 at alpha = 1. The golden trace was recorded by
 * gen_golden.py at eps = 684. Those are different numbers for good reasons and
 * neither is wrong -- but a node commissioned at 502 makes different transmit
 * decisions from a trace recorded at 684, so its reconstruction diverges and
 * its outputs cannot be compared with the fixture at all. That is precisely
 * what happened the first time this ran: hours 0-10 agreed, and everything
 * after the first differing transmit decision did not.
 *
 * So the bench commissions at the trace's eps, which meter_trace.h carries
 * alongside the values it came with. A deployed build uses EPS_INIT.
 * Neither value is invented here; both come from a generated header. */
uint16_t commissioning_eps(void)
{
#ifdef NODE_DEPLOY
    return EPS_INIT;
#else
    return METER_TRACE_EPS;
#endif
}

/* ---- radio --------------------------------------------------------------- */
bool radio_send(const uint8_t *p, uint8_t n)
{
    uint16_t air = 0;
    bool ok = true;

    if (g_hw_radio_live) {
        ok = sx_send(p, n, &air);
        g_hw_air_ms += air;
    }
    g_hw_tx_frames++;
    g_hw_tx_bytes += n;
    return ok;
}

bool radio_wait_downlink(uint8_t *p, uint8_t n, uint16_t timeout_ms)
{
    int16_t got;
    if (!g_hw_radio_live) return false;      /* no gateway on the bench */
    got = sx_recv(p, n, timeout_ms);
    return got == (int16_t)DOWNLINK_BYTES;
}

/* ---- the hour tick -------------------------------------------------------
 * A DEPLOYED node sleeps here, in the watchdog's power-down mode, and the
 * paper's limitation section already records that the resonator drifts and a
 * DS3231 is wanted. On the bench there is nothing to wait for: the trace is
 * the clock, and sleeping 3600 s per sample would make a 384-hour run take
 * sixteen days. So this returns immediately and the fact is stated rather than
 * hidden. ANY ENERGY FIGURE TAKEN WITH THIS BUILD IS MEANINGLESS -- card
 * P3-01 step 3 needs a build that really sleeps, on a board that is not
 * powered through a USB bridge. */
void sleep_until_next_hour(void)
{
}

/* ---- CRC ----------------------------------------------------------------
 * The ATmega328P has no CRC peripheral, so the name's "_hw" is aspirational.
 * It forwards to the same software CRC the gateway uses. */
uint16_t crc16_hw(const uint8_t *p, uint8_t n)
{
    return np_crc16(p, n);
}

/* ---- downlink ------------------------------------------------------------ */
void apply_downlink(const uint8_t *dl, codec_t *st)
{
    downlink_t d;
    if (!decode_downlink(dl, DOWNLINK_BYTES, &d)) return;

    /* eps is the one field that changes codec behaviour, and it is carried
     * precisely so the threshold can be retuned without reflashing. Writing it
     * is not arithmetic: the value is the gateway's, used as given. */
    st->eps = d.eps;

    /* CMD_CLOCK_CORRECTION and CMD_SET_N_MAX_TX are accepted by the format and
     * not acted on here: the first needs an RTC the bench does not have, the
     * second needs N_MAX_TX to be mutable, and it is a compile-time constant
     * in the GENERATED edge_config.h. Both are recorded as gaps in
     * node/README.md rather than being half-implemented. */
}

/* ---- the bench hook ------------------------------------------------------
 * main.ino calls this once per hour. In a deployed build it is empty; on the
 * bench it streams what the codec produced so node_host.c can diff it against
 * golden_vectors.h. It reads the result, never computes one. */
void bench_hour(uint32_t seq, const step_result_t *r)
{
#ifdef NODE_DEPLOY
    (void)seq; (void)r;     /* a deployed node reports to the gateway, not a PC */
#else
    uart_puts("H ");
    uart_put_u32(seq);
    uart_putc(' '); uart_put_i32(r->u_hat);
    uart_putc(' '); uart_put_i32(r->e);
    uart_putc(' '); uart_put_u32((uint32_t)((r->tx ? 1u : 0u)
                                          | (r->event ? 2u : 0u)
                                          | (r->clamped ? 4u : 0u)));
    uart_putc('\n');
#endif
}
