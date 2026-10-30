/* hil_device.c -- the hardware-in-the-loop responder. ATmega328P.
 *
 * Runs the REAL edge_codec.c on the REAL silicon and answers questions about
 * it over the UART. The host (hil_host.c) holds the golden vectors and drives
 * the whole trace, hour by hour.
 *
 * WHY THE HOST HOLDS THE VECTORS. golden_vectors.h is ~11.5 kB and its arrays
 * are plain `static const`, which avr-gcc places in .data -- i.e. copied into
 * the 2 kB of SRAM at startup. It does not fit, and making it fit would mean
 * adding PROGMEM to a GENERATED file. So the trace is streamed instead. This
 * device holds one codec_t (486 B) and a line buffer, and nothing else.
 *
 * THE INVARIANT THIS FILE KEEPS, and it is the same one main.ino keeps:
 * NO ARITHMETIC LIVES HERE. Every number this file prints came out of
 * edge_codec.c. It parses, it dispatches, it formats. The moment a codec
 * decision migrates into this file, the equivalence test stops covering it and
 * starts covering this file's opinion of it instead.
 */
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "../edge_codec.h"
#include "hil_uart.h"
#include "hil_proto.h"

/* ---- stack headroom -----------------------------------------------------
 * This image uses two thirds of the part's SRAM, so "there is enough stack"
 * should be a measurement and not a hope. The free region between .bss and the
 * stack pointer is painted at entry; the M command then counts how much paint
 * survives, which is the high-water mark of everything the codec and this
 * responder have done since. PLAYBOOK rule 2: report the measurement.
 */
extern uint8_t _end;
#define STACK_PAINT 0xC5

static void stack_paint(void)
{
    uint8_t  here;
    uint8_t *p  = &_end;
    uint8_t *sp = &here;
    while (p < sp - 16) *p++ = STACK_PAINT;   /* 16 B of slack for our own frame */
}

static uint16_t stack_unused(void)
{
    uint8_t *p = &_end;
    uint16_t n = 0;
    while (*p == STACK_PAINT) { n++; p++; }
    return n;
}

static codec_t  g_codec;
static char     g_line[HIL_LINE_MAX];
static int16_t  g_true24[24];
static uint8_t  g_snap[SNAPSHOT_BYTES];

/* ---- parsing ----------------------------------------------------------
 * Hand-rolled rather than strtol(): it keeps the parser's behaviour on
 * malformed input explicit and visible, and avoids dragging in locale-aware
 * stdlib code for the sake of reading integers.
 */
static const char *skip_ws(const char *p)
{
    while (*p == ' ' || *p == '\t') p++;
    return p;
}

/* Parses one decimal integer. Returns NULL if there is not one there. */
static const char *parse_i32(const char *p, int32_t *out)
{
    int32_t  v = 0;
    uint8_t  neg = 0, digits = 0;

    p = skip_ws(p);
    if (*p == '-') { neg = 1; p++; }
    else if (*p == '+') { p++; }
    while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; digits++; }
    if (!digits) return NULL;
    *out = neg ? -v : v;
    return p;
}

static void err(const char *why)
{
    uart_puts("E ");
    uart_puts(why);
    uart_putc('\n');
}

/* ---- commands ---------------------------------------------------------- */

/* The handshake. The host compares every field against its own headers and
 * refuses to run on a mismatch -- because an image built from a different
 * edge_config.h would otherwise fail as an unexplained byte difference at some
 * arbitrary hour, which is a genuinely confusing way to spend an afternoon. */
static void cmd_version(void)
{
    uart_puts("V ");
    uart_put_u32(HIL_PROTO);
    uart_putc(' '); uart_put_u32((uint32_t)BOARD_ID);
    uart_putc(' '); uart_put_u32((uint32_t)sizeof(codec_t));
    uart_putc(' '); uart_put_u32((uint32_t)SNAPSHOT_BYTES);
    uart_putc(' '); uart_put_u32((uint32_t)N_W);
    uart_putc(' '); uart_put_u32((uint32_t)BUF_LEN);
    uart_putc(' '); uart_put_u32((uint32_t)SYNC_SAFE);
    uart_putc(' '); uart_puts(FEATURE_SPEC_HASH);
    uart_putc('\n');
}

static void cmd_init(const char *p)
{
    int32_t eps, u0, t0;
    if (!(p = parse_i32(p, &eps)) ||
        !(p = parse_i32(p, &u0))  ||
        !(p = parse_i32(p, &t0))) { err("I needs eps u0 t0"); return; }
    codec_init(&g_codec, (uint16_t)eps, (int16_t)u0, (uint32_t)t0);
    uart_puts("I OK\n");
}

static void cmd_predict(void)
{
    uart_puts("P ");
    uart_put_i32(codec_predict(&g_codec));
    uart_putc('\n');
}

static void cmd_is_boundary(const char *p)
{
    int32_t t;
    if (!parse_i32(p, &t)) { err("T needs t"); return; }
    uart_puts("T ");
    uart_putc(codec_is_boundary(&g_codec, (uint32_t)t) ? '1' : '0');
    uart_putc('\n');
}

/* "B -" passes true_24 = NULL, which is legal in sync_safe mode and is a path
 * the host must be able to exercise on the silicon, not just on x86-64. */
static void cmd_day_boundary(const char *p)
{
    uint8_t i;
    int32_t v;

    p = skip_ws(p);
    if (*p == '-') { codec_day_boundary(&g_codec, NULL); uart_puts("B OK\n"); return; }

    for (i = 0; i < 24; i++) {
        if (!(p = parse_i32(p, &v))) { err("B needs 24 values or -"); return; }
        g_true24[i] = (int16_t)v;
    }
    codec_day_boundary(&g_codec, g_true24);
    uart_puts("B OK\n");
}

static void cmd_step(const char *p)
{
    int32_t u, clamped;
    step_result_t r;

    if (!(p = parse_i32(p, &u)) ||
        !(p = parse_i32(p, &clamped))) { err("S needs u clamped"); return; }

    codec_step(&g_codec, (int16_t)u, clamped != 0, &r);

    uart_puts("S ");
    uart_put_i32(r.e);
    uart_putc(' '); uart_put_i32(r.u_hat);
    uart_putc(' '); uart_put_i32(r.u_recon);
    uart_putc(' '); uart_put_u32(r.eps_eff);
    uart_putc(' '); uart_putc(r.tx      ? '1' : '0');
    uart_putc(' '); uart_putc(r.event   ? '1' : '0');
    uart_putc(' '); uart_putc(r.clamped ? '1' : '0');
    uart_putc('\n');
}

static void cmd_resync(void)
{
    uint8_t i;
    codec_resync_payload(&g_codec, g_true24);
    uart_puts("R ");
    uart_put_u32(codec_resync_bitmap(&g_codec));
    for (i = 0; i < 24; i++) { uart_putc(' '); uart_put_i32(g_true24[i]); }
    uart_putc('\n');
}

static void cmd_snapshot(void)
{
    uint16_t i;
    codec_snapshot(&g_codec, g_snap);
    uart_puts("Q ");
    for (i = 0; i < SNAPSHOT_BYTES; i++) uart_put_hex8(g_snap[i]);
    uart_putc('\n');
}

int main(void)
{
    stack_paint();
    uart_init();

    /* The banner exists so the host can tell "the board just reset" from "the
     * board is wedged". Optiboot holds the port for a moment after DTR drops;
     * the host discards '#' lines and polls V until one answers. */
    uart_puts("# READY hil proto ");
    uart_put_u32(HIL_PROTO);
    uart_puts(" board ");
    uart_put_u32(BOARD_ID);
    uart_putc('\n');

    for (;;) {
        const char *p;
        char cmd;

        if (uart_getline(g_line, sizeof(g_line)) == 0) continue;
        cmd = g_line[0];
        p   = g_line + 1;

        switch (cmd) {
        case '#': break;                    /* comment: ignore silently */
        case 'V': cmd_version();        break;
        case 'I': cmd_init(p);          break;
        case 'P': cmd_predict();        break;
        case 'T': cmd_is_boundary(p);   break;
        case 'B': cmd_day_boundary(p);  break;
        case 'S': cmd_step(p);          break;
        case 'N': codec_new_day(&g_codec); uart_puts("N OK\n"); break;
        case 'R': cmd_resync();         break;
        case 'Q': cmd_snapshot();       break;
        case 'M': uart_puts("M "); uart_put_u32(stack_unused());
                  uart_putc('\n'); break;
        default:  err("unknown command"); break;
        }
    }
}
