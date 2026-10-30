/* radio_device.c -- the radio bench. ATmega328P.
 *
 * A command responder over the UART that drives sx127x.c directly, so the host
 * can coordinate TWO boards: one transmitting, one listening. That is what the
 * second Uno is for. A single board cannot test a radio link at all -- it
 * cannot hear itself -- and card P3-01 step 5 (in-situ frame loss) is
 * impossible without a pair.
 *
 * Commands, one line each, every reply echoing its letter (same convention as
 * firmware/hil):
 *
 *   V              -> V <proto> <board> <regversion>
 *   F <hz>         -> F OK <hz>              set the carrier
 *   T <n>          -> T <sent> <air_ms>      send n 8-byte test frames
 *   X              -> X <len> <air_ms>       send a real encoded EXCEPTION
 *   Y              -> Y <len> <air_ms>       send a real encoded RESYNC
 *   R <ms>         -> R <len> <rssi> <snr> <hex>   receive, or R 0 on timeout,
 *                                                  or R -1 on a CRC failure
 *   S              -> S <reg42> <opmode> <freq_hz>
 *
 * THE PA IS AT ITS MINIMUM SETTING AND MUST STAY THERE. See node_config.h.
 */
#include <stdint.h>
#include <stdbool.h>
#include <util/delay.h>

#include "node_config.h"
#include "node_protocol.h"
#include "sx127x.h"
#include "../hil/hil_uart.h"
#include "../edge_config.h"

#define HIL_PROTO_LOCAL 2

static char     g_line[128];
static uint8_t  g_buf[RESYNC_MAX_BYTES];
static uint8_t  g_ver;
static uint32_t g_freq = SX_FREQ_HZ;

static const char *skip_ws(const char *p)
{
    while (*p == ' ' || *p == '\t') p++;
    return p;
}

static const char *parse_u32(const char *p, uint32_t *out)
{
    uint32_t v = 0;
    uint8_t digits = 0;
    p = skip_ws(p);
    while (*p >= '0' && *p <= '9') { v = v * 10u + (uint32_t)(*p - '0'); p++; digits++; }
    if (!digits) return 0;
    *out = v;
    return p;
}

/* A deterministic 8-byte frame, so a received payload can be checked without
 * any shared state beyond this file. */
static void fill_test_frame(uint8_t *p, uint8_t seq)
{
    uint16_t crc;
    p[0] = 0xA5; p[1] = 0x5A; p[2] = BOARD_ID; p[3] = seq;
    p[4] = 0x00; p[5] = 0xFF;
    crc = np_crc16(p, 6);
    p[6] = (uint8_t)(crc & 0xFF);
    p[7] = (uint8_t)(crc >> 8);
}

static void do_tx(uint8_t n)
{
    uint16_t air = 0, total = 0;
    uint8_t  i, sent = 0;
    for (i = 0; i < n; i++) {
        fill_test_frame(g_buf, i);
        if (sx_send(g_buf, 8, &air)) { sent++; total = (uint16_t)(total + air); }
        _delay_ms(50);
    }
    uart_puts("T "); uart_put_u32(sent);
    uart_putc(' ');  uart_put_u32(total);
    uart_putc('\n');
}

/* A REAL exception frame, built by the same encoder the application uses. */
static void do_tx_exception(void)
{
    uint16_t air = 0;
    uint8_t n = encode_exception(g_buf, sizeof(g_buf), NODE_ID, 7,
                                 FLAG_EVENT, -1234);
    bool ok = sx_send(g_buf, n, &air);
    uart_puts("X "); uart_put_u32(ok ? n : 0);
    uart_putc(' ');  uart_put_u32(air);
    uart_putc('\n');
}

/* A REAL resync frame -- 24 values, the tx bitmap and the weight vector, the
 * frame whose size Table II quotes at 83 B. */
static void do_tx_resync(void)
{
    int16_t vals[24];
    int32_t w[N_W];
    uint16_t air = 0;
    uint8_t i, n;
    for (i = 0; i < 24; i++) vals[i] = (int16_t)(2500 + i * 37);
    for (i = 0; i < N_W; i++) w[i] = (int32_t)(1000L * (i + 1));
    n = encode_resync(g_buf, sizeof(g_buf), NODE_ID, 3, 11, vals,
                      0x00A5A5UL, w);
    if (!n) { uart_puts("Y 0 0\n"); return; }
    {
        bool ok = sx_send(g_buf, n, &air);
        uart_puts("Y "); uart_put_u32(ok ? n : 0);
        uart_putc(' ');  uart_put_u32(air);
        uart_putc('\n');
    }
}

static void do_rx(uint16_t ms)
{
    int16_t got = sx_recv(g_buf, sizeof(g_buf), ms);
    uart_puts("R ");
    uart_put_i32(got);
    if (got > 0) {
        int16_t i;
        uart_putc(' '); uart_put_i32(sx_last_rssi());
        uart_putc(' '); uart_put_i32(sx_last_snr());
        uart_putc(' ');
        for (i = 0; i < got; i++) uart_put_hex8(g_buf[i]);
    }
    uart_putc('\n');
}

int main(void)
{
    uart_init();
    g_ver = sx_init();

    uart_puts("# READY radio proto ");
    uart_put_u32(HIL_PROTO_LOCAL);
    uart_puts(" board ");
    uart_put_u32(BOARD_ID);
    uart_putc('\n');

    for (;;) {
        const char *p;
        uint32_t v;

        if (uart_getline(g_line, sizeof(g_line)) == 0) continue;
        p = g_line + 1;

        switch (g_line[0]) {
        case '#': break;
        case 'V':
            uart_puts("V "); uart_put_u32(HIL_PROTO_LOCAL);
            uart_putc(' ');  uart_put_u32(BOARD_ID);
            uart_putc(' ');  uart_put_u32(g_ver);
            uart_putc('\n');
            break;
        case 'F':
            if (!parse_u32(p, &v)) { uart_puts("E F needs hz\n"); break; }
            g_freq = v;
            sx_standby();
            sx_set_frequency(g_freq);
            uart_puts("F OK "); uart_put_u32(g_freq); uart_putc('\n');
            break;
        case 'T':
            if (!parse_u32(p, &v)) v = 1;
            do_tx((uint8_t)(v > 20 ? 20 : v));
            break;
        case 'X': do_tx_exception(); break;
        case 'Y': do_tx_resync();    break;
        case 'R':
            if (!parse_u32(p, &v)) v = 5000;
            do_rx((uint16_t)(v > 60000 ? 60000 : v));
            break;
        case 'S':
            uart_puts("S "); uart_put_u32(sx_read_reg(0x42));
            uart_putc(' ');  uart_put_u32(sx_read_reg(0x01));
            uart_putc(' ');  uart_put_u32(g_freq);
            uart_putc('\n');
            break;
        default: uart_puts("E unknown command\n"); break;
        }
    }
}
