/* node_boot.cpp -- the entry point main.ino does not carry.
 *
 * An Arduino sketch has setup() and loop() and gets its main() from the core.
 * This project does not use the Arduino core -- there is no IDE here and the
 * application needs none of it -- so this file supplies the same three lines
 * the core's main.cpp does, plus the bench's framing.
 *
 * Keeping it separate means main.ino stays a sketch: it would still build
 * unmodified under the Arduino IDE, which is the form the paper describes.
 */
#include <stdint.h>

extern "C" {
#include "../edge_codec.h"
#include "node_config.h"
#include "sx127x.h"
#include "meter_trace.h"
#include "../hil/hil_uart.h"

extern uint8_t  g_hw_radio_ok;
extern uint8_t  g_hw_radio_live;
extern uint16_t g_hw_hour;
extern uint16_t g_hw_tx_frames;
extern uint32_t g_hw_tx_bytes;
extern uint32_t g_hw_air_ms;
}

void setup(void);
void loop(void);

/* PEAK STACK, MEASURED. ../README.md's row says "Total SRAM incl. radio +
 * stack", and static .data+.bss is only half of that. The free region between
 * .bss and the stack pointer is painted at entry and what survives is counted
 * at the end, so the row can be filled with a total rather than a part. Same
 * technique as hil_device.c. */
extern "C" { extern uint8_t _end; }
#define STACK_PAINT 0xC5

static void stack_paint(void)
{
    uint8_t  here;
    uint8_t *p  = &_end;
    uint8_t *sp = &here;
    while (p < sp - 16) *p++ = STACK_PAINT;
}

static uint16_t stack_unused(void)
{
    uint8_t *p = &_end;
    uint16_t n = 0;
    while (*p == STACK_PAINT) { n++; p++; }
    return n;
}

int main(void)
{
    uint16_t hour;

    stack_paint();

#ifndef NODE_DEPLOY
    uart_init();
    uart_puts("# READY node proto 2 board ");
    uart_put_u32(BOARD_ID);
    uart_putc('\n');
#endif

    /* The radio is brought up and IDENTIFIED, but the PA is not keyed during a
     * trace replay: 163 exception frames at SF12 is about three minutes of
     * airtime to prove something the link test proves in seconds. Frames are
     * still fully encoded and counted, so the byte totals are real.
     * `make link-test` and `make toa` are where the radio actually transmits. */
    g_hw_radio_ok   = sx_init();
#ifdef NODE_DEPLOY
    g_hw_radio_live = 1;            /* a deployed node really transmits */
#else
    g_hw_radio_live = 0;
    uart_puts("# radio RegVersion 0x");
    uart_put_hex8(g_hw_radio_ok);
    uart_puts(g_hw_radio_ok == 0x12 ? " SX1276/77/78/79\n"
            : g_hw_radio_ok == 0x22 ? " SX1272/73\n"
                                    : " NO RADIO ANSWERING\n");
    sx_sleep();

    uart_puts("# trace ");
    uart_put_u32(METER_TRACE_N);
    uart_puts(" hours, eps ");
    uart_put_u32(METER_TRACE_EPS);
    uart_putc('\n');
#endif

    setup();

    /* setup() consumed hour 0 to seed the codec, exactly as a commissioned
     * node does, so the replay covers the remaining hours. */
#ifdef NODE_DEPLOY
    (void)hour;
    for (;;) loop();            /* a deployed node never stops */
#else
    for (hour = g_hw_hour; hour < METER_TRACE_N; hour++)
        loop();

    uart_puts("# FRAMES ");
    uart_put_u32(g_hw_tx_frames);
    uart_puts(" BYTES ");
    uart_put_u32(g_hw_tx_bytes);
    uart_puts(" AIRMS ");
    uart_put_u32(g_hw_air_ms);
    uart_putc('\n');
    uart_puts("# STACKFREE ");
    uart_put_u32(stack_unused());
    uart_putc('\n');
    uart_puts("# NODE DONE\n");
#endif

    for (;;) { }
}
