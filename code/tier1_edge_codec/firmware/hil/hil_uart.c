/* hil_uart.c -- polled USART0. No interrupts, no ring buffer.
 *
 * Polling is deliberate. The device does exactly one thing at a time -- read a
 * command, run it, answer it -- so an ISR would buy nothing and would add a
 * source of nondeterminism to a harness whose entire job is to prove a
 * deterministic result. Flow control is the protocol itself: the host does not
 * send the next command until it has read the reply to the last one, so the
 * 2-byte hardware FIFO never overruns.
 *
 * Baud: U2X0 = 1, so UBRR = F_CPU/(8*baud) - 1. At F_CPU = 16 MHz and a
 * nominal 115200 that is 16.361, truncating to 16, i.e. an actual 117647 baud
 * and an error of +2.12 %. That is inside the ~2.5 % an 8N1 frame tolerates
 * and it is the divisor the stock Arduino core ships. The static assertion
 * below refuses to build if a different F_CPU/baud pair pushes the error past
 * 2.5 %, so this stays a checked fact rather than a remembered one.
 */
#include <avr/io.h>
#include "hil_uart.h"
#include "hil_proto.h"

#define UBRR_VAL   ((F_CPU + 4UL * HIL_BAUD) / (8UL * HIL_BAUD) - 1UL)
#define BAUD_ACTUAL (F_CPU / (8UL * (UBRR_VAL + 1UL)))
#define BAUD_ERR_PPT (((BAUD_ACTUAL > HIL_BAUD) \
                       ? (BAUD_ACTUAL - HIL_BAUD) : (HIL_BAUD - BAUD_ACTUAL)) \
                      * 1000UL / HIL_BAUD)

/* C99 has no static_assert; this is the portable negative-array idiom. */
typedef char hil_baud_error_within_25_permille[(BAUD_ERR_PPT <= 25) ? 1 : -1];

void uart_init(void)
{
    UBRR0H = (uint8_t)(UBRR_VAL >> 8);
    UBRR0L = (uint8_t)(UBRR_VAL & 0xFF);
    UCSR0A = (1 << U2X0);
    UCSR0B = (1 << RXEN0) | (1 << TXEN0);
    UCSR0C = (1 << UCSZ01) | (1 << UCSZ00);   /* 8N1 */
}

void uart_putc(char c)
{
    while (!(UCSR0A & (1 << UDRE0))) { }
    UDR0 = (uint8_t)c;
}

void uart_puts(const char *s)
{
    while (*s) uart_putc(*s++);
}

void uart_put_u32(uint32_t v)
{
    char d[10];
    uint8_t n = 0;
    if (v == 0) { uart_putc('0'); return; }
    while (v) { d[n++] = (char)('0' + (v % 10u)); v /= 10u; }
    while (n) uart_putc(d[--n]);
}

void uart_put_i32(int32_t v)
{
    /* Negate into uint32_t so INT32_MIN does not overflow. */
    if (v < 0) { uart_putc('-'); uart_put_u32((uint32_t)0 - (uint32_t)v); }
    else       { uart_put_u32((uint32_t)v); }
}

void uart_put_hex8(uint8_t b)
{
    static const char H[] = "0123456789abcdef";
    uart_putc(H[b >> 4]);
    uart_putc(H[b & 0x0F]);
}

static char uart_getc(void)
{
    while (!(UCSR0A & (1 << RXC0))) { }
    return (char)UDR0;
}

uint8_t uart_getline(char *buf, uint8_t max)
{
    uint8_t n = 0;
    for (;;) {
        char c = uart_getc();
        if (c == '\n') break;
        if (c == '\r') continue;
        if (n + 1u < max) buf[n++] = c;
        /* else: silently drop, but keep consuming to the newline so the next
         * command still starts on a frame boundary. */
    }
    buf[n] = '\0';
    return n;
}
