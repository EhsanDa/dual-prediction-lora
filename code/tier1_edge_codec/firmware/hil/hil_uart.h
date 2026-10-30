/* hil_uart.h -- polled USART0 for the ATmega328P. Device side only. */
#ifndef HIL_UART_H
#define HIL_UART_H

#include <stdint.h>

void uart_init(void);
void uart_putc(char c);
void uart_puts(const char *s);
void uart_put_i32(int32_t v);
void uart_put_u32(uint32_t v);
void uart_put_hex8(uint8_t b);

/* Blocks until '\n'. Returns the length, NUL-terminated, '\r' stripped.
 * A line longer than max-1 is truncated; the remainder is discarded. */
uint8_t uart_getline(char *buf, uint8_t max);

#endif /* HIL_UART_H */
