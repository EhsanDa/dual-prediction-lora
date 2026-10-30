/* sx127x.h -- minimal SX1276/77/78/79 LoRa driver. Raw point-to-point.
 *
 * NOT LoRaWAN, and that is a design decision with a reason: an LMIC stack needs
 * roughly 1.5-2 kB of SRAM and ~30 kB of Flash and does not fit alongside the
 * application on a 2 kB / 32 kB part. There is one gateway per building, so
 * LoRaWAN's join, addressing and network-server machinery buy nothing here.
 * See docs/reference/design-rationale.md:182.
 */
#ifndef SX127X_H
#define SX127X_H

#include <stdint.h>
#include <stdbool.h>

#define SX_VERSION_SX1276 0x12
#define SX_VERSION_SX1272 0x22

/* Brings SPI up, checks RegVersion, and configures the modem from
 * node_config.h. Returns the RegVersion byte read -- 0x12 or 0x22 on success,
 * 0x00/0xFF when nothing is answering. */
uint8_t sx_init(void);

/* Override the carrier without re-initialising, for the band scan. */
void    sx_set_frequency(uint32_t hz);

/* Blocking send. Returns false on timeout waiting for TxDone.
 * `*air_ms` receives the measured time from TX-start to TxDone, which is what
 * `make toa` compares against edge/protocol.py's Semtech model. */
bool    sx_send(const uint8_t *buf, uint8_t len, uint16_t *air_ms);

/* Blocking receive. Returns the payload length, or 0 on timeout, or a negative
 * value if a packet arrived with a bad CRC -- which is a DIFFERENT outcome from
 * silence and must not be collapsed into it. */
int16_t sx_recv(uint8_t *buf, uint8_t max, uint16_t timeout_ms);

int16_t sx_last_rssi(void);     /* dBm */
int8_t  sx_last_snr(void);      /* dB */

void    sx_sleep(void);
void    sx_standby(void);
uint8_t sx_read_reg(uint8_t reg);
void    sx_write_reg(uint8_t reg, uint8_t val);

#endif /* SX127X_H */
