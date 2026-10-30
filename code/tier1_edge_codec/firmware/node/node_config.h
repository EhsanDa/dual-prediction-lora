/* node_config.h -- everything about THIS BENCH, in one place.
 *
 * Wiring, board identity and radio settings. Nothing here is a codec
 * parameter: those live in the GENERATED edge_config.h and are not touched.
 */
#ifndef NODE_CONFIG_H
#define NODE_CONFIG_H

/* ---- which physical board -------------------------------------------------
 * The two Unos on the bench report the IDENTICAL USB serial, so a port name is
 * not an identity. Baked in at flash time (`make node-flash BOARD=2`) and
 * reported in the banner; the host asks rather than assumes. Deliberately NOT
 * edge_config.h's NODE_ID, which is a protocol field. */
#ifndef BOARD_ID
#define BOARD_ID 1
#endif

/* ---- radio wiring ---------------------------------------------------------
 * Measured by firmware/hil/probe_device.c on 2026-09-28, on BOTH boards:
 * SX127x RegVersion = 0x12 with NSS on D10. Hardware SPI, so SCK/MOSI/MISO are
 * fixed at D13/D11/D12 by the part.
 *
 * DIO0 IS DELIBERATELY NOT USED. The probe found a pin held low by the module
 * on each board -- D3 on board 1, D2 on board 2 -- which is consistent with
 * DIO0 but is not proof, and the two differ. Rather than depend on a guess
 * that is not even consistent across the bench, the driver POLLS RegIrqFlags
 * over SPI. It costs one SPI transaction per poll and removes the wiring
 * question entirely. If a DIO0 line is ever confirmed, it is an optimisation,
 * not a correction. */
#define SX_NSS_PIN     10
#define SX_RESET_PIN   -1      /* not wired to a known pin; POR reset is used */

/* ---- radio settings -------------------------------------------------------
 * SF12/BW125/CR4-5 are what the paper's airtime model assumes
 * (edge/protocol.py::time_on_air, Table II: 991 ms and 3449 ms). The bench
 * must match the model or a measured airtime means nothing.
 *
 * POWER IS MINIMUM, ALWAYS. +2 dBm (about 1.6 mW), the lowest the PA_BOOST
 * path offers. The boards sit centimetres apart.
 *
 * PA_BOOST rather than RFO is not a power choice, it is a wiring fact: on
 * RA-02, RFM95W and most SX127x breakouts the antenna is bonded to PA_BOOST
 * and RFO is not connected. Transmitting through RFO completes normally and
 * radiates nothing, which is a genuinely confusing failure -- see sx127x.c.
 *
 * Raising this is a deliberate act and a regulatory decision, not a default.
 * Valid range on this path is 2..17 dBm. */
#define SX_TX_POWER_DBM 2
#define SX_FREQ_HZ     433000000UL   /* overridden by -DSX_FREQ_HZ during band-scan */
#define SX_SF          12
#define SX_BW_KHZ      125
#define SX_CODING_RATE 5             /* 4/5 */
#define SX_PREAMBLE    8
#define SX_SYNC_WORD   0x12          /* private network; 0x34 is LoRaWAN */
#define SX_EXPLICIT_HEADER 1
#define SX_CRC_ON      1

#endif /* NODE_CONFIG_H */
