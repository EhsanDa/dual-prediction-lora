/* sx127x.c -- SX1276/77/78/79 LoRa driver, polled. ATmega328P.
 *
 * Register names and semantics from the Semtech SX1276/77/78/79 datasheet
 * rev 7, chapter 4 (LoRa mode) and the register map in chapter 6. The same
 * document edge/protocol.py:238 cites for the airtime formula, so the settings
 * here and the model there describe the same radio.
 *
 * POLLED, NO INTERRUPTS. RegIrqFlags is read over SPI rather than waiting on
 * DIO0, because the DIO0 wiring is not confirmed and is not even the same pin
 * on the two bench boards. See node_config.h.
 */
#include <avr/io.h>
#include <util/delay.h>

#include "node_config.h"
#include "sx127x.h"

/* ---- registers (LoRa mode) --------------------------------------------- */
#define REG_FIFO                0x00
#define REG_OP_MODE             0x01
#define REG_FRF_MSB             0x06
#define REG_FRF_MID             0x07
#define REG_FRF_LSB             0x08
#define REG_PA_CONFIG           0x09
#define REG_OCP                 0x0B
#define REG_LNA                 0x0C
#define REG_FIFO_ADDR_PTR       0x0D
#define REG_FIFO_TX_BASE_ADDR   0x0E
#define REG_FIFO_RX_BASE_ADDR   0x0F
#define REG_FIFO_RX_CURRENT     0x10
#define REG_IRQ_FLAGS           0x12
#define REG_RX_NB_BYTES         0x13
#define REG_PKT_SNR_VALUE       0x19
#define REG_PKT_RSSI_VALUE      0x1A
#define REG_MODEM_CONFIG_1      0x1D
#define REG_MODEM_CONFIG_2      0x1E
#define REG_SYMB_TIMEOUT_LSB    0x1F
#define REG_PREAMBLE_MSB        0x20
#define REG_PREAMBLE_LSB        0x21
#define REG_PAYLOAD_LENGTH      0x22
#define REG_MODEM_CONFIG_3      0x26
#define REG_DETECTION_OPTIMIZE  0x31
#define REG_DETECTION_THRESHOLD 0x37
#define REG_SYNC_WORD           0x39
#define REG_VERSION             0x42
#define REG_PA_DAC              0x4D

#define MODE_LONG_RANGE         0x80
#define MODE_SLEEP              0x00
#define MODE_STDBY              0x01
#define MODE_TX                 0x03
#define MODE_RX_CONTINUOUS      0x05

#define IRQ_RX_TIMEOUT          0x80
#define IRQ_RX_DONE             0x40
#define IRQ_CRC_ERROR           0x20
#define IRQ_TX_DONE             0x08

static int16_t g_rssi;
static int8_t  g_snr;
static uint8_t g_version;      /* what RegVersion read at init: 0x12 or 0x22 */

/* ---- pins -------------------------------------------------------------- */
static void nss_high(void)
{
#if SX_NSS_PIN < 8
    PORTD |= (uint8_t)(1 << SX_NSS_PIN);
#else
    PORTB |= (uint8_t)(1 << (SX_NSS_PIN - 8));
#endif
}
static void nss_low(void)
{
#if SX_NSS_PIN < 8
    PORTD &= (uint8_t)~(1 << SX_NSS_PIN);
#else
    PORTB &= (uint8_t)~(1 << (SX_NSS_PIN - 8));
#endif
}
static void nss_init(void)
{
#if SX_NSS_PIN < 8
    DDRD |= (uint8_t)(1 << SX_NSS_PIN);
#else
    DDRB |= (uint8_t)(1 << (SX_NSS_PIN - 8));
#endif
    nss_high();
}

/* ---- SPI --------------------------------------------------------------- */
static void spi_init(void)
{
    /* SCK PB5, MOSI PB3 out; MISO PB4 in. PB2 (hardware SS) MUST be an output
     * even when unused, or the SPI block leaves master mode the instant
     * something pulls it low. */
    DDRB |= (1 << PB5) | (1 << PB3) | (1 << PB2);
    DDRB &= (uint8_t)~(1 << PB4);
    SPCR = (1 << SPE) | (1 << MSTR) | (1 << SPR0);   /* f/16 = 1 MHz, mode 0 */
    SPSR = 0;
}

static uint8_t spi_xfer(uint8_t b)
{
    SPDR = b;
    while (!(SPSR & (1 << SPIF))) { }
    return SPDR;
}

uint8_t sx_read_reg(uint8_t reg)
{
    uint8_t v;
    nss_low();
    spi_xfer(reg & 0x7F);
    v = spi_xfer(0x00);
    nss_high();
    return v;
}

void sx_write_reg(uint8_t reg, uint8_t val)
{
    nss_low();
    spi_xfer(reg | 0x80);
    spi_xfer(val);
    nss_high();
}

/* ---- modes ------------------------------------------------------------- */
static void set_mode(uint8_t mode)
{
    sx_write_reg(REG_OP_MODE, (uint8_t)(MODE_LONG_RANGE | mode));
}

void sx_sleep(void)   { set_mode(MODE_SLEEP); }
void sx_standby(void) { set_mode(MODE_STDBY); }

void sx_set_frequency(uint32_t hz)
{
    /* Frf = f * 2^19 / 32e6. Done as a 64-bit product to avoid the rounding a
     * 32-bit intermediate would introduce at 915 MHz. */
    uint64_t frf = ((uint64_t)hz << 19) / 32000000UL;
    sx_write_reg(REG_FRF_MSB, (uint8_t)(frf >> 16));
    sx_write_reg(REG_FRF_MID, (uint8_t)(frf >> 8));
    sx_write_reg(REG_FRF_LSB, (uint8_t)(frf));
}

static uint8_t bw_code(void)
{
    /* Only the bandwidths this project uses. 125 kHz is what the model assumes. */
#if   SX_BW_KHZ == 125
    return 7;
#elif SX_BW_KHZ == 250
    return 8;
#elif SX_BW_KHZ == 500
    return 9;
#else
#error "unsupported SX_BW_KHZ"
#endif
}

uint8_t sx_init(void)
{
    uint8_t ver;

    nss_init();
    spi_init();
    _delay_ms(10);

    ver = sx_read_reg(REG_VERSION);
    if (ver != SX_VERSION_SX1276 && ver != SX_VERSION_SX1272) return ver;
    g_version = ver;

    /* LoRa mode can only be entered from sleep. */
    set_mode(MODE_SLEEP);
    _delay_ms(10);
    set_mode(MODE_STDBY);

    sx_set_frequency(SX_FREQ_HZ);

    sx_write_reg(REG_FIFO_TX_BASE_ADDR, 0x00);
    sx_write_reg(REG_FIFO_RX_BASE_ADDR, 0x00);

    /* LNA: max gain, boost on. */
    sx_write_reg(REG_LNA, (uint8_t)(sx_read_reg(REG_LNA) | 0x03));

    /* MINIMUM POWER ON THE PA_BOOST PIN.
     *
     * PaSelect MUST be 1 here. The first version of this driver used the RFO
     * path (PA_CONFIG = 0x00), which is the datasheet's low-power output and
     * looked like the conservative choice -- and nothing was ever received, on
     * any band. On RA-02, RFM95W and the other common SX127x breakouts the
     * antenna is bonded to PA_BOOST and the RFO pin is not connected at all,
     * so transmitting through RFO radiates essentially nothing. The transmit
     * still completes and TxDone still fires, which is why it looks like a
     * receiver fault rather than a transmitter one.
     *
     * With PaSelect = 1 and OutputPower = 0, Pout = 17 - (15 - 0) = +2 dBm,
     * about 1.6 mW -- the lowest the boost path offers, and the intended
     * minimum. PA_DAC stays 0x84; 0x87 is the +20 dBm mode and must never be
     * set here. Raising power is a regulatory decision, not a default.
     * See node_config.h. */
    sx_write_reg(REG_PA_CONFIG, (uint8_t)(0x80 | (SX_TX_POWER_DBM - 2)));
    sx_write_reg(REG_PA_DAC, 0x84);
    sx_write_reg(REG_OCP, 0x0B);            /* 45 mA trim, well above RFO need */

    /* ModemConfig1: BW | CR | explicit header */
    sx_write_reg(REG_MODEM_CONFIG_1,
                 (uint8_t)((bw_code() << 4)
                           | (((SX_CODING_RATE - 4) & 0x07) << 1)
                           | (SX_EXPLICIT_HEADER ? 0x00 : 0x01)));
    /* ModemConfig2: SF | CRC on */
    sx_write_reg(REG_MODEM_CONFIG_2,
                 (uint8_t)(((SX_SF & 0x0F) << 4) | (SX_CRC_ON ? 0x04 : 0x00)));

    /* LOW DATA RATE OPTIMISATION IS MANDATORY when the symbol time exceeds
     * 16 ms, which at BW125 means SF11 and SF12. edge/protocol.py's airtime
     * model applies it on exactly the same condition; if the two disagreed,
     * measured airtime could not be compared with the model at all.
     * AgcAutoOn is set in the same register. */
    sx_write_reg(REG_MODEM_CONFIG_3,
                 (uint8_t)(((SX_SF >= 11 && SX_BW_KHZ == 125) ? 0x08 : 0x00) | 0x04));

    sx_write_reg(REG_PREAMBLE_MSB, (uint8_t)(SX_PREAMBLE >> 8));
    sx_write_reg(REG_PREAMBLE_LSB, (uint8_t)(SX_PREAMBLE & 0xFF));
    sx_write_reg(REG_SYNC_WORD, SX_SYNC_WORD);

    /* SF-dependent detection settings, datasheet 4.1.1.7. */
    sx_write_reg(REG_DETECTION_OPTIMIZE,  (SX_SF == 6) ? 0xC5 : 0xC3);
    sx_write_reg(REG_DETECTION_THRESHOLD, (SX_SF == 6) ? 0x0C : 0x0A);

    set_mode(MODE_STDBY);
    return ver;
}

/* ---- a guarded IRQ read -------------------------------------------------
 *
 * SPI READS ARE NOT TRUSTWORTHY WHILE THE PA IS KEYED ON THIS HARDWARE, and
 * that is a measured fact, not a precaution. Polling RegVersion during a
 * transmission on these boards returns 0xAA instead of 0x12, and RegOpMode
 * returns 0x2A instead of 0x83, for the whole ~1 s the frame is on the air;
 * both read correctly again the moment it ends. The likely cause is supply
 * droop -- the Uno's 3.3 V rail comes from the USB bridge's regulator and is
 * marginal for PA_BOOST -- but the software consequence is what matters here.
 *
 * 0xAA has bit 3 set, and bit 3 IS TxDone. So a naive poll reads garbage,
 * concludes the frame is finished after one millisecond, and drops the radio
 * into standby mid-transmission. The frames still arrived, because the abort
 * came after the RX had already heard them, so the only visible symptom was an
 * airtime of 1 ms against a model of 991 -- which looks like a misconfigured
 * modem and is not.
 *
 * The guard is to validate the bus before believing it: RegVersion is a
 * constant, so if it does not read back correctly the bus is unreliable at
 * this instant and the IRQ byte is not to be trusted either. Two reads instead
 * of one, on a path that is idle for seconds at a time.
 *
 * DIO0 would avoid the question, but its wiring is unconfirmed and differs
 * between the two bench boards -- see node_config.h.
 */
static bool irq_set(uint8_t mask)
{
    if (sx_read_reg(REG_VERSION) != g_version) return false;   /* bus is noisy */
    return (sx_read_reg(REG_IRQ_FLAGS) & mask) != 0;
}

/* ---- transmit ---------------------------------------------------------- */
bool sx_send(const uint8_t *buf, uint8_t len, uint16_t *air_ms)
{
    uint8_t i;
    uint32_t elapsed = 0;

    set_mode(MODE_STDBY);
    sx_write_reg(REG_FIFO_ADDR_PTR, 0x00);
    sx_write_reg(REG_PAYLOAD_LENGTH, len);

    nss_low();
    spi_xfer(REG_FIFO | 0x80);
    for (i = 0; i < len; i++) spi_xfer(buf[i]);
    nss_high();

    sx_write_reg(REG_IRQ_FLAGS, 0xFF);          /* write-1-to-clear */
    set_mode(MODE_TX);

    /* An SF12 83-byte frame is ~3.4 s, so the ceiling is generous. Timing in
     * 1 ms steps: the quantity being measured is seconds long, so a
     * millisecond tick is three digits of resolution and needs no timer. */
    while (!irq_set(IRQ_TX_DONE)) {
        _delay_ms(1);
        if (++elapsed > 8000UL) {
            sx_write_reg(REG_IRQ_FLAGS, 0xFF);
            set_mode(MODE_STDBY);
            if (air_ms) *air_ms = 0;
            return false;
        }
    }
    sx_write_reg(REG_IRQ_FLAGS, IRQ_TX_DONE);
    set_mode(MODE_STDBY);
    if (air_ms) *air_ms = (uint16_t)elapsed;
    return true;
}

/* ---- receive ----------------------------------------------------------- */
int16_t sx_recv(uint8_t *buf, uint8_t max, uint16_t timeout_ms)
{
    uint32_t waited = 0;
    uint8_t flags, len, cur, i;

    set_mode(MODE_STDBY);
    sx_write_reg(REG_IRQ_FLAGS, 0xFF);
    sx_write_reg(REG_FIFO_ADDR_PTR, 0x00);
    set_mode(MODE_RX_CONTINUOUS);

    for (;;) {
        /* The PA is not keyed while receiving, so the bus is quiet here -- but
         * the guard costs one register read and removes a whole class of
         * doubt, so it is applied on both paths. */
        if (irq_set(IRQ_RX_DONE)) break;
        _delay_ms(1);
        if (++waited >= timeout_ms) {
            set_mode(MODE_STDBY);
            return 0;
        }
    }
    flags = sx_read_reg(REG_IRQ_FLAGS);

    /* A CRC error is NOT silence. Collapsing the two would make a marginal
     * link look like an idle one, which is the opposite of the truth and
     * exactly the distinction a frame-loss measurement depends on. */
    if (flags & IRQ_CRC_ERROR) {
        sx_write_reg(REG_IRQ_FLAGS, 0xFF);
        set_mode(MODE_STDBY);
        return -1;
    }

    len = sx_read_reg(REG_RX_NB_BYTES);
    cur = sx_read_reg(REG_FIFO_RX_CURRENT);
    sx_write_reg(REG_FIFO_ADDR_PTR, cur);

    if (len > max) len = max;
    nss_low();
    spi_xfer(REG_FIFO & 0x7F);
    for (i = 0; i < len; i++) buf[i] = spi_xfer(0x00);
    nss_high();

    /* SNR is signed, quarter-dB. RSSI's offset is band-dependent: -157 for the
     * HF port (>=779 MHz), -164 for LF. Datasheet 5.5.5. */
    g_snr = (int8_t)(((int8_t)sx_read_reg(REG_PKT_SNR_VALUE)) / 4);
    {
        int16_t raw = (int16_t)sx_read_reg(REG_PKT_RSSI_VALUE);
        g_rssi = (int16_t)(raw + (SX_FREQ_HZ >= 779000000UL ? -157 : -164));
        /* Below the noise floor the packet RSSI needs the SNR correction. */
        if (g_snr < 0) g_rssi = (int16_t)(g_rssi + g_snr);
    }

    sx_write_reg(REG_IRQ_FLAGS, 0xFF);
    set_mode(MODE_STDBY);
    return (int16_t)len;
}

int16_t sx_last_rssi(void) { return g_rssi; }
int8_t  sx_last_snr(void)  { return g_snr; }
