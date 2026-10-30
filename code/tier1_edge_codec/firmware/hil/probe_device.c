/* probe_device.c -- peripheral discovery. ATmega328P. Bring-up only.
 *
 * NOT part of the reported pipeline and not part of the codec. It answers one
 * question: what is actually wired to this board? The host cannot answer it --
 * anything on the Arduino's pins sits behind the MCU and is invisible to USB,
 * so the only instrument that can see a sensor is the board itself.
 *
 * WHAT THIS CAN AND CANNOT ESTABLISH, because the difference matters when
 * reading the output:
 *
 *   I2C      an address that ACKs is POSITIVE evidence something is there.
 *            The part name printed beside it is a lookup of a conventional
 *            address and is a GUESS -- several families share addresses.
 *   SX127x   reading 0x12 from RegVersion is a specific magic number, so a hit
 *            is strong evidence of an SX1276/77/78/79. A miss only means "not
 *            on the NSS pins tried".
 *   analog   a voltage tells you a pin is driven, not what drives it.
 *            MQ-series gas sensors are resistive and have NO digital identity:
 *            this probe can show you a reading and can never name the part.
 *   digital  a pin that will not pull up has something holding it low. What,
 *            is not knowable from here.
 *
 * IT IS NOT PASSIVE. An I2C address sweep and an SPI NSS assertion are writes
 * to a shared bus. They are as gentle as a probe can be -- no device register
 * is ever written -- but a part mid-transaction can be disturbed. Do not run
 * this against a live deployment.
 */
#include <stdint.h>
#include <stdbool.h>
#include <avr/io.h>
#include <util/delay.h>

#include "hil_uart.h"
#include "hil_proto.h"

/* ---- I2C (TWI) --------------------------------------------------------- */

static void twi_init(void)
{
    /* ~100 kHz at 16 MHz: SCL = F_CPU / (16 + 2*TWBR*prescaler), TWBR = 72,
     * prescaler 1. Slow on purpose -- a long bus with no pull-ups to spare
     * fails at 400 kHz and that failure looks like "nothing is connected". */
    TWSR = 0;
    TWBR = 72;
    TWCR = (1 << TWEN);
}

static uint8_t twi_wait(void)
{
    uint16_t guard = 0;
    /* A bus held low by a wedged device would hang forever without this. */
    while (!(TWCR & (1 << TWINT))) { if (++guard > 20000) return 0xFF; _delay_us(1); }
    return TWSR & 0xF8;
}

static void twi_stop(void)
{
    TWCR = (1 << TWINT) | (1 << TWEN) | (1 << TWSTO);
    _delay_us(50);
}

/* true if the address ACKed a write. */
static bool twi_probe(uint8_t addr)
{
    uint8_t st;

    TWCR = (1 << TWINT) | (1 << TWSTA) | (1 << TWEN);
    st = twi_wait();
    if (st != 0x08 && st != 0x10) { twi_stop(); return false; }   /* START */

    TWDR = (uint8_t)(addr << 1);                                  /* SLA+W */
    TWCR = (1 << TWINT) | (1 << TWEN);
    st = twi_wait();
    twi_stop();
    return (st == 0x18);                                          /* ACK */
}

/* Conventional addresses. A GUESS, printed as one. */
static const char *i2c_guess(uint8_t a)
{
    switch (a) {
    case 0x0C: case 0x0D:            return "AK8963/HMC5883 magnetometer?";
    case 0x18: case 0x19:            return "LIS3DH/LSM303 accel?";
    case 0x20: case 0x21: case 0x27: return "PCF8574 / LCD backpack?";
    case 0x23:                       return "BH1750 light?";
    case 0x29:                       return "TSL2561/VL53L0X?";
    case 0x38: case 0x39:            return "AHT10/AHT20 or APDS-9960?";
    case 0x3C: case 0x3D:            return "SSD1306 OLED?";
    case 0x40:                       return "SHT21/HTU21/INA219?";
    case 0x44: case 0x45:            return "SHT3x temp/humidity?";
    case 0x48: case 0x49: case 0x4A: case 0x4B: return "ADS1115 / LM75?";
    case 0x4D:                       return "MCP3021 / ADS1015?";
    case 0x50: case 0x57:            return "AT24Cxx EEPROM / MAX30102?";
    case 0x53:                       return "ADXL345?";
    case 0x58:                       return "SGP30 GAS sensor?";
    case 0x5A:                       return "CCS811 GAS (or MLX90614)?";
    case 0x5B:                       return "CCS811 GAS (ADDR high)?";
    case 0x5C: case 0x5D:            return "AM2320 / BMP180?";
    case 0x68:                       return "DS3231/DS1307 RTC or MPU6050?";
    case 0x69:                       return "MPU6050 (AD0 high)?";
    case 0x6A: case 0x6B:            return "LSM6DS3 / L3GD20?";
    case 0x76:                       return "BME280/BMP280/BME680 (gas)?";
    case 0x77:                       return "BME280/BMP280/BME680 (alt)?";
    default:                         return "";
    }
}

/* ---- SPI / SX127x ------------------------------------------------------ */

/* Arduino digital pin -> (port, bit). Only the pins we use as NSS candidates.
 * D2-D7 are PORTD 2-7; D8-D13 are PORTB 0-5. */
static void nss_high(uint8_t pin) {
    if (pin < 8) { DDRD |= (uint8_t)(1 << pin);       PORTD |= (uint8_t)(1 << pin); }
    else         { DDRB |= (uint8_t)(1 << (pin - 8)); PORTB |= (uint8_t)(1 << (pin - 8)); }
}
static void nss_low(uint8_t pin) {
    if (pin < 8) PORTD &= (uint8_t)~(1 << pin);
    else         PORTB &= (uint8_t)~(1 << (pin - 8));
}

static void spi_init(void)
{
    /* SCK PB5, MOSI PB3, SS PB2 out; MISO PB4 in. SS must be an output or the
     * SPI block drops out of master mode the moment something pulls it low. */
    DDRB |= (1 << PB5) | (1 << PB3) | (1 << PB2);
    DDRB &= (uint8_t)~(1 << PB4);
    SPCR = (1 << SPE) | (1 << MSTR) | (1 << SPR1);   /* f/64 = 250 kHz, mode 0 */
    SPSR = 0;
}

static uint8_t spi_xfer(uint8_t b)
{
    SPDR = b;
    while (!(SPSR & (1 << SPIF))) { }
    return SPDR;
}

static uint8_t sx127x_read_reg(uint8_t nss, uint8_t reg)
{
    uint8_t v;
    nss_low(nss);
    _delay_us(5);
    spi_xfer(reg & 0x7F);        /* bit7 clear = read; never a write */
    v = spi_xfer(0x00);
    _delay_us(5);
    nss_high(nss);
    return v;
}

/* ---- ADC --------------------------------------------------------------- */

static void adc_init(void)
{
    ADMUX  = (1 << REFS0);                            /* AVcc reference */
    ADCSRA = (1 << ADEN) | (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0); /* /128 */
}

static uint16_t adc_read(uint8_t ch)
{
    ADMUX = (uint8_t)((ADMUX & 0xF0) | (ch & 0x0F));
    _delay_us(200);                                   /* let the mux settle */
    ADCSRA |= (1 << ADSC);
    while (ADCSRA & (1 << ADSC)) { }
    return ADC;
}

/* ---- report ------------------------------------------------------------ */

static void hr(void) { uart_puts("# ----------------------------------------\n"); }

int main(void)
{
    uint8_t a, i, found;
    static const uint8_t nss_cands[] = {10, 8, 6, 7, 9, 4, 2, 3};

    uart_init();
    uart_puts("# PROBE Arduino peripheral discovery, proto ");
    uart_put_u32(HIL_PROTO);
    uart_puts(" board ");
    uart_put_u32(BOARD_ID);
    uart_putc('\n');
    uart_puts("# read-only by intent: no device register is ever written\n");

    /* ---- I2C ---- */
    hr();
    uart_puts("I2C scan 0x08-0x77 (100 kHz):\n");
    twi_init();
    found = 0;
    for (a = 0x08; a <= 0x77; a++) {
        if (twi_probe(a)) {
            const char *g = i2c_guess(a);
            uart_puts("  ACK 0x"); uart_put_hex8(a);
            if (g[0]) { uart_puts("  "); uart_puts(g); }
            uart_putc('\n');
            found++;
        }
    }
    if (!found) uart_puts("  none -- no I2C device answered\n");
    uart_puts("I2C devices: "); uart_put_u32(found); uart_putc('\n');

    /* ---- SPI / SX127x ---- */
    hr();
    uart_puts("SX127x LoRa probe (RegVersion 0x42 over SPI):\n");
    spi_init();
    for (i = 0; i < sizeof(nss_cands); i++) nss_high(nss_cands[i]);
    _delay_ms(10);
    found = 0;
    for (i = 0; i < sizeof(nss_cands); i++) {
        uint8_t pin = nss_cands[i];
        uint8_t v   = sx127x_read_reg(pin, 0x42);
        const char *what =
            (v == 0x12) ? "SX1276/77/78/79 -- LoRa CONFIRMED" :
            (v == 0x22) ? "SX1272/73 -- LoRa CONFIRMED"       :
            (v == 0x00 || v == 0xFF) ? "" : "unknown device responding";
        uart_puts("  NSS D");
        uart_put_u32(pin);
        uart_puts(" -> 0x"); uart_put_hex8(v);
        if (what[0]) { uart_puts("  "); uart_puts(what); found++; }
        uart_putc('\n');
    }
    if (!found)
        uart_puts("  no SX127x on these NSS pins (0x00/0xFF = nothing driving MISO)\n");

    /* ---- analog ---- */
    hr();
    uart_puts("Analog A0-A5 (10-bit, AVcc ref), floating then pulled up:\n");
    adc_init();
    for (i = 0; i < 6; i++) {
        uint16_t f, p;
        /* floating */
        DDRC  &= (uint8_t)~(1 << i);
        PORTC &= (uint8_t)~(1 << i);
        _delay_ms(2);
        f = adc_read(i);
        /* internal pull-up ~36k: a low reading here means something is holding
         * the pin down, i.e. a real source rather than an open input */
        PORTC |= (uint8_t)(1 << i);
        _delay_ms(5);
        p = adc_read(i);
        PORTC &= (uint8_t)~(1 << i);

        uart_puts("  A"); uart_put_u32(i);
        uart_puts("  float="); uart_put_u32(f);
        uart_puts("  pullup="); uart_put_u32(p);
        if (p < 900) uart_puts("   <- driven, something is connected");
        else if (f > 20) uart_puts("   <- floating but not at rail");
        uart_putc('\n');
    }
    uart_puts("note: an MQ-series gas sensor can appear here as a voltage and\n"
              "      can NEVER be identified by this probe -- it is resistive.\n");

    /* ---- digital ---- */
    hr();
    uart_puts("Digital D2-D9 pull-up test:\n");
    for (i = 2; i <= 9; i++) {
        uint8_t up;
        if (i < 8) { DDRD &= (uint8_t)~(1 << i);       PORTD |= (uint8_t)(1 << i); }
        else       { DDRB &= (uint8_t)~(1 << (i - 8)); PORTB |= (uint8_t)(1 << (i - 8)); }
        _delay_ms(2);
        up = (i < 8) ? (uint8_t)((PIND >> i) & 1)
                     : (uint8_t)((PINB >> (i - 8)) & 1);
        if (i < 8) PORTD &= (uint8_t)~(1 << i);
        else       PORTB &= (uint8_t)~(1 << (i - 8));

        uart_puts("  D"); uart_put_u32(i);
        uart_puts(up ? "  pulls up (open or high-Z)\n"
                     : "  HELD LOW <- something is driving it\n");
    }

    hr();
    uart_puts("# PROBE DONE\n");

    for (;;) { }
}
