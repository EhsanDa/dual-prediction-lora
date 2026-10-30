/* radio_host.c -- drive two boards at once: one transmits, one listens.
 *
 *   --scan   which band are these modules actually built for?
 *   --link   do real encoded frames cross the air and decode?
 *   --toa    does measured time-on-air match edge/protocol.py's model?
 *
 * A single board cannot test a radio: it cannot hear itself. This is what the
 * second Uno buys, and it is also the precondition for card P3-01 step 5.
 */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <poll.h>
#include <time.h>

static long now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000L + ts.tv_nsec / 1000000L;
}

static int port_open(const char *dev)
{
    struct termios tio;
    int fd = open(dev, O_RDWR | O_NOCTTY);
    if (fd < 0) { fprintf(stderr, "open %s: %s\n", dev, strerror(errno)); return -1; }
    if (tcgetattr(fd, &tio) != 0) { close(fd); return -1; }
    cfmakeraw(&tio);
    cfsetispeed(&tio, B115200);
    cfsetospeed(&tio, B115200);
    tio.c_cflag |= (CLOCAL | CREAD | HUPCL);
    tio.c_cflag &= ~CRTSCTS;
    tio.c_cc[VMIN] = 0; tio.c_cc[VTIME] = 0;
    if (tcsetattr(fd, TCSANOW, &tio) != 0) { close(fd); return -1; }
    return fd;
}

static int rd_line(int fd, char *buf, size_t max, int timeout_ms)
{
    size_t n = 0;
    long deadline = now_ms() + timeout_ms;
    for (;;) {
        struct pollfd pfd = { fd, POLLIN, 0 };
        long remain = deadline - now_ms();
        char c;
        if (remain <= 0) { buf[n] = '\0'; return -1; }
        if (poll(&pfd, 1, (int)remain) <= 0) { buf[n] = '\0'; return -1; }
        if (read(fd, &c, 1) != 1) continue;
        if (c == '\r') continue;
        if (c == '\n') { buf[n] = '\0'; return (int)n; }
        if (n + 1 < max) buf[n++] = c;
    }
}

static int wr(int fd, const char *s)
{
    size_t left = strlen(s);
    while (left) {
        ssize_t w = write(fd, s, left);
        if (w < 0) { if (errno == EINTR) continue; return -1; }
        s += w; left -= (size_t)w;
    }
    return 0;
}

/* Send a command and wait for the reply beginning with `expect`. */
static int ask(int fd, const char *cmd, char expect, char *reply, size_t max,
               int timeout_ms)
{
    int tries;
    if (wr(fd, cmd) != 0) return -1;
    for (tries = 0; tries < 40; tries++) {
        int n = rd_line(fd, reply, max, timeout_ms);
        if (n < 0) return -1;
        if (n == 0 || reply[0] == '#') continue;
        if (reply[0] == 'E') { fprintf(stderr, "device: %s\n", reply); return -1; }
        if (reply[0] == expect) return 0;
    }
    return -1;
}

static int wait_ready(int fd, int *board, int *ver)
{
    char line[256];
    int tries;
    tcflush(fd, TCIOFLUSH);
    for (tries = 0; tries < 20; tries++) {
        if (ask(fd, "V\n", 'V', line, sizeof line, 400) == 0) {
            unsigned proto, b, v;
            if (sscanf(line, "V %u %u %u", &proto, &b, &v) == 3) {
                *board = (int)b; *ver = (int)v;
                return 0;
            }
        }
    }
    return -1;
}

/* Semtech SX1276 datasheet 4.1.1.6, the same formula edge/protocol.py:238
 * implements. Reproduced here so a measurement is compared with the model
 * rather than with a remembered number. */
static double time_on_air(int payload, int sf, double bw, int cr, int preamble,
                          int explicit_header, int crc_on)
{
    double tsym = pow(2.0, sf) / bw;
    int de = (tsym > 0.016) ? 1 : 0;
    double num = 8.0 * payload - 4.0 * sf + 28.0 + (crc_on ? 16.0 : 0.0)
                 - (explicit_header ? 0.0 : 20.0);
    double den = 4.0 * (sf - 2.0 * de);
    double n = ceil(num / den) * (cr + 4);
    double npay = (n < 0.0 ? 0.0 : n) + 8.0;
    return (preamble + 4.25) * tsym + npay * tsym;
}

struct band { const char *name; unsigned long hz; };
static const struct band BANDS[] = {
    { "433 MHz", 433000000UL },
    { "470 MHz", 470000000UL },
    { "868 MHz", 868000000UL },
    { "915 MHz", 915000000UL },
};
#define NBANDS ((int)(sizeof BANDS / sizeof BANDS[0]))

#define FRAMES_PER_BAND 3

static int do_scan(int tx, int rx)
{
    int i, best = -1;
    double best_rssi = -1e9;

    printf("\nBAND SCAN -- %d frames per band, PA at its minimum setting\n",
           FRAMES_PER_BAND);
    printf("The module's matching network and filter are band-specific, so the\n"
           "band it is built for answers far louder than the others.\n\n");
    printf("%-10s %8s %10s %8s\n", "BAND", "HEARD", "RSSI dBm", "SNR dB");

    for (i = 0; i < NBANDS; i++) {
        char cmd[64], reply[512];
        int heard = 0, k;
        double rssi_sum = 0, snr_sum = 0;

        snprintf(cmd, sizeof cmd, "F %lu\n", BANDS[i].hz);
        if (ask(tx, cmd, 'F', reply, sizeof reply, 2000) != 0) return 2;
        if (ask(rx, cmd, 'F', reply, sizeof reply, 2000) != 0) return 2;

        for (k = 0; k < FRAMES_PER_BAND; k++) {
            int len = 0, r = 0, s = 0;
            /* Arm the receiver first: it blocks in RX while the sender keys. */
            if (wr(rx, "R 6000\n") != 0) return 2;
            usleep(150000);
            if (ask(tx, "T 1\n", 'T', reply, sizeof reply, 12000) != 0) return 2;
            if (rd_line(rx, reply, sizeof reply, 8000) < 0) continue;
            if (sscanf(reply, "R %d %d %d", &len, &r, &s) >= 1 && len > 0) {
                heard++; rssi_sum += r; snr_sum += s;
            }
        }
        if (heard) {
            double ar = rssi_sum / heard, as = snr_sum / heard;
            printf("%-10s %4d/%-3d %10.1f %8.1f\n", BANDS[i].name, heard,
                   FRAMES_PER_BAND, ar, as);
            if (ar > best_rssi) { best_rssi = ar; best = i; }
        } else {
            printf("%-10s %4d/%-3d %10s %8s\n", BANDS[i].name, 0,
                   FRAMES_PER_BAND, "-", "-");
        }
    }

    if (best < 0) {
        printf("\nNothing heard on any band. Check the antennas and that both\n"
               "boards carry the radio image (make boards).\n");
        return 1;
    }
    printf("\nSTRONGEST: %s at %.1f dBm.\n", BANDS[best].name, best_rssi);
    printf("Set SX_FREQ_HZ in node_config.h accordingly -- and confirm that\n"
           "band is licensed where the node will run before transmitting at\n"
           "anything above the minimum power used here.\n");
    return 0;
}

static int do_link(int tx, int rx)
{
    char reply[512], rxline[512];
    int len = 0, r = 0, s = 0, tlen = 0, air = 0, fails = 0;
    const char *kinds[2] = { "X", "Y" };
    const char *names[2] = { "exception", "resync" };
    int i;

    printf("\nLINK TEST -- real encoded frames, built by the same encoder the\n"
           "application uses (node_protocol.c), across the air.\n\n");

    for (i = 0; i < 2; i++) {
        char cmd[8];
        snprintf(cmd, sizeof cmd, "%s\n", kinds[i]);

        if (wr(rx, "R 12000\n") != 0) return 2;
        usleep(150000);
        if (ask(tx, cmd, kinds[i][0], reply, sizeof reply, 20000) != 0) return 2;
        sscanf(reply + 1, "%d %d", &tlen, &air);

        if (rd_line(rx, rxline, sizeof rxline, 15000) < 0) {
            printf("  %-10s sent %3d B in %4d ms -- NOT RECEIVED\n",
                   names[i], tlen, air);
            fails++;
            continue;
        }
        len = 0;
        sscanf(rxline, "R %d %d %d", &len, &r, &s);
        if (len == tlen) {
            printf("  %-10s sent %3d B in %4d ms -- received %3d B, "
                   "RSSI %d dBm, SNR %d dB, CRC ok\n",
                   names[i], tlen, air, len, r, s);
        } else if (len < 0) {
            printf("  %-10s sent %3d B -- received with a BAD CRC\n",
                   names[i], tlen);
            fails++;
        } else {
            printf("  %-10s sent %3d B -- received %d B, LENGTH MISMATCH\n",
                   names[i], tlen, len);
            fails++;
        }
    }

    if (fails) {
        printf("\n%d of 2 frame types did not make it.\n", fails);
        return 1;
    }
    printf("\nOK: both frame types crossed the air and arrived CRC-clean.\n");
    return 0;
}

static int do_toa(int tx)
{
    char reply[512];
    int tlen = 0, air = 0, bad = 0;
    const char *kinds[2] = { "X", "Y" };
    const char *names[2] = { "exception", "resync" };
    int i;

    printf("\nTIME ON AIR -- measured against edge/protocol.py's Semtech model\n");
    printf("SF12, BW125, CR4/5, 8-symbol preamble, explicit header, CRC on.\n");
    printf("This validates the DRIVER's configuration. It is not card P3-01\n");
    printf("step 4, which is a deliberate measurement the author runs.\n\n");
    printf("%-10s %6s %10s %10s %8s\n", "FRAME", "BYTES", "MEAS ms", "MODEL ms", "DIFF");

    for (i = 0; i < 2; i++) {
        char cmd[8];
        double model;
        snprintf(cmd, sizeof cmd, "%s\n", kinds[i]);
        if (ask(tx, cmd, kinds[i][0], reply, sizeof reply, 20000) != 0) return 2;
        if (sscanf(reply + 1, "%d %d", &tlen, &air) != 2 || tlen == 0) {
            printf("%-10s  transmit failed\n", names[i]);
            bad++; continue;
        }
        model = time_on_air(tlen, 12, 125000.0, 1, 8, 1, 1) * 1000.0;
        printf("%-10s %6d %10d %10.0f %7.1f%%\n", names[i], tlen, air, model,
               100.0 * (air - model) / model);
        /* The device times in 1 ms polling steps and includes mode-switch
         * latency, so a couple of percent is expected. Ten is not. */
        if (fabs(air - model) / model > 0.10) bad++;
    }

    if (bad) {
        printf("\nMeasured airtime departs from the model by more than 10%%.\n"
               "The radio is not configured as the model assumes -- check SF,\n"
               "bandwidth, coding rate and the low-data-rate optimisation.\n");
        return 1;
    }
    printf("\nOK: measured airtime agrees with the model, so the driver is\n"
           "configured as edge/protocol.py assumes.\n");
    return 0;
}

int main(int argc, char **argv)
{
    const char *txdev = NULL, *rxdev = NULL;
    int mode = 0, i, tx = -1, rx = -1, b, v, rc;

    for (i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--scan")) mode = 1;
        else if (!strcmp(argv[i], "--link")) mode = 2;
        else if (!strcmp(argv[i], "--toa"))  mode = 3;
        else if (!strcmp(argv[i], "--tx") && i + 1 < argc) txdev = argv[++i];
        else if (!strcmp(argv[i], "--rx") && i + 1 < argc) rxdev = argv[++i];
    }
    if (!mode || !txdev || (mode != 3 && !rxdev)) {
        fprintf(stderr, "usage: radio_host --scan|--link|--toa "
                        "--tx <port> [--rx <port>]\n");
        return 2;
    }

    if ((tx = port_open(txdev)) < 0) return 2;
    if (wait_ready(tx, &b, &v) != 0) {
        fprintf(stderr, "no radio image answering on %s -- "
                        "make radio-flash BOARD=n\n", txdev);
        return 2;
    }
    printf("TX: %s  board %d  RegVersion 0x%02x%s\n", txdev, b, v,
           v == 0x12 ? " (SX1276/77/78/79)" : v == 0x22 ? " (SX1272/73)" : " -- NO RADIO");
    if (v != 0x12 && v != 0x22) return 2;

    if (rxdev) {
        if ((rx = port_open(rxdev)) < 0) return 2;
        if (wait_ready(rx, &b, &v) != 0) {
            fprintf(stderr, "no radio image answering on %s\n", rxdev);
            return 2;
        }
        printf("RX: %s  board %d  RegVersion 0x%02x%s\n", rxdev, b, v,
               v == 0x12 ? " (SX1276/77/78/79)" : v == 0x22 ? " (SX1272/73)" : " -- NO RADIO");
        if (v != 0x12 && v != 0x22) return 2;
    }

    rc = (mode == 1) ? do_scan(tx, rx)
       : (mode == 2) ? do_link(tx, rx)
                     : do_toa(tx);

    close(tx);
    if (rx >= 0) close(rx);
    return rc;
}
