/* hil_host.c -- replay the golden trace against the codec running on a REAL
 * ATmega328P, over the serial port. The hardware half of test_host.c.
 *
 * test_host.c proves edge_codec.c matches edge/codec.py when compiled by gcc
 * for x86-64. That is the useful check, and it is not the whole claim: the
 * deployed codec is compiled by avr-gcc for an 8-bit target with 16-bit ints,
 * different shift and promotion costs, and a different compiler generation.
 * This program closes that gap by running the SAME vectors through the SAME
 * comparison set on the silicon the paper names.
 *
 * It deliberately includes ../golden_vectors.h -- the same fixture test_host.c
 * uses, unmodified. There is no second source of truth and no new generated
 * artifact: if the two programs ever disagree, the difference is the silicon,
 * which is the entire point.
 *
 * Exit: 0 byte-identical, 1 mismatch, 2 setup/protocol failure.
 */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <poll.h>
#include <time.h>

#include "../edge_codec.h"
#include "../golden_vectors.h"
#include "hil_proto.h"

#define REPLY_MAX 2048          /* a Q reply is 2 + 2*SNAPSHOT_BYTES + 1 */

static int  g_fd       = -1;
static int  failures   = 0;
static long g_rtt_ns   = 0;     /* cumulative time spent waiting on the board */
static long g_exchanges = 0;

/* ---- serial ------------------------------------------------------------ */

static int serial_open(const char *dev)
{
    struct termios tio;
    int fd = open(dev, O_RDWR | O_NOCTTY);
    if (fd < 0) { fprintf(stderr, "open %s: %s\n", dev, strerror(errno)); return -1; }

    if (tcgetattr(fd, &tio) != 0) {
        fprintf(stderr, "tcgetattr: %s\n", strerror(errno)); close(fd); return -1;
    }
    cfmakeraw(&tio);
    cfsetispeed(&tio, B115200);
    cfsetospeed(&tio, B115200);
    /* CLOCAL: we are not a modem. HUPCL: drop DTR on close, so the NEXT open
     * resets the board and every run starts from the same state -- the run is
     * idempotent by construction rather than by remembering to power-cycle. */
    tio.c_cflag |= (CLOCAL | CREAD | HUPCL);
    tio.c_cflag &= ~CRTSCTS;
    tio.c_cflag &= ~CSTOPB;
    tio.c_cflag &= ~PARENB;
    tio.c_cc[VMIN]  = 0;
    tio.c_cc[VTIME] = 0;        /* timeouts are done with poll() instead */

    if (tcsetattr(fd, TCSANOW, &tio) != 0) {
        fprintf(stderr, "tcsetattr: %s\n", strerror(errno)); close(fd); return -1;
    }
    return fd;
}

static long now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000L + ts.tv_nsec;
}

/* Reads one '\n'-terminated line. Returns length, or -1 on timeout/error. */
static int read_line(char *buf, size_t max, int timeout_ms)
{
    size_t n = 0;
    long deadline = now_ns() + (long)timeout_ms * 1000000L;

    for (;;) {
        struct pollfd pfd;
        long remain_ns = deadline - now_ns();
        int  ready;
        char c;

        if (remain_ns <= 0) { buf[n] = '\0'; return -1; }

        pfd.fd = g_fd; pfd.events = POLLIN; pfd.revents = 0;
        ready = poll(&pfd, 1, (int)(remain_ns / 1000000L) + 1);
        if (ready < 0) { if (errno == EINTR) continue; return -1; }
        if (ready == 0) { buf[n] = '\0'; return -1; }

        if (read(g_fd, &c, 1) != 1) continue;
        if (c == '\r') continue;
        if (c == '\n') { buf[n] = '\0'; return (int)n; }
        if (n + 1 < max) buf[n++] = c;
    }
}

static int write_all(const char *s)
{
    size_t left = strlen(s);
    while (left) {
        ssize_t w = write(g_fd, s, left);
        if (w < 0) { if (errno == EINTR) continue; return -1; }
        s += w; left -= (size_t)w;
    }
    return 0;
}

/* Send one command, read the reply, check it starts with the expected letter.
 *
 * Every reply echoes its command letter, and that is the whole
 * resynchronisation strategy: a stray banner or a leftover line from a
 * previous run is skipped here rather than being mistaken for an answer. */
static int exchange(const char *cmd, char expect, char *reply, size_t max)
{
    int tries;
    long t0;

    if (write_all(cmd) != 0) {
        fprintf(stderr, "write failed: %s\n", strerror(errno)); return -1;
    }
    t0 = now_ns();
    for (tries = 0; tries < 8; tries++) {
        int n = read_line(reply, max, 3000);
        if (n < 0) {
            fprintf(stderr, "timeout waiting for '%c' (sent: %.*s)\n",
                    expect, (int)strcspn(cmd, "\n"), cmd);
            return -1;
        }
        if (n == 0 || reply[0] == '#') continue;        /* banner or blank */
        if (reply[0] == 'E') {
            fprintf(stderr, "device error: %s\n", reply); return -1;
        }
        if (reply[0] != expect) continue;               /* stale line: skip */
        g_rtt_ns += now_ns() - t0;
        g_exchanges++;
        return 0;
    }
    fprintf(stderr, "no '%c' reply after 8 lines\n", expect);
    return -1;
}

/* ---- comparison -------------------------------------------------------- */

static void fail(const char *what, uint32_t t, long got, long want)
{
    if (failures < 20)
        printf("  MISMATCH %-10s at t=%4u: got %ld, want %ld\n",
               what, t, got, want);
    failures++;
}

static int hexval(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static void dump_snap_diff(uint32_t t, const uint8_t *got, const uint8_t *want)
{
    int i, shown = 0;
    printf("  MISMATCH snapshot at t=%u:\n", t);
    for (i = 0; i < SNAPSHOT_BYTES && shown < 12; i++) {
        if (got[i] != want[i]) {
            printf("    byte %3d: got 0x%02x, want 0x%02x\n", i, got[i], want[i]);
            shown++;
        }
    }
}

/* ---- handshake --------------------------------------------------------- */

/* Refuse to run against an image built from a different edge_config.h. Without
 * this the failure mode is an unexplained byte difference at an arbitrary
 * hour, which looks exactly like a real equivalence failure and is not one. */
static int handshake(void)
{
    char reply[REPLY_MAX];
    unsigned proto, board, sz, snapb, nw, buflen, syncsafe;
    char hash[64];
    int attempt;

    /* Opening the port resets the board into Optiboot. Poll until the
     * application answers; ~2.5 s covers the bootloader's window. */
    for (attempt = 0; attempt < 12; attempt++) {
        tcflush(g_fd, TCIOFLUSH);
        if (write_all("V\n") != 0) return -1;
        if (read_line(reply, sizeof reply, 250) > 0 && reply[0] == 'V') goto got;
    }
    fprintf(stderr, "no V reply -- is the HIL image flashed? (make hil-flash)\n");
    return -1;
got:
    if (sscanf(reply, "V %u %u %u %u %u %u %u %63s",
               &proto, &board, &sz, &snapb, &nw, &buflen, &syncsafe, hash) != 8) {
        fprintf(stderr, "malformed V reply: %s\n", reply); return -1;
    }

    printf("device: BOARD %u, proto %u, sizeof(codec_t) = %u B, snapshot %u B, "
           "N_W %u, BUF_LEN %u, SYNC_SAFE %u\n", board, proto, sz, snapb, nw,
           buflen, syncsafe);
    printf("host:   proto %u, sizeof(codec_t) = %u B, snapshot %u B, "
           "N_W %u, BUF_LEN %u, SYNC_SAFE %u\n", (unsigned)HIL_PROTO,
           (unsigned)sizeof(codec_t), (unsigned)SNAPSHOT_BYTES,
           (unsigned)N_W, (unsigned)BUF_LEN, (unsigned)SYNC_SAFE);

    if (proto != HIL_PROTO) {
        fprintf(stderr, "FATAL: device speaks protocol %u, host speaks %u -- "
                "reflash (make hil-flash)\n", proto, (unsigned)HIL_PROTO); return -1;
    }
    if (snapb != (unsigned)SNAPSHOT_BYTES || nw != (unsigned)N_W ||
        buflen != (unsigned)BUF_LEN || syncsafe != (unsigned)SYNC_SAFE) {
        fprintf(stderr, "FATAL: device was built from a different "
                "edge_config.h -- reflash (make hil-flash)\n"); return -1;
    }
    if (strcmp(hash, FEATURE_SPEC_HASH) != 0) {
        fprintf(stderr, "FATAL: feature spec hash differs: device %s, host %s\n",
                hash, FEATURE_SPEC_HASH); return -1;
    }
    /* sz is the AVR figure and WILL differ from the host's sizeof -- 486 vs
     * 488, which is findings/2026-09-14-firmware-sram.md. Reported, not
     * asserted. */
    if (sz != (unsigned)sizeof(codec_t))
        printf("note: codec_t is %u B on AVR and %u B here -- expected, see "
               "docs/findings/2026-09-14-firmware-sram.md\n",
               sz, (unsigned)sizeof(codec_t));
    return 0;
}

/* ---- main -------------------------------------------------------------- */

int main(int argc, char **argv)
{
    const char *dev = (argc > 1) ? argv[1] : "/dev/ttyACM0";
    char cmd[512], reply[REPLY_MAX];
    uint8_t snap[SNAPSHOT_BYTES];
    uint32_t t;
    int bi = 0, si = 0, skipped = 0;
    int e, uhat, urecon, tx, ev, cl;
    unsigned epse;

    printf("edge_codec.c on ATmega328P vs edge/codec.py -- %d hours, "
           "eps=%u, u0=%d\n", GOLDEN_N, (unsigned)GOLDEN_EPS, GOLDEN_U0);
    printf("port: %s\n", dev);
    /* The board that answers is named in the handshake below, because the port
     * name is not an identity -- both boards share a USB serial. */

    if (SNAPSHOT_BYTES != GOLDEN_SNAPSHOT_BYTES) {
        printf("FATAL: snapshot is %d bytes here and %d in the vectors\n",
               SNAPSHOT_BYTES, GOLDEN_SNAPSHOT_BYTES);
        return 2;
    }
    if (GOLDEN_N_W != N_W) {
        printf("FATAL: golden vectors have %d weights, firmware has %d\n",
               GOLDEN_N_W, N_W);
        return 2;
    }
    {
        static const int32_t w_cfg[N_W] = W_INIT;
        int i;
        for (i = 0; i < N_W; i++)
            if (w_cfg[i] != GOLDEN_W_INIT[i]) {
                printf("FATAL: w_init[%d] is %ld in edge_config.h and %ld in "
                       "the golden vectors -- regenerate both\n", i,
                       (long)w_cfg[i], (long)GOLDEN_W_INIT[i]);
                return 2;
            }
    }

    if ((g_fd = serial_open(dev)) < 0) return 2;
    if (handshake() != 0) { close(g_fd); return 2; }

    snprintf(cmd, sizeof cmd, "I %u %d %u\n",
             (unsigned)GOLDEN_EPS, GOLDEN_U0, (unsigned)GOLDEN_T_START);
    if (exchange(cmd, 'I', reply, sizeof reply) != 0) { close(g_fd); return 2; }

    for (t = 0; t < (uint32_t)GOLDEN_N; t++) {
        if (GOLDEN_BOUNDARY[t] == 1) {
            /* the resync arrived and was acknowledged */
            int i, off = snprintf(cmd, sizeof cmd, "B");
            for (i = 0; i < 24; i++)
                off += snprintf(cmd + off, sizeof cmd - (size_t)off, " %d",
                                GOLDEN_TRUE24[bi][i]);
            snprintf(cmd + off, sizeof cmd - (size_t)off, "\n");
            if (exchange(cmd, 'B', reply, sizeof reply) != 0) { close(g_fd); return 2; }
            bi++;
        } else if (GOLDEN_BOUNDARY[t] == 2) {
            /* lost: keep the old weights AND the old buffer, exactly as the
             * mirror does when it never receives the packet */
            if (exchange("N\n", 'N', reply, sizeof reply) != 0) { close(g_fd); return 2; }
            skipped++;
        }

        snprintf(cmd, sizeof cmd, "S %d %d\n",
                 GOLDEN_U[t], GOLDEN_CLAMPED[t] != 0 ? 1 : 0);
        if (exchange(cmd, 'S', reply, sizeof reply) != 0) { close(g_fd); return 2; }
        if (sscanf(reply, "S %d %d %d %u %d %d %d",
                   &e, &uhat, &urecon, &epse, &tx, &ev, &cl) != 7) {
            fprintf(stderr, "malformed S reply at t=%u: %s\n", t, reply);
            close(g_fd); return 2;
        }

        if (uhat != GOLDEN_UHAT[t]) fail("u_hat", t, uhat, GOLDEN_UHAT[t]);
        if (e    != GOLDEN_E[t])    fail("e", t, e, GOLDEN_E[t]);
        {
            uint8_t f = (uint8_t)((tx ? 1 : 0) | (ev ? 2 : 0) | (cl ? 4 : 0));
            if (f != GOLDEN_FLAGS[t]) fail("flags", t, f, GOLDEN_FLAGS[t]);
        }

        if (si < GOLDEN_N_SNAP && GOLDEN_SNAP_AT[si] == t) {
            const char *h;
            int i;
            if (exchange("Q\n", 'Q', reply, sizeof reply) != 0) { close(g_fd); return 2; }
            h = reply + 1;
            while (*h == ' ') h++;
            if ((int)strlen(h) != 2 * SNAPSHOT_BYTES) {
                fprintf(stderr, "snapshot at t=%u is %d hex chars, want %d\n",
                        t, (int)strlen(h), 2 * SNAPSHOT_BYTES);
                close(g_fd); return 2;
            }
            for (i = 0; i < SNAPSHOT_BYTES; i++) {
                int hi = hexval(h[2 * i]), lo = hexval(h[2 * i + 1]);
                if (hi < 0 || lo < 0) {
                    fprintf(stderr, "bad hex in snapshot at t=%u\n", t);
                    close(g_fd); return 2;
                }
                snap[i] = (uint8_t)((hi << 4) | lo);
            }
            if (memcmp(snap, GOLDEN_SNAP[si], SNAPSHOT_BYTES) != 0) {
                dump_snap_diff(t, snap, GOLDEN_SNAP[si]);
                failures++;
            }
            si++;
        }
    }

    close(g_fd);

    if (si != GOLDEN_N_SNAP) {
        printf("FATAL: only %d of %d snapshots were reached\n", si, GOLDEN_N_SNAP);
        return 2;
    }
    if (bi != GOLDEN_N_BOUNDARY) {
        printf("FATAL: only %d of %d resyncs were applied\n", bi, GOLDEN_N_BOUNDARY);
        return 2;
    }
    if (failures) {
        printf("\nFAILED: %d mismatches. The codec running on this ATmega328P "
               "is NOT equivalent\nto the Python reference, and a node flashed "
               "with it would desynchronise from the NanoPi.\n", failures);
        return 1;
    }

    printf("\nOK: %d hours, %d snapshots, %d resyncs applied, %d lost -- "
           "byte-identical ON THE ATmega328P.\n",
           GOLDEN_N, GOLDEN_N_SNAP, bi, skipped);
    if (g_exchanges)
        printf("%ld exchanges, mean round trip %.2f ms.\n",
               g_exchanges, (double)g_rtt_ns / (double)g_exchanges / 1e6);
    return 0;
}
