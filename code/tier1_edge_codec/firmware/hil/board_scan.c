/* board_scan.c -- find out which board is on which port, by asking.
 *
 * THE PROBLEM THIS SOLVES. The two Unos on the bench report the IDENTICAL USB
 * serial number, 24238313635351910130. They are distinct devices -- different
 * USB paths, different original flash contents -- but nothing in their USB
 * descriptors tells them apart. /dev/serial/by-id/ can hold only one symlink
 * for the pair, and ttyACM0/ttyACM1 are handed out in enumeration order, so a
 * replug can swap them silently.
 *
 * Attributing a measurement to "the board on ttyACM0" is therefore unsound.
 * Every image built here reports a BOARD_ID baked in at flash time, and this
 * program opens each port, asks `V`, and prints who answered.
 *
 *   ./board_scan            list every port and the board on it
 *   ./board_scan 2          print the port for board 2, or exit 1
 *
 * The second form is what the Makefile uses to resolve BOARD= into a port.
 */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <poll.h>
#include <time.h>
#include <glob.h>

#include "hil_proto.h"

static long now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000L + ts.tv_nsec / 1000000L;
}

static int read_line(int fd, char *buf, size_t max, int timeout_ms)
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

/* Returns the board id, or -1 (unknown), or -2 (answered, but speaks a
 * DIFFERENT protocol version -- i.e. a stale image is flashed).
 *
 * The version check is the point of the exercise. An image built before
 * BOARD_ID existed still answers V, and its second field is sizeof(codec_t),
 * which would be reported as "board 486". Silently believing that is exactly
 * the failure this program exists to prevent, so a proto mismatch is named. */
static int identify_once(const char *dev, char *what, size_t whatmax)
{
    struct termios tio;
    char line[512];
    int fd, attempt, board = -1;
    long deadline;

    what[0] = '\0';
    if ((fd = open(dev, O_RDWR | O_NOCTTY)) < 0) return -1;
    if (tcgetattr(fd, &tio) != 0) { close(fd); return -1; }
    cfmakeraw(&tio);
    cfsetispeed(&tio, B115200);
    cfsetospeed(&tio, B115200);
    tio.c_cflag |= (CLOCAL | CREAD | HUPCL);
    tio.c_cflag &= ~CRTSCTS;
    tio.c_cc[VMIN] = 0; tio.c_cc[VTIME] = 0;
    if (tcsetattr(fd, TCSANOW, &tio) != 0) { close(fd); return -1; }

    /* Opening reset the board into Optiboot; it holds the port about a second.
     * Banners are collected as they go past -- the probe image never answers V,
     * so its banner is the only way to recognise it. */
    /* A streaming image (node) floods the port right after its banner, so the
     * banner may already have gone past before we opened. HUPCL on close and
     * the open's DTR pulse reset it, so the banner is always about to arrive;
     * we just have to read far enough to catch it. */
    deadline = now_ms() + 4000;
    for (attempt = 0; board < 0 && now_ms() < deadline; attempt++) {
        int n;
        if (write(fd, "V\n", 2) != 2) break;
        while ((n = read_line(fd, line, sizeof line, 250)) >= 0) {
            unsigned proto, b;
            if (n == 0) continue;
            if (line[0] == '#') {
                if (strstr(line, "PROBE") && !what[0])
                    snprintf(what, whatmax, "probe");
                if (sscanf(line, "# PROBE Arduino peripheral discovery, proto %u board %u",
                           &proto, &b) == 2) {
                    board = (proto == HIL_PROTO) ? (int)b : -2;
                    break;
                }
                if (sscanf(line, "# READY hil proto %u board %u", &proto, &b) == 2) {
                    snprintf(what, whatmax, "hil");
                    board = (proto == HIL_PROTO) ? (int)b : -2;
                    break;
                }
                /* The node application and the radio bench announce themselves
                 * the same way but never answer V -- they are not command
                 * protocols. The banner is the only identification they offer,
                 * which is exactly why every image emits one. */
                if (sscanf(line, "# READY node proto %u board %u", &proto, &b) == 2) {
                    snprintf(what, whatmax, "node");
                    board = (proto == HIL_PROTO) ? (int)b : -2;
                    break;
                }
                if (sscanf(line, "# READY radio proto %u board %u", &proto, &b) == 2) {
                    snprintf(what, whatmax, "radio");
                    board = (proto == HIL_PROTO) ? (int)b : -2;
                    break;
                }
                continue;
            }
            if (line[0] == 'V' && sscanf(line, "V %u %u", &proto, &b) == 2) {
                snprintf(what, whatmax, "hil");
                board = (proto == HIL_PROTO) ? (int)b : -2;
                break;
            }
        }
    }
    close(fd);
    return board;
}

/* Closing the port drops DTR and REOPENING pulses it, which resets the board.
 * So a retry is a genuinely fresh start, not a second look at the same stream.
 * That matters for the streaming images: the node application floods the port
 * the moment it boots, and a scan that arrives mid-flood sees no banner. Two
 * cycles is enough in practice; the cost of a spurious "not found" is a failed
 * flash against the wrong board, so it is worth paying for. */
static int identify(const char *dev, char *what, size_t whatmax)
{
    int tries, b = -1;
    for (tries = 0; tries < 3; tries++) {
        b = identify_once(dev, what, whatmax);
        if (b >= 0 || b == -2) return b;
        usleep(200000);
    }
    return b;
}

int main(int argc, char **argv)
{
    glob_t g;
    size_t i;
    int want = (argc > 1) ? atoi(argv[1]) : -1;
    int found = 0, stale = 0;

    if (glob("/dev/ttyACM*", 0, NULL, &g) != 0 || g.gl_pathc == 0) {
        if (want < 0) printf("no /dev/ttyACM* found\n");
        else fprintf(stderr, "no /dev/ttyACM* found\n");
        return 1;
    }

    if (want < 0)
        printf("%-16s %-7s %s\n", "PORT", "BOARD", "IMAGE");

    for (i = 0; i < g.gl_pathc; i++) {
        char what[32];
        int b = identify(g.gl_pathv[i], what, sizeof what);
        if (want < 0) {
            char id[16];
            if (b == -2)     snprintf(id, sizeof id, "stale");
            else if (b < 0)  snprintf(id, sizeof id, "?");
            else             snprintf(id, sizeof id, "%d", b);
            printf("%-16s %-7s %s\n", g.gl_pathv[i], id,
                   what[0] ? what : "(no answer)");
            if (b == -2) stale++;
        }
        if (want >= 0 && b == want) { printf("%s\n", g.gl_pathv[i]); found = 1; break; }
    }
    globfree(&g);

    if (want < 0 && stale)
        printf("\n%d board(s) run an image older than protocol %d and cannot be\n"
               "identified. Reflash: make hil-flash BOARD=<n>\n",
               stale, (int)HIL_PROTO);

    if (want >= 0 && !found) {
        fprintf(stderr, "board %d not found on any /dev/ttyACM*.\n"
                        "Run `make boards` to see what is attached, and\n"
                        "`make hil-flash BOARD=%d` to give a board that id.\n",
                want, want);
        return 1;
    }
    return 0;
}
