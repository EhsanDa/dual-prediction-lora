/* probe_host.c -- read the peripheral probe's report off the board.
 *
 * Deliberately dumb: the device does the thinking, this prints what it says.
 * Exits 0 when it sees "PROBE DONE", 1 if the board goes quiet first.
 */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <poll.h>
#include <time.h>

int main(int argc, char **argv)
{
    const char *dev = (argc > 1) ? argv[1] : "/dev/ttyACM0";
    struct termios tio;
    char line[512];
    size_t n = 0;
    int fd, done = 0;
    struct timespec ts;
    long deadline;

    if ((fd = open(dev, O_RDWR | O_NOCTTY)) < 0) {
        fprintf(stderr, "open %s: %s\n", dev, strerror(errno)); return 2;
    }
    if (tcgetattr(fd, &tio) != 0) { perror("tcgetattr"); close(fd); return 2; }
    cfmakeraw(&tio);
    cfsetispeed(&tio, B115200);
    cfsetospeed(&tio, B115200);
    tio.c_cflag |= (CLOCAL | CREAD | HUPCL);
    tio.c_cflag &= ~CRTSCTS;
    tio.c_cc[VMIN] = 0; tio.c_cc[VTIME] = 0;
    if (tcsetattr(fd, TCSANOW, &tio) != 0) { perror("tcsetattr"); close(fd); return 2; }

    /* The probe runs once at boot, so the reset that opening the port causes
     * IS the trigger. Nothing is sent to the board at all. */
    clock_gettime(CLOCK_MONOTONIC, &ts);
    deadline = ts.tv_sec * 1000000000L + ts.tv_nsec + 15L * 1000000000L;

    for (;;) {
        struct pollfd pfd = { fd, POLLIN, 0 };
        long remain;
        char c;

        clock_gettime(CLOCK_MONOTONIC, &ts);
        remain = deadline - (ts.tv_sec * 1000000000L + ts.tv_nsec);
        if (remain <= 0) break;

        if (poll(&pfd, 1, (int)(remain / 1000000L) + 1) <= 0) continue;
        if (read(fd, &c, 1) != 1) continue;
        if (c == '\r') continue;
        if (c == '\n') {
            line[n] = '\0';
            printf("%s\n", line);
            fflush(stdout);
            if (strstr(line, "PROBE DONE")) { done = 1; break; }
            n = 0;
            continue;
        }
        if (n + 1 < sizeof line) line[n++] = c;
    }

    close(fd);
    if (!done) {
        fprintf(stderr, "\nprobe did not finish within 15 s -- is the probe "
                        "image flashed? (make probe-flash)\n");
        return 1;
    }
    return 0;
}
