/* test_host.c -- replay the Python reference's recorded trace against
 * edge_codec.c with native gcc. Blueprint 10.2 / 10.1 step 5.
 *
 * This turns "the C matches the Python" from an assumption into a checkable
 * fact. It compares u_hat, e, the flags AND the snapshot bytes; comparing only
 * the outputs would miss a state divergence that has not yet surfaced in an
 * output, which is precisely the divergence that hurts later.
 *
 * Any mismatch exits non-zero. There is no tolerance and no warning level:
 * the codec is integer arithmetic, so "close" has no meaning here.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "edge_codec.h"
#include "golden_vectors.h"

static int failures = 0;

static void fail(const char *what, uint32_t t, long got, long want)
{
    if (failures < 20)
        printf("  MISMATCH %-10s at t=%4u: got %ld, want %ld\n",
               what, t, got, want);
    failures++;
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

int main(void)
{
    codec_t st;
    step_result_t r;
    int skipped = 0;
    uint8_t snap[SNAPSHOT_BYTES];
    uint32_t t;
    int bi = 0, si = 0;

    printf("edge_codec.c vs edge/codec.py -- %d hours, eps=%u, u0=%d\n",
           GOLDEN_N, (unsigned)GOLDEN_EPS, GOLDEN_U0);
    printf("sizeof(codec_t) = %u bytes\n", (unsigned)sizeof(codec_t));

    if (SNAPSHOT_BYTES != GOLDEN_SNAPSHOT_BYTES) {
        printf("FATAL: snapshot is %d bytes here and %d in the vectors\n",
               SNAPSHOT_BYTES, GOLDEN_SNAPSHOT_BYTES);
        return 2;
    }

    /* The golden trace records the w_init it was produced with; if the header
     * pair has drifted apart the first prediction already differs and every
     * subsequent comparison is noise, so it is checked before anything else. */
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
    codec_init(&st, GOLDEN_EPS, GOLDEN_U0, GOLDEN_T_START);

    for (t = 0; t < (uint32_t)GOLDEN_N; t++) {
        if (GOLDEN_BOUNDARY[t] == 1) {
            /* the resync arrived and was acknowledged */
            codec_day_boundary(&st, GOLDEN_TRUE24[bi++]);
        } else if (GOLDEN_BOUNDARY[t] == 2) {
            /* lost: keep the old weights AND the old buffer, exactly as the
             * mirror does when it never receives the packet */
            codec_new_day(&st);
            skipped++;
        }

        codec_step(&st, GOLDEN_U[t], GOLDEN_CLAMPED[t] != 0, &r);

        if (r.u_hat != GOLDEN_UHAT[t]) fail("u_hat", t, r.u_hat, GOLDEN_UHAT[t]);
        if (r.e != GOLDEN_E[t])        fail("e", t, r.e, GOLDEN_E[t]);
        {
            uint8_t f = (uint8_t)((r.tx ? 1 : 0) | (r.event ? 2 : 0)
                                  | (r.clamped ? 4 : 0));
            if (f != GOLDEN_FLAGS[t]) fail("flags", t, f, GOLDEN_FLAGS[t]);
        }

        if (si < GOLDEN_N_SNAP && GOLDEN_SNAP_AT[si] == t) {
            codec_snapshot(&st, snap);
            if (memcmp(snap, GOLDEN_SNAP[si], SNAPSHOT_BYTES) != 0) {
                dump_snap_diff(t, snap, GOLDEN_SNAP[si]);
                failures++;
            }
            si++;
        }
    }

    if (si != GOLDEN_N_SNAP) {
        printf("FATAL: only %d of %d snapshots were reached\n",
               si, GOLDEN_N_SNAP);
        return 2;
    }
    if (bi != GOLDEN_N_BOUNDARY) {
        printf("FATAL: only %d of %d resyncs were applied\n",
               bi, GOLDEN_N_BOUNDARY);
        return 2;
    }

    if (failures) {
        printf("\nFAILED: %d mismatches. The C port is NOT equivalent to the "
               "Python reference,\nand a node flashed with it would "
               "desynchronise from the NanoPi.\n", failures);
        return 1;
    }
    printf("\nOK: %d hours, %d snapshots, %d resyncs applied, %d lost -- "
           "byte-identical.\n", GOLDEN_N, GOLDEN_N_SNAP, bi, skipped);
    printf("The C codec reproduces the specification exactly.\n");
    return 0;
}
