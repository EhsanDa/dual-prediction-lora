/* edge_codec.c -- see edge_codec.h. Transliterated from edge/codec.py.
 *
 * Every non-obvious line carries the reason it is written that way, because
 * the failure mode of a "close enough" port here is silent desynchronisation
 * rather than a visibly wrong number.
 */
#include <string.h>
#include "edge_codec.h"

static const uint8_t TAPS[N_TAPS] = TAP_LIST;
static const int32_t W_INIT_ARR[N_W] = W_INIT;

/* ------------------------------------------------------------ primitives */

/* `>>` on a signed type is an ARITHMETIC shift on gcc and avr-gcc, and Python's
 * `>>` floors for negatives too, so the two agree. It is implementation-defined
 * in C89/C99 and defined as arithmetic in C++20; every compiler this will ever
 * see does the arithmetic thing. Named rather than inlined so the fact is
 * recorded where it is relied upon. NOTE it is not division: -5 >> 1 is -3
 * while -5 / 2 is -2. The NLMS update depends on the floor, identically on
 * both sides, so the bias cancels out of the synchronisation argument. */
int32_t ec_sar32(int32_t x, uint8_t n) { return x >> n; }
int64_t ec_sar64(int64_t x, uint8_t n) { return x >> n; }

int32_t ec_clamp(int32_t x, int32_t lo, int32_t hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

/* Round-half-up toward +inf, because the shift floors. Matches
 * edge.fixedpoint.round_shift exactly; do NOT "fix" this to round-half-even. */
int32_t ec_round_shift(int64_t x, uint8_t n)
{
    if (n == 0) return (int32_t)x;
    return (int32_t)((x + ((int64_t)1 << (n - 1))) >> n);
}

/* msb_index(0) == 0, matching the Python helper. Shift-and-count rather than
 * __builtin_clz so the AVR build and the host build are the same code. */
uint8_t ec_msb_index(uint64_t x)
{
    uint8_t r = 0;
    if (x == 0) return 0;
    while (x >>= 1) r++;
    return r;
}

/* ------------------------------------------------------------ ring buffer */
static int16_t hist(const codec_t *st, uint8_t j)
{
    if (j >= BUF_LEN) return st->carry[j - BUF_LEN];
    /* + BUF_LEN before the modulo: (head - j) can go negative and C's % on a
     * negative left operand truncates toward zero, which would index outside
     * the array. Python's % is a floored modulo and would not. This is the
     * single most likely place for the two to silently disagree. */
    return st->buf[(uint8_t)((st->head + BUF_LEN - (j % BUF_LEN)) % BUF_LEN)];
}

#if !SYNC_SAFE
/* Only the legacy path overwrites the ring with true values; under sync_safe
 * the ring holds reconstructions permanently, which is the point. */
static void set_hist(codec_t *st, uint8_t j, int16_t v)
{
    st->buf[(uint8_t)((st->head + BUF_LEN - (j % BUF_LEN)) % BUF_LEN)] = v;
}
#endif

static void push(codec_t *st, int16_t u)
{
    st->head = (uint8_t)((st->head + 1) % BUF_LEN);
    st->buf[st->head] = u;
}

/* ------------------------------------------------------------- prediction */
static void build_phi(const codec_t *st, int32_t *phi)
{
    uint8_t i;
    for (i = 0; i < N_TAPS; i++) phi[i] = hist(st, (uint8_t)(TAPS[i] - 1));
    /* The bias regressor is a CONSTANT, deliberately not 1 << W_Q. At Q20 that
     * would be 1048576, a hundred times the load magnitude, so phi_bias^2 would
     * dominate P and the NLMS normaliser would drive every lag update to zero
     * -- reintroducing the exact defect Q20 exists to fix. */
    phi[N_TAPS] = BIAS_INPUT;
}

static int16_t predict_from(const codec_t *st, const int32_t *phi)
{
    /* int64 accumulator. int32 is provably safe only under the extra
     * preconditions |w| <= 16384 and |u| <= 20000
     * (6 * 16384 * 20000 = 1.97e9 < 2.147e9), which the shipped U_CLAMP of
     * 32000 already violates. On AVR the 64-bit accumulate costs ~150 cycles
     * per hour; at one inference per 3600 s that is free. */
    int64_t acc = 0;
    uint8_t i;
    for (i = 0; i < N_W; i++) acc += (int64_t)st->w[i] * (int64_t)phi[i];
    return (int16_t)ec_clamp(ec_round_shift(acc, W_Q), U_MIN, U_CLAMP);
}

int16_t codec_predict(const codec_t *st)
{
    int32_t phi[N_W];
    build_phi(st, phi);
    return predict_from(st, phi);
}

/* ---------------------------------------------------------------- lifecyle */
void codec_init(codec_t *st, uint16_t eps, int16_t u0, uint32_t t_start)
{
    uint8_t i;
    memset(st, 0, sizeof(*st));
    for (i = 0; i < BUF_LEN; i++) st->buf[i] = u0;
    for (i = 0; i < 24; i++) st->day_true[i] = u0;
    for (i = 0; i < N_W; i++) st->w[i] = W_INIT_ARR[i];
    for (i = 0; i < N_CARRY_MAX; i++) st->carry[i] = u0;
    st->head = 0;
    st->eps = eps;
    st->day_tx = 0;
    st->t = t_start;
    st->t_start = t_start;
    /* With W_INIT = {4096, 0, ...} this is exactly u0. */
    st->u_hat = codec_predict(st);
}

static uint16_t eps_effective(const codec_t *st)
{
    uint32_t d;
    if (st->eps == EPS_NEVER) return EPS_NEVER;
    if (st->n_tx_today >= N_MAX_TX) {
        d = (uint32_t)st->eps * 2u;
        return (uint16_t)(d > 0xFFFEu ? 0xFFFEu : d);
    }
    return st->eps;
}

void codec_step(codec_t *st, int16_t u, bool clamped, step_result_t *out)
{
    int32_t e, ae, s_new;
    uint16_t eps_eff;
    bool tx, event;
    int16_t u_recon;

    e = (int32_t)u - (int32_t)st->u_hat;
    ae = e < 0 ? -e : e;

    /* s[t] is the UPDATED EWMA: blueprint 5.1 step 2 updates before the
     * threshold tests read it. */
    s_new = st->s + ec_sar32(ae - st->s, S_SHIFT);
    st->s = (int16_t)ec_clamp(s_new, 0, 32767);

    eps_eff = eps_effective(st);
    tx = (eps_eff != EPS_NEVER) && (ae > (int32_t)eps_eff);

    /* EVENT is suppressed for the first EVENT_WARMUP hours: s starts at 0, so
     * every early hour with a non-zero residual would otherwise be flagged. */
    event = (st->n_seen >= EVENT_WARMUP) && (ae > (int32_t)K_EVENT * st->s);

    /* THE LINE (blueprint 5.1 step 3). On a silent hour push the PREDICTION.
     * Pushing u here would desynchronise the two predictors within one hour
     * and nothing in the packet stream would reveal it. */
    u_recon = tx ? u : st->u_hat;
    push(st, u_recon);
    {
        uint8_t slot = (uint8_t)(st->t % 24u);
        st->day_true[slot] = u;        /* staged for the RESYNC payload */
        if (tx) st->day_tx |= (uint32_t)1 << slot;
        else    st->day_tx &= ~((uint32_t)1 << slot);
    }

    if (tx) st->n_tx_today++;
    st->n_seen++;
    st->t++;
    st->u_hat = codec_predict(st);     /* blueprint 5.1 step 5 */

    if (out) {
        out->e = (int16_t)e;
        out->u_hat = (int16_t)(u - e); /* the u_hat this hour was scored against */
        out->u_recon = u_recon;
        out->eps_eff = eps_eff;
        out->tx = tx;
        out->event = event;
        out->clamped = clamped;
    }
}

/* ----------------------------------------------------------- day boundary */
bool codec_is_boundary(const codec_t *st, uint32_t t)
{
    return (t % 24u == 0u) && ((t - st->t_start) >= 24u);
}

void codec_new_day(codec_t *st) { st->n_tx_today = 0; st->day_tx = 0; }

void codec_resync_payload(const codec_t *st, int16_t *out24)
{
    uint8_t j;
    for (j = 0; j < 24; j++) out24[j] = st->day_true[j];
}

/* The transmit bitmap. Three bytes on the wire, and it is what lets the hub
 * replay the node's day exactly after any loss (revision R4). */
uint32_t codec_resync_bitmap(const codec_t *st) { return st->day_tx; }

static void nlms_update(codec_t *st, const int32_t *phi, int32_t e)
{
    int64_t P = 0, prod;
    int32_t d;
    uint8_t i, b, sh;
    int64_t nw;

    for (i = 0; i < N_W; i++) P += (int64_t)phi[i] * (int64_t)phi[i];
    b = ec_msb_index((uint64_t)P);

    /* sh = b - W_Q + K_MU, giving an effective mu of 2^-K_MU. The blueprint
     * writes `b - K_MU`, which corresponds to mu = 2^-10 and, at realistic
     * magnitudes, makes every update truncate to 0 (or to -1 for negative
     * products): the filter would never adapt and would drift downward one LSB
     * at a time. See the deviation note in edge/codec.py. */
    {
        int32_t v = (int32_t)b - (int32_t)W_Q + (int32_t)K_MU;
        sh = (uint8_t)ec_clamp(v, 0, 31);
    }

    /* Always round half-up. `sar` floored, applying a uniform -0.5 LSB bias to
     * every tap on every update; that is structural and accumulates. */
    for (i = 0; i < N_W; i++) {
        prod = (int64_t)e * (int64_t)phi[i];
        d = ec_round_shift(prod, sh);
        nw = (int64_t)st->w[i] + (int64_t)d;
        if (nw >  W_CLAMP) nw =  W_CLAMP;
        if (nw < -W_CLAMP) nw = -W_CLAMP;
        st->w[i] = (int32_t)nw;
    }
}

void codec_day_boundary(codec_t *st, const int16_t *true_24)
{
    int32_t phi[N_W];
    uint8_t j, i, off_x, off;
    int32_t e;
    int16_t target;

#if !SYNC_SAFE
    /* legacy: the ring's last 24 entries become the truth, and this whole
     * routine is conditional on the gateway having ACKed the RESYNC that
     * carried them -- which is the two-generals problem revision R4 removes. */
    for (j = 0; j < 24; j++) set_hist(st, j, true_24[23 - j]);
#else
    (void)true_24;              /* sync_safe updates on the reconstruction */
#endif

#if ADAPT
    /* Replay the day. Update j is hour x = D-24+j, whose earliest tap is hour
     * x - MAX_TAP, i.e. offset 23-j+MAX_TAP. At j == 0 that can exceed what the
     * ring addresses, which is what `carry` covers. */
    for (j = 0; j < 24; j++) {
        off_x = (uint8_t)(23 - j);
        for (i = 0; i < N_TAPS; i++) {
            off = (uint8_t)(off_x + TAPS[i]);
            phi[i] = hist(st, off);
        }
        phi[N_TAPS] = BIAS_INPUT;      /* NOT 1 << W_Q; see build_phi */
#if SYNC_SAFE
        target = hist(st, off_x);
#else
        target = true_24[j];
#endif
        e = (int32_t)target - (int32_t)predict_from(st, phi);
        nlms_update(st, phi, e);
    }
#else
    (void)phi; (void)off_x; (void)off; (void)e; (void)target;
#endif

    /* Refresh the carry for the NEXT boundary. N_CARRY is 0 whenever
     * BUF_LEN >= 24 + MAX_TAP, which is the shipped configuration, so this loop
     * compiles away entirely; the #if keeps -Wtype-limits quiet about the
     * always-false comparison rather than suppressing the warning. */
#if N_CARRY > 0
    for (i = 0; i < N_CARRY; i++)
        st->carry[i] = hist(st, (uint8_t)(BUF_LEN - 24 + i));
#endif
    codec_new_day(st);
    /* RE-PREDICT: u_hat for hour D was computed at the end of hour D-1 from the
     * pre-update weights. */
    st->u_hat = codec_predict(st);
}

/* ------------------------------------------------------------- snapshot */
static uint8_t *put16(uint8_t *p, uint16_t v)
{
    *p++ = (uint8_t)(v & 0xFF);
    *p++ = (uint8_t)(v >> 8);
    return p;
}

static uint8_t *put32(uint8_t *p, uint32_t v)
{
    p = put16(p, (uint16_t)(v & 0xFFFF));
    return put16(p, (uint16_t)(v >> 16));
}

/* Byte-for-byte identical to CodecState.snapshot(), little-endian, format
 * "<48hBB6hhhHBBIih". This is what test_host.c diffs against the golden
 * vectors, so the field order here is a contract, not a convenience. */
void codec_snapshot(const codec_t *st, uint8_t *out)
{
    uint8_t i;
    uint8_t *p = out;
    for (i = 0; i < BUF_LEN; i++) p = put16(p, (uint16_t)st->buf[i]);
    *p++ = st->head;
    *p++ = (uint8_t)BUF_LEN;
    for (i = 0; i < N_W; i++) p = put32(p, (uint32_t)st->w[i]);
    p = put16(p, (uint16_t)st->u_hat);
    p = put16(p, (uint16_t)st->s);
    p = put16(p, st->eps);
    *p++ = st->n_tx_today;
    *p++ = (uint8_t)(st->n_seen > 255u ? 255u : st->n_seen);
    p = put32(p, st->t);
    p = put32(p, 0u);                  /* reserved, matches the Python format */
    p = put16(p, (uint16_t)(N_CARRY ? st->carry[0] : 0));
}
