"""
codec.py  --  the Tier-1 reference implementation. Blueprint 4 and 5.1.

This module IS the specification. edge/codec_batch.py, firmware/edge_codec.c
and every test are validated against it. Two constraints follow and are not
negotiable:

  * NO NUMPY, NO FLOAT. Plain Python ints only, so the file transliterates to C
    line by line.
  * Every arithmetic step goes through edge/fixedpoint.py, so a width overflow
    raises here instead of wrapping silently in C.

THE ONE LINE THAT MATTERS
=========================
`step()` pushes `u_hat` into the ring buffer on a silent hour, not `u`. If it
pushed `u`, the Arduino's buffer would contain values the NanoPi has never seen
and the two predictors would diverge within one hour -- the reconstruction
would then be wrong in a way that no packet reveals, because the packets
themselves would still decode cleanly. tests/test_codec.py builds a
deliberately broken variant that pushes `u` and asserts that the
synchronisation test catches it; if that test ever passes, the synchronisation
test has stopped testing anything.

WHAT IS ADAPTIVE AND WHEN
=========================
Weights are FROZEN for the whole of each day. At a day boundary both sides
replay the previous 24 hours through 24 sequential NLMS updates. Within a day,
`u_hat` is a function of frozen weights and a buffer both sides agree on, so
there is nothing left to desynchronise. It also makes the update unbiased: it
sees all 24 hours, not only the ones that happened to trigger a transmission.

SYNC-SAFE MODE (revision R4; NOT the default since the final revision)
=====================================================================
The original design updated the weights on the previous day's TRUE values,
which the node could only apply once the gateway had ACKed the RESYNC that
carried them. That is a two-generals problem, and the measured consequence was
real: a lost ACK left the mirror resynced and the node not, and the protocol
could not self-correct within the day.

`sync_safe` removes it rather than documenting it:

  * The weight update runs on the RECONSTRUCTED values already in the ring --
    values both sides provably hold, each within eps of truth. It is therefore
    UNCONDITIONAL on both sides and needs no packet and no ACK.
  * The ring is never overwritten with true values. It holds reconstructions,
    always, on both sides.
  * The RESYNC still carries the day's true values (for the hub's stored
    history, which is what Y_EXACT is built from), and now also carries a
    24-bit transmit bitmap and the node's post-update weight vector.

With those, the mirror can REPLAY the day exactly: restore the buffer to its
day-start snapshot, then for each hour push the true value where the bitmap
says the node transmitted and its own prediction where it did not. That
reproduces the node's buffer whatever was lost in between, and adopting the
weights from the same packet repairs the weights. So:

    ONE delivered RESYNC repairs the shared state completely, whatever else
    was lost, and no ACK is required for correctness at all.

The ACK/downlink is now purely configuration (new eps, clock correction), which
also means the node no longer has to open an RX window it cannot afford. The
cost is that the filter adapts on values within eps of truth rather than on
truth; that is small and is measured directly by running with
`sync_safe=False`, which restores the original behaviour.

THREE DEVIATIONS FROM THE BLUEPRINT, RECORDED HERE
==================================================
(1) THE NLMS SHIFT. Blueprint 4.6 writes `sh = clamp(b - k_mu, 0, 31)` and
    annotates k_mu with "effective mu ~ 0.25". Those two statements are
    inconsistent, and the literal formula is provably inert:

        w update = sar(e * phi[i], sh)

    Real-valued NLMS is w_real += mu * e * phi / P with w_real = w / 2**w_q, so
    the integer step must be w += 2**w_q * mu * e * phi / P. Matching against
    e * phi >> sh gives 2**sh = P * 2**(-w_q) / mu, i.e.

        sh = msb(P) - w_q + log2(1/mu)      ->   sh = b - w_q + k_mu

    The literal `b - k_mu` corresponds to mu = 2**-10, and at realistic
    magnitudes (loads ~1e4 device units, P ~ 5e8, e ~ 1e2) it makes
    `e * phi >> sh` equal to 0 for positive products and -1 for negative ones:
    the filter would never adapt, and would instead drift monotonically
    downward one LSB per negative term per update. That is not a slow filter,
    it is a broken one, so the annotation is the intended semantics and the
    expression is the typo. tests/test_codec.py pins this empirically via the
    NLMS a-posteriori identity, which the literal reading fails by three orders
    of magnitude.

(2) THE PREDICTION FLOOR. Blueprint 4.5 clamps u_hat to [-u_clamp, +u_clamp].
    But the EXCEPTION packet carries e = u - u_hat as an int16 (blueprint 5.3),
    and with u in [0, 32000] and u_hat in [-32000, 32000] the residual ranges
    over [-32000, 64000] -- which does not fit. A saturated residual breaks
    u = u_hat + e, and with it the epsilon guarantee, silently. Since the
    metered quantity is non-negative by construction, u_hat is clamped to
    [u_min, u_clamp] with u_min = 0, which makes e int16 by arithmetic rather
    than by hope.

(3) THE WEIGHT-UPDATE ROUNDING (revision B1). Blueprint 4.6 specifies `sar`,
    which floors, applying a uniform -0.5 LSB bias to every tap on every
    update. That bias is structural, not data-dependent: it does not average
    out, it accumulates. Measured on the 60-day fixture in
    tests/test_codec.py, converged mean |e| is 147 with `sar` and 100 with
    round-half-up -- a 32% reduction for one added constant before the shift.
    `round_w_update` therefore DEFAULTS TO TRUE. Setting it False restores the
    blueprint's arithmetic and reproduces the 147.
"""

import struct
from dataclasses import dataclass, replace

from edge.fixedpoint import (abs_i, clamp, i16, i32, i64, msb_index,
                             round_shift, sar)

EPS_INF = None          # alpha = inf: never transmit an exception
EPS_INF_WIRE = 0xFFFF   # how EPS_INF is encoded in snapshot() and the downlink
EPS_MAX_FINITE = 0xFFFE


@dataclass(frozen=True)
class CodecParams:
    """Blueprint 4.3, as revised. Every field is load-bearing."""
    # THE 168 TAP IS IN. The blueprint rejected it on a division-of-labour
    # argument (DLinear's 336-hour window already spans two weekly cycles). The
    # R8 ablation measured otherwise, so the argument loses to the data and the
    # tap set carries the weekly lag. buf_len = 24 + 168 makes n_carry zero.
    taps: tuple = (1, 2, 3, 24, 25, 168)
    buf_len: int = 192
    # Q20 IN INT32, NOT Q12 IN INT16. See the RESOLUTION note in the module
    # docstring: at Q12 the whole useful range of step sizes lived between
    # "2 LSBs" and "zero", so the filter had a choice between too-fast and
    # not-at-all. Cost of the fix: 12 B of SRAM.
    w_q: int = 20
    w_init: tuple = (1 << 20, 0, 0, 0, 0, 0, 0)   # persistence; see PRETRAINED
    w_clamp: int = 4 << 20            # still |W| <= 4.0
    # The bias regressor is a CONSTANT, deliberately NOT 1 << w_q. Tying it to
    # w_q was harmless at Q12 (4096, comparable to the loads) but at Q20 it
    # would be 1048576, a hundred times the load magnitude, so phi_bias^2 would
    # dominate P by four orders of magnitude and the NLMS normaliser would push
    # every lag update to zero -- reintroducing the exact defect Q20 exists to
    # fix. Holding it at 4096 keeps P ~ 5e8 and the shift near 14, which is
    # where the revision's own resolution arithmetic puts it.
    bias_input_const: int = 4096
    u_clamp: int = 32000
    u_min: int = 0                    # see deviation (2)
    k_mu: int = 5                     # NLMS step: effective mu = 2**-5 = 1/32
    s_shift: int = 5                  # EWMA alpha = 1/32 for the novelty scale
    k_event: int = 4                  # novelty threshold multiplier
    # DAILY EXCEPTION CAP (revision B2). The cap doubles eps for the rest of the
    # day on exhaustion, which turns the alpha axis into a partly-cap axis and
    # weakens the guarantee to <= 2*eps over most of the interesting range. At
    # SF7/SF9 it protects nothing -- aggregate occupancy at 40 nodes sits well
    # under the ETSI line even at alpha = 0. It earns its keep only at SF12
    # against the 30 s/day TTN fair-use budget. So the EXPERIMENT runs with it
    # non-binding (24) and the firmware keeps it as an airtime safety valve;
    # the binding value is reported as a separate robustness row.
    n_max_tx: int = 24
    event_warmup: int = 24            # hours before EVENT may fire
    # The weight update always rounds half-up. The blueprint's `sar` floored,
    # applying a uniform -0.5 LSB bias to every tap on every update; there was
    # never a reason to keep it and the option is gone.
    sync_safe: bool = False           # legacy is the DEFAULT: it measured both
                                      # better and cheaper at every alpha. See
                                      # the module docstring.
    adapt: bool = True                # False freezes the weights at w_init,
                                      # which IS the classic send-on-delta /
                                      # zero-order-hold baseline (revision R7)
    # ---- the resync-weight repair. OFF by default, deliberately.
    #
    # READ docs/findings/2026-09-28-frame-loss-settled.md BEFORE TOUCHING THIS.
    # What it repairs depends on the NODE it is paired with, and the repository
    # got that wrong for two weeks.
    #
    # Effect, in edge/mirror.py's legacy branch: after the 24 updates, replace
    # the mirror's weights with the frame's ("adopt AFTER"). That is correct
    # only if the frame carries POST-update weights, i.e. for an UNGATED node
    # that updates unconditionally and then transmits -- which is what
    # firmware/node/main.ino does. Against that node it turns a permanent
    # weight divergence after one lost RESYNC into a repair at the next
    # delivered one (4 breaches -> 0 in the 2026-09-14 probe).
    #
    # It is WRONG for the gated node the paper reports (update only on ACK):
    # that node can only put its PRE-update weights in the frame, so adopting
    # them after updating undoes the update on every day, and the two sides
    # separate on a lossless channel (tests/test_protocol_matrix.py). For that
    # node the correct order is adopt BEFORE the updates, modelled in
    # eval/protocol_matrix.py as gateway "adopt_before". sim/run_codec.py hides
    # the difference because it encodes the frame after the node has already
    # "heard" its ACK.
    #
    # Under the paper's protocol a lost RESYNC is symmetric and needs no repair;
    # the failure is a lost ACK, which adopt_before repairs within ~24 + 168 h.
    #
    # Default FALSE: no reported number depends on it. Which node/gateway
    # pairing the paper states is an author decision (request A of
    # docs/conference-revision.md); do not flip this default until it is taken.
    #
    # No-op under sync_safe=True, which already adopts the weights (R4).
    adopt_resync_weights: bool = False

    @property
    def n_w(self):
        return len(self.taps) + 1     # + bias

    @property
    def bias_input(self):
        """The bias regressor. See `bias_input_const` for why it is a constant
        rather than 1 << w_q."""
        return self.bias_input_const

    def with_w_init(self, w):
        """A copy carrying a different initialisation -- how the globally
        pre-trained vector of section 3.2 reaches the codec."""
        return replace(self, w_init=tuple(int(x) for x in w))

    @property
    def max_tap(self):
        return max(self.taps)

    @property
    def n_carry(self):
        """Extra samples the daily update needs beyond what the ring addresses.

        The update replays hours [D-24, D-1]; the earliest tap it reads is hour
        D-24-max_tap, i.e. offset 23 + max_tap from D-1. The ring addresses
        offsets 0 .. buf_len-1. Hence n_carry = 24 + max_tap - buf_len, clamped
        at zero. For the shipped taps it is ZERO, because buf_len is sized at
        24 + max_tap precisely so the carry vanishes; it was one sample under
        the earlier buf_len = 48 design and remains non-zero for any tap set
        whose ring is shorter than 24 + max_tap.
        """
        return max(0, 24 + self.max_tap - self.buf_len)

    def __post_init__(self):
        assert self.buf_len > self.max_tap, "ring buffer shorter than the longest tap"
        assert self.buf_len >= 24, "the daily update needs at least a day of ring"
        assert len(self.w_init) == self.n_w
        assert 0 <= self.u_min < self.u_clamp


PARAMS = CodecParams()

# ---- the four ablation configurations of section 4.1 -------------------------
# A. Persistence + deadband: the classic WSN send-on-delta baseline. w_init IS
#    the persistence predictor, so freezing it gives exactly
#    "transmit when |y[t] - y[t-1]| > eps".
PARAMS_SEND_ON_DELTA = CodecParams(adapt=False)
# B. Global fixed: the pre-trained vector, adaptation disabled. Built at runtime
#    by sim/pretrain.py, because the weights come from a fit, not a constant.
# C. Global + NLMS: the proposed model. PARAMS with w_init replaced.
# D. Per-client ridge: the non-deployable CEILING reference. Per-client weights
#    are passed through run_policy's w_init_per_client argument.

# The old Q12 configuration, kept ONLY so the resolution argument in the module
# docstring can be reproduced rather than asserted. Do not ship it.
PARAMS_Q12_LEGACY = CodecParams(taps=(1, 2, 3, 24, 25), buf_len=48, w_q=12,
                                w_init=(4096, 0, 0, 0, 0, 0), w_clamp=16384,
                                k_mu=2)


@dataclass
class StepResult:
    e: int = 0              # u - u_hat, int16 by construction
    tx: bool = False        # exception packet emitted this hour
    event: bool = False     # novelty flag (labels a packet, never fires one)
    clamped: bool = False   # the quantiser saturated on the way in
    u_hat: int = 0          # what was predicted for THIS hour
    u_recon: int = 0        # what both sides now hold for this hour
    eps_eff: int = None     # threshold actually applied (after the daily cap)


class CodecState:
    """One meter's codec. The Arduino runs this; edge/mirror.py runs the same
    object driven only by the packet stream, and the two must have identical
    sync_snapshot() bytes at every hour."""

    def __init__(self, params: CodecParams = PARAMS, eps=EPS_INF, u0: int = 0,
                 t_start: int = 0):
        p = self.p = params
        self.buf = [i16(u0)] * p.buf_len
        self.head = 0                       # index of the most recent sample
        self.w = [i32(x) for x in p.w_init]
        self.eps = eps if eps is EPS_INF else int(eps)
        self.s = 0                          # EWMA of |e|
        self.n_tx_today = 0
        self.n_seen = 0                     # samples consumed since commissioning
        self.t_start = int(t_start)
        self.t = int(t_start)               # absolute hour of the NEXT sample
        self.carry = [i16(u0)] * p.n_carry  # see CodecParams.n_carry
        self.n_clamp = 0
        self.n_tx_total = 0
        self.n_event_total = 0
        # TRUE values of the current day, staged for the RESYNC payload.
        # Blueprint 4.4 says the ring's most recent 24 entries ARE the payload;
        # they are not, because on a silent hour the ring holds u_hat and the
        # payload has to be the truth. 48 bytes, and in sync_safe mode it is
        # needed for FIDELITY only -- the predictor never reads it.
        self.day_true = [i16(u0)] * 24
        # Transmit bitmap for the current day, so the mirror can replay the day
        # exactly after any loss. 24 bits, 3 bytes on the wire.
        self.day_tx = [False] * 24
        # Buffer snapshot at the start of the current day. The NODE does not
        # need it (its buffer is authoritative); the MIRROR replays from it.
        self.buf_at_day_start = list(self.buf)
        self.head_at_day_start = self.head
        self.u_hat = self.predict()         # prediction for hour t_start

    # ------------------------------------------------------------ ring buffer
    def hist(self, j: int) -> int:
        """Value the codec holds for hour (t - 1 - j): hist(0) is the most
        recent sample. Offsets at or beyond buf_len come from `carry`."""
        p = self.p
        if j < p.buf_len:
            return self.buf[(self.head - j) % p.buf_len]
        k = j - p.buf_len
        assert k < p.n_carry, f"ring index {j} beyond ring + carry"
        return self.carry[k]

    def _push(self, u: int) -> None:
        self.head = (self.head + 1) % self.p.buf_len
        self.buf[self.head] = i16(u)

    def _set_hist(self, j: int, u: int) -> None:
        self.buf[(self.head - j) % self.p.buf_len] = i16(u)

    # ------------------------------------------------------------- prediction
    def phi(self):
        """Regressor for the NEXT hour: taps are lags relative to t+1, so a tap
        of 1 reads hist(0). Blueprint 4.5."""
        return [self.hist(tap - 1) for tap in self.p.taps] + [self.p.bias_input]

    def predict(self) -> int:
        return self._predict_from(self.phi())

    def _predict_from(self, phi) -> int:
        p = self.p
        acc = 0
        for i in range(p.n_w):
            # int64 accumulator. An int32 one is provably safe ONLY under the
            # extra preconditions |w| <= 16384 and |u| <= 20000, which the
            # shipped u_clamp already violates. On AVR the 64-bit accumulate
            # costs ~150 cycles per hour, which at one inference per 3600 s is
            # free.
            acc = i64(acc + i64(self.w[i] * phi[i]))
        return clamp(round_shift(acc, p.w_q), p.u_min, p.u_clamp)

    # ------------------------------------------------------------- hourly step
    def eps_effective(self) -> int:
        """Blueprint 4.9. On exhaustion of the daily exception budget, epsilon
        doubles for the remainder of the day -- a rule both sides evaluate from
        state they both hold. It is NOT a hard cap. With the revised default
        n_max_tx = 24 it never binds, so the guarantee is |recon - truth| <= eps
        unconditionally; the binding value is a separate robustness row."""
        if self.eps is EPS_INF:
            return EPS_INF
        if self.n_tx_today >= self.p.n_max_tx:
            return min(self.eps * 2, EPS_MAX_FINITE)
        return self.eps

    def step(self, u: int, clamped: bool = False, force_tx=None) -> StepResult:
        """Consume the reading for hour self.t. Blueprint 5.1 steps 1-3, 5.

        `force_tx` exists for the MIRROR ONLY and is never used by the device.
        The mirror knows from the packet stream whether an exception was sent;
        deciding again from its own copy of `eps` would make its reconstruction
        depend on state it can only infer.
        """
        p = self.p
        # The replay snapshot must sit at the start of a CALENDAR day, because
        # the transmit bitmap and the RESYNC payload are indexed by t % 24. The
        # constructor takes it at commissioning, which is a calendar-day start
        # only when t_start % 24 == 0. For a meter commissioned mid-day the
        # first sync_safe replay therefore started up to 23 hours early and the
        # mirror left the node on a LOSSLESS channel (client 118, t_start 160:
        # 768 of its first 800 hours out of sync). Retaking it at the first
        # midnight fixes that and is a no-op for aligned meters, where this
        # fires only at t_start itself, on the buffer the constructor already
        # captured. Only replay_day() reads the snapshot, so the node's
        # outputs, the legacy path and every snapshot() byte are unchanged.
        # docs/findings/2026-09-28-frame-loss-settled.md, section 5.
        if self.t % 24 == 0 and self.t - self.t_start < 24:
            self.mark_day_start()
        u = i16(u)
        u_hat = self.u_hat

        e = i32(u - u_hat)
        assert -32768 <= e <= 32767, (
            f"residual {e} does not fit the int16 packet field; with u_min=0 "
            f"this is unreachable -- see deviation (2)")
        ae = abs_i(e)

        # s[t] is the UPDATED EWMA, per blueprint 5.1 step 2 ordering.
        self.s = i16(clamp(self.s + sar(ae - self.s, p.s_shift), 0, 32767))

        eps_eff = self.eps_effective()
        tx = False if eps_eff is EPS_INF else (ae > eps_eff)
        if force_tx is not None:
            tx = bool(force_tx)

        # EVENT is suppressed for the first `event_warmup` hours: s starts at 0,
        # so without this every early hour with a non-zero residual is flagged.
        event = (self.n_seen >= p.event_warmup) and (ae > p.k_event * self.s)

        # ---- THE LINE. Silent hour -> push the PREDICTION, not the truth.
        u_recon = u if tx else u_hat
        self._push(u_recon)
        slot = self.t % 24
        self.day_true[slot] = u
        self.day_tx[slot] = bool(tx)

        if tx:
            self.n_tx_today += 1
            self.n_tx_total += 1
        if event:
            self.n_event_total += 1
        if clamped:
            self.n_clamp += 1
        self.n_seen += 1
        self.t += 1
        self.u_hat = self.predict()          # blueprint 5.1 step 5

        return StepResult(e=e, tx=tx, event=event, clamped=clamped, u_hat=u_hat,
                          u_recon=u_recon, eps_eff=eps_eff)

    def apply_exception(self, e: int, clamped: bool = False) -> StepResult:
        """Mirror-side counterpart of `step` for an hour where an EXCEPTION was
        received: u = u_hat + e, exact."""
        u = i16(self.u_hat + i32(e))
        return self.step(u, clamped=clamped, force_tx=True)

    def apply_silent(self) -> StepResult:
        """Mirror-side counterpart for an hour with no packet: fill with u_hat.

        The transmitter's real e (|e| <= eps) is NOT known here, which is why
        the mirror's `s` is an approximation and is never used for
        reconstruction; EVENT arrives in the packet flags.
        """
        return self.step(self.u_hat, force_tx=False)

    # ----------------------------------------------------------- day boundary
    def is_boundary(self, t: int) -> bool:
        """Both sides evaluate this on the hour index alone, so they agree."""
        return t % 24 == 0 and (t - self.t_start) >= 24

    def mark_day_start(self) -> None:
        """Snapshot for the mirror's replay. Called at the END of every day
        boundary, on both sides, so the two snapshots correspond."""
        self.buf_at_day_start = list(self.buf)
        self.head_at_day_start = self.head

    def day_boundary(self, true_24=None) -> None:
        """Blueprint 4.6 / 5.1 step 4, at the START of hour D.

        sync_safe (default): the 24 NLMS updates run on the RECONSTRUCTED
        values already in the ring. `true_24` is ignored -- it is not needed and
        accepting it silently would invite the caller to think it matters. The
        update is therefore unconditional and identical on both sides with no
        packet involved, which is what removes the two-generals problem
        (revision R4).

        legacy (sync_safe=False): the ring's last 24 entries are overwritten
        with `true_24` and the updates run on those, which is the original
        blueprint behaviour, kept so the fidelity cost of sync_safe can be
        measured rather than argued.

        INDEX ARITHMETIC. On entry the ring holds hours [D-buf_len, D-1] and
        hist(j) is hour D-1-j. Update j replays hour x = D-24+j, whose earliest
        tap is hour x-max_tap, i.e. offset 23-j+max_tap. At j = 0 that is
        23+max_tap, which for the default taps is one past what the ring
        addresses -- hence `carry`. See CodecParams.n_carry.
        """
        p = self.p
        if not p.sync_safe:
            # With adaptation disabled there is nothing to update and nothing to
            # overwrite: the boundary is only a budget reset. Requiring the true
            # day here would force every caller of the frozen configurations
            # (ablation rows A, B and D) to carry a payload none of them uses.
            if p.adapt:
                assert true_24 is not None and len(true_24) == 24, (
                    "legacy mode updates on the true day and must be given it")
                for j in range(24):
                    self._set_hist(j, i16(true_24[23 - j]))

        if p.adapt:
            for j in range(24):
                off_x = 23 - j
                phi = [self.hist(off_x + tap) for tap in p.taps] + [p.bias_input]
                target = self.hist(off_x) if p.sync_safe else i16(true_24[j])
                e = i32(target - self._predict_from(phi))
                self._nlms_update(phi, e)

        # Refresh the carry for the NEXT boundary, which needs offsets
        # buf_len .. buf_len+n_carry-1 measured from hour D+23, i.e. offsets
        # buf_len-24 .. buf_len-24+n_carry-1 measured from hour D-1.
        self.carry = [i16(self.hist(p.buf_len - 24 + i)) for i in range(p.n_carry)]
        self.new_day()
        # RE-PREDICT. u_hat for hour D was computed at the end of hour D-1 from
        # the pre-update weights. Leaving it would discard the update for a full
        # hour, and in legacy mode would leave a mirror that had missed an
        # exception predicting from a buffer the resync just corrected.
        self.u_hat = self.predict()
        self.mark_day_start()

    def new_day(self) -> None:
        """Reset the daily exception budget and the day's staging arrays. This
        happens at EVERY day boundary, whether or not any packet arrived: it is
        a local calendar rule evaluated from the hour index alone."""
        self.n_tx_today = 0
        self.day_tx = [False] * 24

    def replay_day(self, true_24, tx_bitmap, weights=None) -> None:
        """MIRROR ONLY (revision R4). Reproduce the node's last 24 hours exactly.

        Restore the buffer to the day-start snapshot, then push the true value
        wherever the node transmitted and this codec's own prediction wherever
        it did not. Because the node did precisely that, and because the weights
        were frozen all day, the result is the node's buffer regardless of how
        many exception packets were lost in between. Adopting `weights` from the
        same RESYNC then repairs the weights, so ONE delivered RESYNC restores
        the shared state completely.
        """
        assert self.p.sync_safe, "replay is the sync_safe repair path"
        assert len(true_24) == 24 and len(tx_bitmap) == 24
        self.buf = list(self.buf_at_day_start)
        self.head = self.head_at_day_start
        u_hat = self.predict()
        for j in range(24):
            v = i16(true_24[j]) if tx_bitmap[j] else u_hat
            self._push(v)
            u_hat = self.predict()
        if weights is not None:
            self.w = [i32(x) for x in weights]
        self.u_hat = self.predict()

    def _nlms_update(self, phi, e: int) -> None:
        """Division-free NLMS. The ATmega328P has no hardware divide, and a
        software divide in the hourly path is both slow and a place for the two
        implementations to disagree; a normalising shift has neither problem.

        See deviations (1) and (3) for the shift and the rounding.
        """
        p = self.p
        P = 0
        for x in phi:
            P = i64(P + i64(x * x))
        b = msb_index(P)
        sh = clamp(b - p.w_q + p.k_mu, 0, 31)
        for i in range(p.n_w):
            # Always round half-up: `sar` floored, biasing every tap by
            # -0.5 LSB on every update, which is structural and accumulates.
            d = round_shift(i64(e * phi[i]), sh)
            self.w[i] = i32(clamp(self.w[i] + d, -p.w_clamp, p.w_clamp))

    def resync_payload(self):
        """The 24 TRUE readings of hours [D-24, D-1], oldest first. Called at
        the start of hour D, where D % 24 == 0, so day_true[j] is hour D-24+j
        and no rotation is needed."""
        return list(self.day_true)

    def resync_bitmap(self):
        """Which of those 24 hours the node transmitted. 3 bytes on the wire,
        and it is what makes the mirror's replay exact."""
        return list(self.day_tx)

    # ------------------------------------------------------------- snapshot
    def _fmt(self):
        p = self.p
        return (f"<{p.buf_len}hBB{p.n_w}ihhHBBIih",
                f"<{p.buf_len}hB{p.n_w}ihHBIh")

    def sync_snapshot(self) -> bytes:
        """The state the two sides MUST agree on, and only that.

        `s`, the novelty EWMA, is deliberately excluded, and this is not a
        convenience: on a silent hour the transmitter sees its real residual
        (|e| <= eps) while the mirror sees exactly 0, so the two EWMAs genuinely
        diverge. That is harmless -- `s` feeds only the EVENT flag, EVENT is
        carried in the packet flags rather than recomputed, and nothing in the
        reconstruction path reads it. Putting it in the synchronisation test
        anyway would produce a failing test that is telling the truth about the
        wrong thing, and the usual response to that is to weaken the test.
        """
        p = self.p
        eps = EPS_INF_WIRE if self.eps is EPS_INF else min(int(self.eps), EPS_MAX_FINITE)
        return struct.pack(self._fmt()[1], *self.buf, self.head, *self.w,
                           self.u_hat, eps, self.n_tx_today,
                           self.t & 0xFFFFFFFF,
                           self.carry[0] if self.carry else 0)

    def canonical(self) -> bytes:
        """State serialised in HISTORY order rather than ring order.

        The scalar codec starts its ring at head 0 when the meter is
        commissioned; the vectorised one advances a single global head for every
        client. Both therefore agree on hist(j) for every j, but their raw
        buffers are ROTATED relative to each other whenever t_start is not a
        multiple of buf_len. That was invisible while buf_len was 48 and the
        fixture onboarded at hour 48; it is not invisible at buf_len 192.

        Rotation is not a difference in state -- nothing reads the buffer except
        through hist() -- so cross-implementation equivalence is checked here,
        and the raw-layout snapshot() is reserved for the C port, which
        transliterates the scalar codec's layout exactly.
        """
        p = self.p
        eps = EPS_INF_WIRE if self.eps is EPS_INF else min(int(self.eps),
                                                           EPS_MAX_FINITE)
        return struct.pack(
            f"<{p.buf_len}h{p.n_w}ihhHBBI",
            *[self.hist(j) for j in range(p.buf_len)], *self.w, self.u_hat,
            self.s, eps, self.n_tx_today, min(self.n_seen, 255) & 0xFF,
            self.t & 0xFFFFFFFF)

    def snapshot(self) -> bytes:
        """Deterministic serialisation of the COMPLETE state, in a fixed byte
        order. This is what the C port has to reproduce byte for byte and what
        the scalar-versus-batch equivalence test compares -- both DEVICE-side,
        so every field is in scope."""
        p = self.p
        eps = EPS_INF_WIRE if self.eps is EPS_INF else min(int(self.eps), EPS_MAX_FINITE)
        return struct.pack(
            self._fmt()[0], *self.buf, self.head, p.buf_len, *self.w, self.u_hat,
            self.s, eps, self.n_tx_today, min(self.n_seen, 255) & 0xFF,
            self.t & 0xFFFFFFFF, 0, self.carry[0] if self.carry else 0)

    def __repr__(self):
        return (f"CodecState(t={self.t}, u_hat={self.u_hat}, s={self.s}, "
                f"eps={self.eps}, w={self.w})")


def warm_start(params: CodecParams, eps, u0: int, t_start: int) -> CodecState:
    """Commissioning. The ring is filled with the first reading, so the first
    prediction is that reading: with w_init = (4096, 0, ...) the prediction is
    round_shift(4096 * hist(0), 12) = hist(0) exactly, which is the blueprint
    9.3 initialisation test. Both sides commission from the same u0."""
    return CodecState(params, eps, u0=u0, t_start=t_start)
