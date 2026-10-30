# firmware/hil/ — hardware in the loop on a real ATmega328P

`../test_host.c` proves `edge_codec.c` matches `edge/codec.py` when gcc
compiles it for x86-64. That is the useful check and it is not the whole claim:
the deployed codec is compiled by **avr-gcc** for an **8-bit** target with
16-bit `int`, different integer promotions, different shift costs and a
compiler two decades older. This directory closes that gap — the same vectors,
the same comparison set, on the silicon the paper names.

Added 2026-09-16, when a board first became available. It implements blueprint
step 6 (`_attic/2026-09-15-blueprints/TIER1_IMPLEMENTATION_BLUEPRINT.md:619`),
which was designed and never built.

```bash
make id            # read the device signature -- writes nothing
make backup        # archive the board's flash BEFORE overwriting it
make hil           # build + link the harness image, print avr-size
make hil-flash     # flash it
make hil-run       # replay 384 hours against it
make probe         # build the peripheral-discovery image
make probe-flash && make probe-run
make restore       # put the board's original flash back
```

## These sizes are NOT the two `UNKNOWN` rows

`../README.md` carries a resource table with two rows reading `UNKNOWN` —
"Total SRAM incl. radio + stack" and "Flash total" — because there has never
been anything to link. **This directory links. That does not fill those rows.**

The harness image is the codec plus a polled UART and an ASCII command
responder. The rows name the **deployed application**: `main.ino` plus an
SX127x driver plus the meter read and the hourly sleep path, none of which
exists in this repository. The two programs share `edge_codec.o` and nothing
else. A UART, a 200-byte line buffer and a 432-byte snapshot staging buffer are
machinery the deployed node will never carry.

This is enforced, not merely asserted. `make hil` depends on `make guard`,
which fails the build if any of `main.ino`'s seven hardware symbols appears in
these sources **or** is defined in `hil.elf`:

```
guard: none of the seven externs is defined. UNKNOWN rows intact.
```

`make sizecheck` is the second guard. `../README.md` records the codec at
3 044 B measured at `F_CPU = 8000000UL`; this directory builds the same source
at 16 MHz for the Uno. The target proves the figure is unmoved rather than
assuming it:

```
codec Flash @ F_CPU=8000000UL  = 3044 B
codec Flash @ F_CPU=16000000UL = 3044 B
identical -- the recorded 3 044 B is unaffected by the 16 MHz build.
```

`F_CPU` is 16 MHz **here** and 8 MHz in `../Makefile`, deliberately. The two
Makefiles are separate and write different object files so that measuring the
bench can never silently restate the committed table.

## Why the host holds the golden vectors

The blueprint sentence says "flash the Arduino with the same golden vectors".
That is not what this does, and the deviation is deliberate.

`../golden_vectors.h` is ≈ 11.5 kB (`GOLDEN_SNAP` alone is 17 × 432 B). Its
arrays are plain `static const`, which avr-gcc places in `.data` — copied into
the part's 2 kB of SRAM at startup. It does not fit. Making it fit means
`PROGMEM` or `__flash`, which means changing the meaning of a file whose first
line reads *"GENERATED … Do not edit"* — if not by editing it, then by
`-Dstatic=...` from the command line, which is the same thing with less
visibility and breaks silently the next time `make regen` runs.

So the host streams the trace instead, one hour per round trip, and the device
holds only `codec_t` plus small buffers. The host program `#include`s the same
`../golden_vectors.h` that `test_host.c` does, unmodified: one fixture, one
source of truth, no new generated artifact. The codec never sees the vectors —
it consumes one `int16_t` per call — so where they are stored is not part of
what is being proved.

It is also not a ceiling: a trace longer than flash costs nothing here.

## The wire protocol

ASCII, line-oriented, `\n`-terminated, and **every reply begins with the letter
of the command that produced it**. That last property is the entire
resynchronisation strategy: on anything unexpected, read to the next newline
and check the leading letter. No escaping, no endianness contract, no CRC
layer — and the whole thing is drivable by hand:

```bash
stty -F /dev/ttyACM0 115200 raw -echo
cat /dev/ttyACM0 &
printf 'V\n' > /dev/ttyACM0
```

Lines beginning `#` are banners and are ignored by the host.

| Host sends | Device replies | Calls |
|---|---|---|
| `V` | `V <proto> <sizeof> <snapbytes> <n_w> <buf_len> <sync_safe> <spec_hash>` | — |
| `I <eps> <u0> <t0>` | `I OK` | `codec_init` |
| `P` | `P <u_hat>` | `codec_predict` |
| `T <t>` | `T <0\|1>` | `codec_is_boundary` |
| `B <v0> … <v23>` | `B OK` | `codec_day_boundary` |
| `B -` | `B OK` | `codec_day_boundary(…, NULL)` |
| `S <u> <clamped>` | `S <e> <u_hat> <u_recon> <eps_eff> <tx> <event> <clamped>` | `codec_step` |
| `N` | `N OK` | `codec_new_day` |
| `R` | `R <bitmap> <v0> … <v23>` | `codec_resync_payload` + `_bitmap` |
| `Q` | `Q <864 hex chars>` | `codec_snapshot` |
| `M` | `M <bytes>` | unused painted stack — see below |
| anything else | `E <reason>` | — |

**The `V` handshake is load-bearing.** The host compares every field against
its own compiled-in headers and refuses to run on a mismatch. Without it, an
image built from a different `edge_config.h` fails as an unexplained byte
difference at an arbitrary hour — which looks exactly like a real equivalence
failure and is not one. The `sizeof(codec_t)` field is *reported*, not
asserted: 486 on AVR against 488 on x86-64 is
`docs/findings/2026-09-14-firmware-sram.md`, and is expected.

**Reset handling.** Opening `/dev/ttyACM0` asserts DTR and resets the board
into Optiboot, which holds the port for about a second. The device prints
`# READY hil 1` on boot; the host flushes, then polls `V` with a ~2.5 s budget,
discarding `#` lines. `HUPCL` is left on, so closing the port resets the board
and **every run starts from the same state** — the run is idempotent by
construction rather than by remembering to power-cycle.

**Lock-step, always.** The host sends one command and waits for one reply. That
is not politeness: the part has a 2-byte hardware RX FIFO and no flow control,
so a second command arriving while the device is inside `codec_step` would be
dropped. Lock-step makes that impossible.

**The host never retries.** On any protocol anomaly it aborts non-zero. After a
device reset the codec state is unrecoverable, so a silent retry could turn a
real fault into a green run — the one thing a harness backing an equivalence
claim must never do.

### Baud

16 MHz with `U2X0 = 1`, so `UBRR = F_CPU/(8·baud) − 1` = 16.36 → 16, giving an
actual 117 647 against a nominal 115 200: **+2.12 %**. The datasheet's
recommended maximum receiver error for 8N1 at 8× oversampling is ±2.5 %, so
this is inside tolerance, and it is the divisor every Arduino ships with.
`hil_uart.c` carries a compile-time assertion that refuses to build if a
different `F_CPU`/baud pair pushes the error past 2.5 % — a guard against
someone setting `F_CPU` back to 8 MHz out of habit and getting silent garbage.

Without `U2X0`, 115 200 at 16 MHz gives −3.55 % and does not work. If framing
errors ever appear, **500000** is the escape hatch: UBRR = 3, exactly 0.00 %
error, and `B500000` is a Linux termios constant (`B250000` is not).

## Stack headroom is measured, not hoped

The image uses about two thirds of the part's SRAM, so the free region between
`.bss` and the stack is painted at entry and the `M` command counts what
survives. Measured 2026-09-16, after the deepest path (`codec_day_boundary`,
the codec's largest function at 858 B) and a full snapshot:

| | |
|---|---|
| static (`.data` + `.bss`) | 1 390 B |
| unused stack at boot | 612 B |
| **worst-case unused stack observed** | **487 B** |

Peak stack use is therefore about 171 B. Re-run `make hil-run` then query `M`
if any of this file's code changes.

## The peripheral probe

`probe_device.c` is bring-up only — not part of the codec, not part of the
reported pipeline. It answers "what is wired to this board?", which the host
cannot answer: anything on the Arduino's pins sits behind the MCU and is
invisible to USB.

Read `docs/reference/hardware-bench.md` for what it can and cannot establish.
The short version: an I²C ACK and an SX127x `RegVersion` are positive evidence;
an analog voltage is not an identity, and an MQ-series gas sensor can never be
identified by any probe because it is a resistor.

**It is not passive.** An I²C address sweep and an SPI NSS assertion are writes
to a shared bus. No device register is ever written, but do not run it against
a live deployment.

## This is not part of `repro/`

The harness needs hardware, so it can never be part of the reproduction
pipeline. It writes nothing to `results/` and nothing to `results_frozen/`, and
no number it produces is a paper artifact without a decision to make it one.
Card **P3-01** steps 3–5 (energy, time-on-air, in-situ frame loss) remain open:
this directory is the instrument those steps will be built on, not the
measurement.

**The board is an Uno R3; the paper's node is a Nano.** Both are ATmega328P at
16 MHz, so code size, SRAM and arithmetic transfer exactly — but the Uno adds a
16U2 USB bridge, a linear regulator and a power LED. **Any energy figure taken
on this board is the Uno's, not the node's**, and must not be written into
P3-01 step 3.

## Files

| File | Purpose |
|---|---|
| `hil_proto.h` | The wire protocol in one place. Included by both sides so they cannot drift. |
| `hil_uart.h/.c` | Polled USART0. No interrupts — determinism over throughput. Compile-time baud-error assertion. |
| `hil_device.c` | The AVR responder. Parses, dispatches, formats. **Contains no arithmetic** — the same invariant `main.ino` holds. |
| `hil_host.c` | The host driver. Includes `../golden_vectors.h` and diffs byte-for-byte. |
| `probe_device.c` | Peripheral discovery: I²C sweep, SX127x probe, analog and digital survey. |
| `probe_host.c` | Reads the probe's report. Sends nothing — the port-open reset is the trigger. |

`hil_host.c` repeats the comparison logic of `../test_host.c` rather than
sharing it. Factoring out a common `compare.c` would churn the exact artifact
backing P3-01 step 2's "DONE 2026-09-14, byte-identical" claim, for no
behavioural gain. If you change one, read the other.
