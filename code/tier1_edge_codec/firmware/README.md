# firmware/ — the C port and the resource table

The Python in `edge/codec.py` is the specification. This directory is a
transliteration of it, plus the machinery that proves the transliteration is
faithful.

```
make regen    # edge_config.h + golden_vectors.h, from Python
make test     # build with gcc, replay the golden vectors, diff byte for byte
make avr      # build for the ATmega328P and print avr-size (needs avr-gcc)
```

`make test` must pass before anything is flashed. It is not a smoke test: a C
codec that is *nearly* right does not degrade gracefully here. It desynchronises
from the NanoPi, and the reconstruction then becomes wrong in a way that no
packet reveals, because the packets themselves still decode cleanly.

## What the equivalence test covers

`main.ino` now lives in [`node/`](node/README.md) alongside the radio driver and
the wire-format encoders it needs; `firmware/` itself holds the codec, the
golden vectors and the host equivalence test.

`golden_vectors.h` is a 384-hour trace recorded from the Python reference,
deliberately containing every branch that matters. The generator asserts each
one is present and refuses to emit a trace that is missing any:

| Branch | In the current trace |
|---|---|
| exceptions | 163 |
| silent hours (the push-`u_hat` path) | 221, longest run 16 |
| day boundaries with the resync applied | 14 |
| day boundary with the resync **lost** | 1 |
| `EVENT` flag firing after warm-up | 5 |
| quantiser clamp | 1 |
| lost-ACK branch | **not present, and cannot be** — see below |

`SYNC_SAFE 0` (legacy) is now the shipped default, because it measured both
better and cheaper at every α; `SYNC_SAFE 1` stays compiled as a robustness
option against ACK loss and is exercised by `tests/test_codec.py`.
`gen_golden.py` adds or drops the lost-ACK branch from its assertion list
automatically depending on which mode it is generating for, so whichever build
is compiled is the one that is covered.

`test_host.c` compares `u_hat`, `e`, the flag byte at **every** hour and the
full 128-byte snapshot every 24 hours. Comparing only outputs would miss a
state divergence that has not surfaced in an output yet — which is exactly the
divergence that hurts later.

Current status: **byte-identical, 384/384 hours, 17/17 snapshots** — on the
host under gcc, and since 2026-09-16 **on a real ATmega328P** as well, via
[`hil/`](hil/README.md). `make -C hil hil-flash hil-run`.

## Resource table

Blueprint §10.4 says measure, do not estimate, and report your measurements
even when they disagree with the predicted figures. Several do — the state is
2.4× the predicted size, and the Flash is well under it.

Measured 2026-09-14 with `avr-gcc` 5.4.0 / `avr-size` (binutils 2.26),
`-Os -mmcu=atmega328p`, card P3-01 step 1.

| Quantity | Blueprint target | Measured | Source |
|---|---|---|---|
| Codec state (`sizeof(codec_t)`) | ~200 B (48-tap Q12 design) | **486 B on AVR**; 488 B on x86-64 | `avr-nm -S` / `test_host` |
| Codec code, AVR `.text.*` | — | **3 010 B** | `make avr` |
| Codec `.rodata` (`TAPS`, `W_INIT_ARR`) | — | **34 B** — Flash *and* SRAM, see below | `make avr` |
| Codec Flash footprint | ~13 KB Flash *total* | **3 044 B** = 9.3 % of 32 KB | `make avr`, berkeley `text` |
| Codec-attributable SRAM | — | **520 B** = 25.4 % of 2 048 | 486 + 34, linked probe |
| Total SRAM incl. radio + stack | under ~750 B of 2 048 | **≈ 879 B** = 544 B static + 335 B measured peak stack | `node/`, `avr-size -C` + painted stack |
| Flash total | ~13 KB of 32 KB | **8 742 B** = 26.7 % | `node/`, `make -C node node-deploy` |
| Cycles per `codec_step` + `codec_predict` | well under 0.1 ms | **not measured** | needs avr-gdb or a scope |

**The two totals were `UNKNOWN` until 2026-09-28, and this is what changed.**
For most of this project `main.ino` was a skeleton with **seven** unimplemented
hardware symbols — two of them not even declared, so it failed at compile before
the linker was reached — and there was no SX127x driver. Stubbing them would
have produced a number that was not the thing the rows name, so the rows stayed
blank. That was the right call and it is recorded in
`docs/findings/2026-09-16-board-bringup.md` §3.

They are now filled because the application genuinely exists:
[`node/`](node/README.md) implements all seven against a real SX1278, links,
runs on an ATmega328P, and reproduces the Python reference byte for byte over
384 hours — the whole application, not just the codec. The figures above come
from `make -C node node-deploy`, which strips the bench instrumentation so the
number means what the row says, plus a stack high-water measured by painting.

**Two qualifications travel with those two rows**, and they belong next to them:
the 335 B stack is measured on the *bench* build (identical deepest path, but
the deployed build has no UART to report it), and the dataset replay stands in
for a meter interface — there is no meter and there is not meant to be — so a
production build differs by whatever a real sensor driver costs. Flash lands
well under the ~13 kB prediction; **SRAM lands about 130 B over the ~750 B
one**, and still fits at 43 % of the part. See
`docs/findings/2026-09-28-node-and-radio-bringup.md` §2.

### `hil/` links too, and its size is **not** these rows.

Since 2026-09-16 there is a board on the bench and
[`hil/`](hil/README.md) — the hardware-in-the-loop harness — which **does**
link, and therefore produces an `.elf`, a `.hex` and real `avr-size` totals.
**Those totals are a different measurement of a different program and must
never be copied into the two `UNKNOWN` rows above.**

The harness is the codec plus a polled UART and an ASCII command responder. The
rows name the *deployed application*: `main.ino` plus a radio driver plus the
meter read and the hourly sleep path. The two programs share `edge_codec.o` and
nothing else, and the harness implements **none** of the seven externs — which
is enforced by `make guard`, not by this sentence.

| Quantity | Measured 2026-09-16 | Source |
|---|---|---|
| HIL image Flash (`.text` + `.data`) | **5 212 B** = 15.9 % of 32 KB | `avr-size -C hil/hil.elf` |
| HIL image static SRAM (`.data` + `.bss`) | **1 390 B** = 67.9 % of 2 048 | `avr-size -C hil/hil.elf` |
| HIL worst-case unused stack | **487 B** | painted stack, `M` command, after `codec_day_boundary` + snapshot |
| — of which codec | 520 B | unchanged, from the table above |
| Deployed application, Flash total | **8 742 B** | `node/`, not this image |
| Deployed application, SRAM incl. radio + stack | **≈ 879 B** | `node/`, not this image |

The last two rows are repeated on purpose: this is the table where somebody is
most likely to reach for the wrong number. The harness is **not** the
application — it has no radio, no meter and no sleep path, and it implements
none of the seven externs, which `make -C hil guard` enforces.

The harness is built at `F_CPU = 16000000UL` for the Uno on the bench, against
the 8 MHz the table above was recorded at, into a separate object in a separate
directory. `make -C hil sizecheck` proves the codec's own size is unmoved by
the clock — `edge_codec.c` includes only `<string.h>` and `edge_codec.h`, no
`F_CPU` and no `<avr/*>`:

```
codec Flash @ F_CPU=8000000UL  = 3044 B
codec Flash @ F_CPU=16000000UL = 3044 B
```

If that check ever fails, stop and report it.

**The x86-64 `.text` figure was never a proxy, and now there is proof.** The
README previously recorded ~1.9 KB for the codec object on x86-64 `-Os`. The AVR
build is **3 044 B — about 60 % larger.** Do not quote host figures for Flash.

### Measuring the state under the AVR ABI

`sizeof(codec_t)` cannot be read out of `make avr`: the struct is
caller-allocated (`static codec_t g_codec` in `main.ino`), so it appears in no
section of `edge_codec.avr.o`. To measure it:

```bash
printf '#include "edge_codec.h"\ncodec_t probe;\n' > /tmp/probe.c
avr-gcc -std=c99 -Os -mmcu=atmega328p -DF_CPU=8000000UL -fno-common \
        -I. -c /tmp/probe.c -o /tmp/probe.o
avr-size -A /tmp/probe.o          # .bss = 486
```

**486 B, confirming the field sum in `docs/findings/2026-09-14-firmware-sram.md`
and not the 482 B that `eval/tap_ablation.py::node_state_bytes()` computes.**
That discrepancy is card P1-07 and is deliberately not fixed here.

### `.rodata` is SRAM on this part, and nothing counts it

`TAPS` (6 B) and `W_INIT_ARR` (28 B) are `static const`, and avr-ld places
`.rodata` in the **data** region: the AVR cannot load from Flash with ordinary
instructions, so the linker copies both tables into SRAM at startup. A linked
probe image shows `.data = 34`, `.bss = 486`, total **520 B**.

`node_state_bytes()` models `codec_t` only and so misses all 34 B — a gap
**8.5× larger than the +4 B** already recorded against it. See
`docs/findings/2026-09-14-avr-const-sram.md`.

### Where the state actually goes

```
buf[192]      384 B   ring of RECONSTRUCTED values (24 + 168 = 192 slots)
day_true[24]   48 B   TRUE values of the current day, staged for the RESYNC
w[7]           28 B   Q20 weights, int32
carry[1]        2 B   N_CARRY is 0 for these taps, but edge_config.h pins
                      N_CARRY_MAX 1 so the array is never zero-length
day_tx          4 B   transmit bitmap for the current day
t, n_seen,     12 B   counters
  t_start
u_hat, s, eps   6 B
head,           2 B
  n_tx_today
              ----
              486 B   sizeof(codec_t) on AVR -- MEASURED, see the table above
TAPS +         34 B   static const, and avr-ld puts .rodata in SRAM
  W_INIT_ARR
              ----
              520 B   of 2048 B, with the radio driver at ~150 B
```

**This block previously read `carry[0] 0 B` and totalled 482 B**, mirroring
`eval/tap_ablation.py::node_state_bytes()`. Both the 0 and the 482 were wrong;
the AVR build settles it at 486 + 34. The Python model is still uncorrected on
purpose — card P1-07, because it feeds two frozen artifacts.

**Where the growth went, and why each byte was bought:**

| Change | Cost | Bought |
|---|---|---|
| ring 48 → 192 slots | +288 B | the lag-168 tap, which the ablation earned |
| weights Q12 int16 → Q20 int32 | +16 B | a *usable* step size — see below |
| `day_tx` bitmap | +4 B | exact hub-side replay after packet loss |

### The Q20 resolution argument

This is the engineering point of the whole edge model, and it belongs in the
paper rather than in a repo note. At Q12, with loads around 1e4 device units,
the NLMS update `(e·φ) >> (b − Q_W + k_mu)` evaluates to:

| `k_mu` | μ | shift | Δw, Q12 LSBs | Δw, Q20 LSBs |
|---|---|---|---|---|
| 2 | 0.25 | 19 / 11 | **2** | 488 |
| 3 | 0.125 | 20 / 12 | 1 | 244 |
| 4 | 0.0625 | 21 / 13 | **0** | 122 |
| 5 | 0.031 | 22 / 14 | **0** | 61 |

The entire useful range of step sizes lived between "2 LSBs" and "zero". Any
attempt to reduce μ to a sane value quantised the update to nothing and froze
the filter at its initialisation — so the predictor had a choice between
too-fast and not-at-all, and it lost to a frozen persistence predictor. Sixteen
bytes fixed it. `tests/test_revisions.py::test_q12_update_quantises_to_nothing_and_q20_does_not`
reproduces the table exactly.

**One trap the change creates.** The bias regressor must NOT be `1 << W_Q`. At
Q12 that was 4096, comparable to load magnitudes; at Q20 it is 1 048 576, a
hundred times the loads, so `φ_bias²` dominates `P` by four orders of magnitude
and the NLMS normaliser drives every *lag* update back to zero — reintroducing
the exact defect Q20 exists to fix. `BIAS_INPUT` is therefore a separate
constant fixed at 4096. This bit the C port during this revision: `build_phi`
was updated and the day-boundary regressor was not, and `make test` caught it
as an 878-byte snapshot divergence at the first boundary.

`day_tx` is new in revision R4 and it is what makes synchronisation
unconditional: it rides in the daily RESYNC (3 bytes on the wire) and lets the
hub replay the node's day exactly after any amount of exception loss. Four bytes
of SRAM and sixteen of daily airtime bought the removal of a two-generals
failure, which is a good trade.

**This is far above the blueprint's §4.4 table, and every byte is accounted
for above.**
§4.4 states that the 48-slot ring does double duty as the daily resync payload,
so no second buffer is needed. It cannot:

1. The ring must hold **reconstructed** values, or the mirror cannot follow it.
   On a silent hour that is `u_hat`, not the truth. The RESYNC payload has to be
   the truth. Hence `day_true[24]`, +48 B.
2. The daily update replays 24 hours whose taps reach back `max_tap` more, i.e.
   `24 + max_tap` hours of history, from a `buf_len` ring. Hence
   `carry[n_carry]` with `n_carry = max(0, 24 + max_tap - buf_len)` — one
   sample for the default taps, and zero for the 168-lag variant, which sizes
   its ring at 192 precisely so the carry vanishes.

Still comfortably inside the 2 KB part, and worth stating plainly rather than
letting the prose keep a number the code contradicts.

## Known gaps, carried into the write-up

- **`w_init` is a fitted vector, not a constant.** `edge_config.h` must be
  generated with `--pretrained edge_pretrained.json`, or the node ships the
  persistence initialisation, which is ablation row A rather than the proposed
  model. `test_host.c` refuses to run if the golden vectors and the config
  disagree about `w_init`.
- **`SYNC_SAFE` is a compile-time constant.** The two modes have different
  arithmetic, so a node flashed with one cannot talk to a hub running the
  other. The generated `edge_config.h` records which. The RESYNC carries its
  repair fields (transmit bitmap, weights; 32 B) in BOTH builds -- legacy's
  gateway does not read them, and the reported airtime counts them
  (corrected 2026-10-13; this line used to say they were absent in legacy).
- **Clock drift.** The Nano's resonator drifts minutes per day and the ε
  guarantee depends on both sides agreeing which hour a sample belongs to.
  Samples are indexed by sequence number and the phase is corrected in the
  daily downlink; a DS3231 RTC is recommended for deployment. Unmodelled in
  simulation.
- **Radio.** Raw point-to-point SX127x, not LoRaWAN. LMIC needs ~1.5–2 KB SRAM
  and ~30 KB Flash and does not fit alongside the application on a 2 KB / 32 KB
  part, and with one gateway per building it buys nothing. This constraint
  invalidates a lot of otherwise-plausible designs.
- **`main.ino` is an UNGATED node.** It is no longer a skeleton (since
  2026-09-28 it compiles, links and runs — `node/`), but its day boundary calls
  `codec_day_boundary` unconditionally while this build is legacy
  (`SYNC_SAFE 0`). The paper's protocol, the simulator and these golden vectors
  update only on ACK. Harmless on a lossless channel; under loss it is a
  different protocol — `docs/findings/2026-09-28-frame-loss-settled.md` §2, §8
  and `docs/tasks.md` P3-08 (later work, not the conference revision). Nothing
  in `main.ino` does arithmetic — that all lives in `edge_codec.c`, where the
  equivalence test can reach it.
