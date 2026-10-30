/* hil_proto.h -- the host<->device wire protocol, in one place.
 *
 * Included by BOTH hil_device.c (AVR) and hil_host.c (host), so the two cannot
 * drift. If you change anything here, bump HIL_PROTO: the host refuses to run
 * against a device reporting a different number, which turns a confusing
 * byte-mismatch into a clear error message.
 *
 * ASCII, line-oriented, '\n'-terminated, and every reply begins with the
 * letter of the command that produced it. That last property is the whole
 * resynchronisation strategy: on anything unexpected, read to the next newline
 * and check the leading letter. No escaping, no endianness contract, no CRC
 * layer -- and you can drive the board by hand with `cat` while debugging.
 *
 * Lines beginning '#' are banners or comments and are ignored by the host.
 *
 *   V                     -> V <proto> <board_id> <sizeof_codec_t>
 *                              <snapshot_bytes> <n_w> <buf_len> <sync_safe>
 *                              <feature_spec_hash>
 *   I <eps> <u0> <t0>     -> I OK
 *   P                     -> P <u_hat>
 *   T <t>                 -> T <0|1>                  codec_is_boundary
 *   B <v0> ... <v23>      -> B OK                     codec_day_boundary
 *   B -                   -> B OK                     ... with true_24 = NULL
 *   S <u> <clamped>       -> S <e> <u_hat> <u_recon> <eps_eff> <tx> <event>
 *                              <clamped>
 *   N                     -> N OK                     codec_new_day
 *   R                     -> R <bitmap> <v0> ... <v23>
 *   Q                     -> Q <hex>                  2*SNAPSHOT_BYTES chars
 *   M                     -> M <bytes>                unused (painted) stack,
 *                                                     i.e. worst-case headroom
 *   <anything else>       -> E <reason>
 */
#ifndef HIL_PROTO_H
#define HIL_PROTO_H

#define HIL_PROTO 2

/* Long enough for "B" + 24 signed 16-bit decimals + separators, with room. */
#define HIL_LINE_MAX 200

#define HIL_BAUD 115200UL

/* BOARD_ID identifies WHICH PHYSICAL BOARD this image is running on, and it
 * exists because the two Unos on the bench report the IDENTICAL USB serial
 * number (24238313635351910130). They are distinct devices, but nothing in
 * their USB descriptors separates them: /dev/serial/by-id/ can hold only one
 * symlink for the pair, and ttyACM0/ttyACM1 are handed out in enumeration
 * order, so they swap on replug.
 *
 * So the port name is not an identity and must never be treated as one. The
 * id is baked in at flash time (`make hil-flash BOARD=2`) and reported in the
 * banner and the V reply; the host finds a board by ASKING each port who it
 * is. See docs/reference/hardware-bench.md.
 *
 * It is deliberately NOT edge_config.h's NODE_ID, which is a protocol field
 * belonging to the codec's commissioning. Conflating the two would make a
 * bench cabling detail leak into packet contents. */
#ifndef BOARD_ID
#define BOARD_ID 1
#endif

#endif /* HIL_PROTO_H */
