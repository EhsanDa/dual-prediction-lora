# firmware/hil/Makefile -- hardware in the loop on a real ATmega328P.
#
#   make ports        list serial devices and their USB IDs
#   make id           read the device signature (touches nothing)
#   make backup       read the board's flash to _attic/ BEFORE overwriting it
#   make hil          build + link the HIL image, print avr-size
#   make hil-flash    flash it
#   make hil-run      build the host driver and replay 384 hours against it
#   make probe        build + link the peripheral-discovery image
#   make probe-flash  flash it
#   make probe-run    read its report
#   make restore      put the archived original flash back
#
# F_CPU IS 16 MHz HERE AND THAT IS DELIBERATE. The parent firmware/Makefile
# uses 8 MHz, which is the documented condition of the committed resource
# table in ../README.md. The board on the bench is an Uno R3 at 16 MHz, and
# the UART divisor depends on it. Keeping the two Makefiles separate means
# measuring the bench cannot silently restate the table. The codec's own size
# is unaffected either way -- it contains no timing code.

MCU        ?= atmega328p
F_CPU      ?= 16000000UL

# WHICH BOARD. The two Unos on the bench report the IDENTICAL USB serial, so
# ttyACM0/ttyACM1 are not identities and can swap on replug. BOARD is baked
# into the image at flash time and every target resolves it to a port by ASKING
# each board who it is (./board_scan). PORT= still overrides, for the one case
# that needs it: flashing a board that does not yet carry an id.
BOARD      ?= 1
PORT       ?= $(shell ./board_scan $(BOARD) 2>/dev/null)
PROGRAMMER ?= arduino
UPLOAD_BAUD?= 115200

AVR_CC     ?= avr-gcc
AVR_SIZE   ?= avr-size
AVR_OBJCOPY?= avr-objcopy
AVRDUDE    ?= avrdude
CC         ?= gcc

AVR_CFLAGS ?= -std=c99 -Wall -Wextra -Os -mmcu=$(MCU) -DF_CPU=$(F_CPU) \
              -DBOARD_ID=$(BOARD) -ffunction-sections -fdata-sections -I.. -I.
AVR_LDFLAGS?= -mmcu=$(MCU) -Wl,--gc-sections

# _DEFAULT_SOURCE, not -std=gnu99: cfmakeraw() and CRTSCTS are BSD extensions
# that strict C99 hides. Widening the feature-test macro exposes exactly those
# while keeping the same -std=c99 -Werror the rest of the repository builds under.
HOST_CFLAGS?= -std=c99 -D_DEFAULT_SOURCE -Wall -Wextra -Werror -O2 -I.. -I.

# Each board's original flash has its own archive. PLAYBOOK rule 3.
ATTIC_1    := ../../../../_attic/2026-09-16-board-flash-backup
ATTIC_2    := ../../../../_attic/2026-09-28-board2-flash-backup
ATTIC      := $(ATTIC_$(BOARD))
AVRDUDE_FLAGS := -c $(PROGRAMMER) -p $(MCU) -P $(PORT) -b $(UPLOAD_BAUD)

EXTERNS := meter_read_kwh radio_send radio_wait_downlink \
           sleep_until_next_hour crc16_hw encode_resync apply_downlink

.PHONY: all ports boards id backup restore hil hil-flash hil-run guard \
        sizecheck probe probe-flash probe-run clean

all: hil probe

# ---- information, writes nothing -----------------------------------------

ports:
	@for d in /dev/ttyACM* /dev/ttyUSB*; do \
	  [ -e "$$d" ] || continue; \
	  echo "$$d"; \
	  udevadm info -q property -n "$$d" 2>/dev/null \
	    | grep -E '^(ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL_SHORT|ID_MODEL)=' \
	    | sed 's/^/    /' || true; \
	done; \
	command -v lsusb >/dev/null && lsusb | grep -i arduino || true

# WHO IS ON WHICH PORT. The only sound way to tell the two boards apart.
boards: board_scan
	@./board_scan

board_scan: board_scan.c hil_proto.h
	$(CC) $(HOST_CFLAGS) -o $@ board_scan.c

id:
	$(AVRDUDE) $(AVRDUDE_FLAGS) -n

# ---- backup and restore ---------------------------------------------------
# PLAYBOOK rule 3: nothing is deleted. The board's original flash is archived
# before anything overwrites it. `backup` refuses to clobber an existing
# archive -- the first read is the one worth keeping.

backup:
	@mkdir -p $(ATTIC)
	@if [ -f $(ATTIC)/flash-before.hex ]; then \
	  echo "$(ATTIC)/flash-before.hex already exists -- refusing to overwrite."; \
	  echo "That file is the board as it was found. Move it aside by hand if"; \
	  echo "you really mean to re-read a board you have already flashed."; \
	  exit 1; \
	fi
	$(AVRDUDE) $(AVRDUDE_FLAGS) -U flash:r:$(ATTIC)/flash-before.hex:i
	@md5sum $(ATTIC)/flash-before.hex

restore:
	@test -f $(ATTIC)/flash-before.hex || \
	  { echo "no archived flash at $(ATTIC)"; exit 1; }
	$(AVRDUDE) $(AVRDUDE_FLAGS) -D -U flash:w:$(ATTIC)/flash-before.hex:i

# ---- the HIL image --------------------------------------------------------

HIL_SRC := hil_device.c hil_uart.c ../edge_codec.c
HIL_OBJ := $(notdir $(HIL_SRC:.c=.hil.o))

# .board records which BOARD the current objects were built for, so changing
# BOARD= forces a rebuild instead of flashing a stale id -- the exact mistake
# board_scan exists to catch.
.board: FORCE
	@[ -f $@ ] && [ "$$(cat $@)" = "$(BOARD)" ] || echo $(BOARD) > $@
FORCE:

hil_device.hil.o: hil_device.c ../edge_codec.h ../edge_config.h hil_proto.h hil_uart.h .board
	$(AVR_CC) $(AVR_CFLAGS) -c $< -o $@
hil_uart.hil.o: hil_uart.c hil_uart.h hil_proto.h .board
	$(AVR_CC) $(AVR_CFLAGS) -c $< -o $@
edge_codec.hil.o: ../edge_codec.c ../edge_codec.h ../edge_config.h
	$(AVR_CC) $(AVR_CFLAGS) -c $< -o $@

hil.elf: $(HIL_OBJ)
	$(AVR_CC) $(AVR_LDFLAGS) $^ -o $@

hil.hex: hil.elf
	$(AVR_OBJCOPY) -O ihex -R .eeprom $< $@

# ---- the two guards -------------------------------------------------------
# These exist because the most valuable numbers in this repository are the ones
# this directory could most easily corrupt.

# GUARD 1. The harness must implement NONE of main.ino's seven hardware
# symbols. If it ever did, its linked size would start to look like the thing
# the two UNKNOWN rows in ../README.md name, and someone would eventually copy
# it there. A comment would not stop that; a failing build does.
guard: hil.elf
	@for s in $(EXTERNS); do 	  if grep -qw "$$s" *.c *.h 2>/dev/null; then 	    echo "GUARD FAILED: '$$s' appears in hil/ sources."; 	    echo "The harness must implement none of main.ino's seven externs."; 	    exit 1; fi; 	done
	@if avr-nm -g --defined-only hil.elf 2>/dev/null 	   | grep -wE '$(shell echo $(EXTERNS) | tr " " "|")'; then 	  echo "GUARD FAILED: hil.elf defines one of main.ino's externs."; exit 1; fi
	@echo "guard: none of the seven externs is defined. UNKNOWN rows intact."

# GUARD 2. ../README.md records the codec at 3 044 B, measured at F_CPU = 8 MHz.
# This directory builds the same source at 16 MHz. edge_codec.c includes only
# <string.h> and edge_codec.h -- no F_CPU, no <avr/*> -- so the figure should be
# untouched. "Should be" is not good enough for a number in a paper, so this
# target proves it instead.
sizecheck: ../edge_codec.avr.o edge_codec.hil.o
	@a=$$($(AVR_SIZE) -A ../edge_codec.avr.o | awk '/^\.(text|rodata)\./{s+=$$2} END{print s+0}'); 	 b=$$($(AVR_SIZE) -A edge_codec.hil.o    | awk '/^\.(text|rodata)\./{s+=$$2} END{print s+0}'); 	 echo "codec Flash @ F_CPU=8000000UL  = $$a B"; 	 echo "codec Flash @ F_CPU=$(F_CPU) = $$b B"; 	 if [ "$$a" != "$$b" ]; then 	   echo "STOP. The codec's size depends on F_CPU. ../README.md's 3 044 B is"; 	   echo "recorded at 8 MHz. Report this to the author; do not adjust it."; 	   exit 1; fi; 	 echo "identical -- the recorded 3 044 B is unaffected by the 16 MHz build."

../edge_codec.avr.o:
	$(MAKE) -C .. avr >/dev/null

hil: hil.hex guard
	@echo "--- HIL image on $(MCU) at $(F_CPU) ---"
	@$(AVR_SIZE) -C --mcu=$(MCU) hil.elf
	@echo
	@echo "This is the codec PLUS a UART command responder. It is NOT the"
	@echo "figure the two UNKNOWN rows in ../README.md name -- those name the"
	@echo "deployed application, with a radio driver and the sleep path, which"
	@echo "does not exist in this repository. Those rows stay UNKNOWN."

hil-flash: hil.hex
	$(AVRDUDE) $(AVRDUDE_FLAGS) -D -U flash:w:hil.hex:i

hil_host: hil_host.c ../golden_vectors.h ../edge_codec.h ../edge_config.h hil_proto.h
	$(CC) $(HOST_CFLAGS) -o $@ hil_host.c

hil-run: hil_host board_scan
	@test -n "$(PORT)" || { echo "board $(BOARD) not found -- run 'make boards'"; exit 1; }
	./hil_host $(PORT)

# ---- the peripheral probe -------------------------------------------------

probe_device.probe.o: probe_device.c hil_uart.h hil_proto.h .board
	$(AVR_CC) $(AVR_CFLAGS) -c $< -o $@
hil_uart.probe.o: hil_uart.c hil_uart.h hil_proto.h .board
	$(AVR_CC) $(AVR_CFLAGS) -c $< -o $@

probe.elf: probe_device.probe.o hil_uart.probe.o
	$(AVR_CC) $(AVR_LDFLAGS) $^ -o $@

probe.hex: probe.elf
	$(AVR_OBJCOPY) -O ihex -R .eeprom $< $@

probe: probe.hex
	@echo "--- probe image on $(MCU) at $(F_CPU) ---"
	@$(AVR_SIZE) -C --mcu=$(MCU) probe.elf

probe-flash: probe.hex
	$(AVRDUDE) $(AVRDUDE_FLAGS) -D -U flash:w:probe.hex:i

probe_host: probe_host.c
	$(CC) $(HOST_CFLAGS) -o $@ probe_host.c

probe-run: probe_host board_scan
	@test -n "$(PORT)" || { echo "board $(BOARD) not found -- run 'make boards'"; exit 1; }
	./probe_host $(PORT)

clean:
	rm -f *.o *.elf *.hex hil_host probe_host board_scan .board
