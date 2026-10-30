# firmware/node/Makefile -- the deployed node application, on real hardware.
#
#   make boards      which board is on which port (shared with ../hil)
#   make node        build + link the application, print avr-size
#   make node-flash  flash it            (BOARD=1|2)
#   make node-run    replay the dataset trace and diff against golden_vectors.h
#   make radio       build the radio bench image
#   make band-scan   find which band the modules are actually built for
#   make link-test   board 1 transmits real frames, board 2 receives and decodes
#   make toa         measure time-on-air against edge/protocol.py's model
#
# F_CPU IS 16 MHz HERE. ../Makefile uses 8 MHz, which is the documented
# condition of the resource table in ../README.md, and the two must not be
# confused. `make -C ../hil sizecheck` proves the codec's own size is
# unaffected by the clock.

MCU        ?= atmega328p
F_CPU      ?= 16000000UL
BOARD      ?= 1
PEER       ?= 2
PORT       ?= $(shell ../hil/board_scan $(BOARD) 2>/dev/null)
PEER_PORT  ?= $(shell ../hil/board_scan $(PEER) 2>/dev/null)

AVR_CC     ?= avr-gcc
AVR_CXX    ?= avr-g++
AVR_SIZE   ?= avr-size
AVR_OBJCOPY?= avr-objcopy
AVRDUDE    ?= avrdude
CC         ?= gcc

AVRDUDE_FLAGS := -c arduino -p $(MCU) -P $(PORT) -b 115200

COMMON     := -Wall -Wextra -Os -mmcu=$(MCU) -DF_CPU=$(F_CPU) \
              -DBOARD_ID=$(BOARD) -ffunction-sections -fdata-sections \
              -I.. -I. -I../hil
AVR_CFLAGS ?= -std=c99 $(COMMON)
AVR_CXXFLAGS ?= -std=c++11 -fno-exceptions -fno-threadsafe-statics $(COMMON)
AVR_LDFLAGS?= -mmcu=$(MCU) -Wl,--gc-sections
HOST_CFLAGS?= -std=c99 -D_DEFAULT_SOURCE -Wall -Wextra -Werror -O2 -I.. -I. -I../hil

.PHONY: all boards node node-flash node-run node-deploy radio radio-flash \
        band-scan link-test toa regen driver-size clean

all: node radio

boards:
	$(MAKE) -C ../hil boards

regen:
	cd ../.. && python -m export.gen_meter_trace

# ---- the application ------------------------------------------------------

.board: FORCE
	@[ -f $@ ] && [ "$$(cat $@)" = "$(BOARD)" ] || echo $(BOARD) > $@
FORCE:

DEPS := node_config.h node_protocol.h sx127x.h meter_trace.h \
        ../edge_codec.h ../edge_config.h .board

main.o:          main.ino $(DEPS);      $(AVR_CXX) $(AVR_CXXFLAGS) -x c++ -c $< -o $@
node_boot.o:     node_boot.cpp $(DEPS); $(AVR_CXX) $(AVR_CXXFLAGS) -c $< -o $@
node_hw.o:       node_hw.c $(DEPS);     $(AVR_CC)  $(AVR_CFLAGS)  -c $< -o $@
node_protocol.o: node_protocol.c $(DEPS); $(AVR_CC) $(AVR_CFLAGS) -c $< -o $@
sx127x.o:        sx127x.c $(DEPS);      $(AVR_CC)  $(AVR_CFLAGS)  -c $< -o $@
nuart.o:         ../hil/hil_uart.c $(DEPS); $(AVR_CC) $(AVR_CFLAGS) -c $< -o $@
edge_codec.node.o: ../edge_codec.c ../edge_codec.h ../edge_config.h .board
	$(AVR_CC) $(AVR_CFLAGS) -c $< -o $@

NODE_OBJ := main.o node_boot.o node_hw.o node_protocol.o sx127x.o nuart.o \
            edge_codec.node.o

node.elf: $(NODE_OBJ)
	$(AVR_CXX) $(AVR_LDFLAGS) $^ -o $@

node.hex: node.elf
	$(AVR_OBJCOPY) -O ihex -R .eeprom $< $@

node: node.hex
	@echo "--- the NODE APPLICATION on $(MCU) at $(F_CPU), board $(BOARD) ---"
	@$(AVR_SIZE) -C --mcu=$(MCU) node.elf
	@echo
	@echo "THIS is the linked deployed application: codec + protocol + radio"
	@echo "driver + scheduling. It is what ../README.md's two totals rows name."
	@echo "The radio driver's own size is printed by 'make driver-size'."

# THE DEPLOYED BUILD. The bench build carries UART banners, the per-hour
# report and a 384-hour replay loop that a deployed node has no use for, and
# quoting its size as "the application" would overstate it. This target strips
# them so ../README.md's two totals rows can be filled with a figure that means
# what the row says. Read the caveat that target prints.
node-deploy:
	@$(MAKE) --no-print-directory clean-obj
	@$(MAKE) --no-print-directory node.elf 	   AVR_CFLAGS="-std=c99 -DNODE_DEPLOY $(COMMON)" 	   AVR_CXXFLAGS="-std=c++11 -fno-exceptions -fno-threadsafe-statics -DNODE_DEPLOY $(COMMON)" >/dev/null
	@echo "--- the DEPLOYED node application (bench instrumentation removed) ---"
	@$(AVR_SIZE) -C --mcu=$(MCU) node.elf
	@echo
	@echo "Still contained in this figure: the dataset replay stands in for a"
	@echo "meter interface, because there is no meter and there is not meant to"
	@echo "be. A production build swaps meter_trace.h and meter_read_kwh() for a"
	@echo "real sensor driver; that difference is the one thing these rows still"
	@echo "cannot account for, and it is stated rather than guessed."
	@$(MAKE) --no-print-directory clean-obj

.PHONY: clean-obj
clean-obj:
	@rm -f *.o node.elf node.hex

.PHONY: driver-size
driver-size: sx127x.o node_protocol.o
	@echo "--- transceiver driver, measured ---"
	@$(AVR_SIZE) -A sx127x.o | awk '/^\.(text|rodata)\./{s+=$$2} END{printf "sx127x.c   .text+.rodata = %d B\n", s+0}'
	@$(AVR_SIZE) -A node_protocol.o | awk '/^\.(text|rodata)\./{s+=$$2} END{printf "node_protocol.c           = %d B\n", s+0}'
	@echo "The paper's [P094] estimates 'a minimal transceiver driver at about"
	@echo "150 bytes'. Compare, and report the difference to the author."

node-flash: node.hex
	@test -n "$(PORT)" || { echo "board $(BOARD) not found -- make boards"; exit 1; }
	$(AVRDUDE) $(AVRDUDE_FLAGS) -D -U flash:w:node.hex:i

node_host: node_host.c node_expected.h ../edge_codec.h ../edge_config.h
	$(CC) $(HOST_CFLAGS) -o $@ node_host.c

node-run: node_host
	@test -n "$(PORT)" || { echo "board $(BOARD) not found -- make boards"; exit 1; }
	./node_host $(PORT)

# ---- the radio bench ------------------------------------------------------

radio_device.o: radio_device.c $(DEPS); $(AVR_CC) $(AVR_CFLAGS) -c $< -o $@

radio.elf: radio_device.o sx127x.o node_protocol.o nuart.o
	$(AVR_CC) $(AVR_LDFLAGS) $^ -o $@

radio.hex: radio.elf
	$(AVR_OBJCOPY) -O ihex -R .eeprom $< $@

radio: radio.hex
	@$(AVR_SIZE) -C --mcu=$(MCU) radio.elf

radio-flash: radio.hex
	@test -n "$(PORT)" || { echo "board $(BOARD) not found -- make boards"; exit 1; }
	$(AVRDUDE) $(AVRDUDE_FLAGS) -D -U flash:w:radio.hex:i

# -lm: the airtime model needs pow/ceil/fabs. It reimplements
# edge/protocol.py::time_on_air so a measurement is compared against the model
# rather than against a remembered number.
radio_host: radio_host.c
	$(CC) $(HOST_CFLAGS) -o $@ radio_host.c -lm

band-scan: radio_host
	./radio_host --scan --tx $(shell ../hil/board_scan 1) --rx $(shell ../hil/board_scan 2)

link-test: radio_host
	./radio_host --link --tx $(shell ../hil/board_scan 1) --rx $(shell ../hil/board_scan 2)

toa: radio_host
	./radio_host --toa --tx $(shell ../hil/board_scan 1)

clean:
	rm -f *.o *.elf *.hex node_host radio_host .board
