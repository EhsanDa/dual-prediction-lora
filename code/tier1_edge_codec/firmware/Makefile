# firmware/Makefile
#
#   make test    build and run the host equivalence test (needs gcc only)
#   make avr     build for the ATmega328P and print the resource table
#   make regen   regenerate edge_config.h and golden_vectors.h from Python
#
# `make avr` needs avr-gcc. It is not required to develop or to validate the
# codec -- `make test` does that on the host -- but it IS required to produce
# the resource table in the write-up, and blueprint 10.4 says those numbers are
# measured and committed, never estimated.

CC      ?= gcc
CFLAGS  ?= -std=c99 -Wall -Wextra -Werror -O2

AVR_CC     ?= avr-gcc
AVR_SIZE   ?= avr-size
MCU        ?= atmega328p
F_CPU      ?= 8000000UL
AVR_CFLAGS ?= -std=c99 -Wall -Wextra -Os -mmcu=$(MCU) -DF_CPU=$(F_CPU) \
              -ffunction-sections -fdata-sections

PY      ?= python
REPO    := ..

.PHONY: test avr regen clean

test: test_host
	./test_host

test_host: edge_codec.c test_host.c edge_codec.h edge_config.h golden_vectors.h
	$(CC) $(CFLAGS) -o $@ edge_codec.c test_host.c

regen:
	cd $(REPO) && $(PY) -m export.gen_header --data_dir synth_data --client 0 \
		--alpha 1.0 --out firmware/edge_config.h
	cd $(REPO) && $(PY) -m export.gen_golden --out firmware/golden_vectors.h

avr: edge_codec.avr.o
	@echo "--- ATmega328P resource table (codec only, no radio, no app) ---"
	$(AVR_SIZE) edge_codec.avr.o
	@echo
	@$(AVR_SIZE) -A edge_codec.avr.o
	@echo
	@$(AVR_SIZE) -A edge_codec.avr.o | awk '\
	  $$1 ~ /^\.text\./   { code += $$2 } \
	  $$1 ~ /^\.rodata\./ { ro   += $$2 } \
	  END { printf "codec code   sum(.text.*)   = %d B\n", code; \
	        printf "const tables sum(.rodata.*) = %d B  (SRAM too -- see README)\n", ro; \
	        printf "codec Flash  total          = %d B\n", code + ro }'
	@echo
	@echo "text = codec code + .rodata initialisers = the codec's Flash cost."
	@echo "data/bss read 0 here because codec_t is caller-allocated (main.ino),"
	@echo "and because -fdata-sections emits .rodata.* which only becomes SRAM"
	@echo "at link time -- avr-ld places .rodata in the data region. Measure the"
	@echo "state separately; see the recipe in firmware/README.md."
	@echo "Do NOT use avr-size -C on this object: it sums only the literal"
	@echo ".text/.data/.bss names and therefore reports 0 with -ffunction-sections."

edge_codec.avr.o: edge_codec.c edge_codec.h edge_config.h
	$(AVR_CC) $(AVR_CFLAGS) -c edge_codec.c -o $@

clean:
	rm -f test_host *.o
