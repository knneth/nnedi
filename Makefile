CFLAGS=-m32 -O2 -g -Wall -Wno-unused-function -std=gnu99 -msse2 -mfpmath=sse -fomit-frame-pointer
CXXFLAGS=-m32 -O2 -g -Wall
LDFLAGS=-L/usr/x11r6/lib

upscale: upscale.o nnedi.o nnedi-a.o tables.o
	$(CXX) -m32 -o $@ $+ $(LDFLAGS) -lpng -lz

%.o: %.asm
	yasm -f elf -o $@ $<
	strip -x $@

nnedi.o: nnedi_dsp.c

nnedi-a.o: x86inc.asm nnedi-a.h

nnedi-a.h: nnedi.c
	grep '#define NNS' nnedi.c | sed -e 's/#/%/' > nnedi-a.h
