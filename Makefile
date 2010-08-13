CFLAGS=-O2 -g -Wall -std=gnu99 -msse2 -mfpmath=sse -fomit-frame-pointer
CXXFLAGS=-O2 -g -Wall
LDFLAGS=-L/usr/x11r6/lib

upscale: upscale.o nnedi.o nnedi-a.o tables.o
	$(CXX) -o $@ $+ $(LDFLAGS) -lpng -lz

%.o: %.asm
	yasm -f elf64 -DARCH_X86_64 -o $@ $<

nnedi-a.o: x86inc.asm nnedi-a.h

nnedi-a.h: nnedi.c
	grep '#define NNS' nnedi.c | sed -e 's/#/%/' > nnedi-a.h
