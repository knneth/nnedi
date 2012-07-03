OPTFLAGS=-O2 -g -Wall -msse -mfpmath=sse -fomit-frame-pointer -DARCH_X86 -Wno-sign-compare
CFLAGS=$(OPTFLAGS) -std=gnu99
CXXFLAGS=$(OPTFLAGS)

upscale: upscale.o libnnedi.a
	$(CXX) -o $@ $+ $(LDFLAGS) -lpng -lz -lpthread

libnnedi.a: nnedi.o nnedi-a.o nnedi_cimg.o tables.o
	$(AR) rc $@ $+

%.o: %.asm
	yasm -f elf64 -DARCH_X86_64=1 -o $@ $<
	strip -x $@

nnedi.o: nnedi_asm.c
nnedi-a.o: x86inc.asm

.PHONY: clean
clean:
	rm -f upscale libnnedi.a *.o
