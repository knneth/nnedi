OPTFLAGS=-O2 -g -Wall -msse -mfpmath=sse -fomit-frame-pointer -Wno-sign-compare -DARCH_X86=1 -DARCH_X86_64=1 -DSYS_LINUX=1
CFLAGS=$(OPTFLAGS) -std=gnu99
CXXFLAGS=$(OPTFLAGS)

style:
	astyle --style=otbs --min-conditional-indent=0 --break-blocks \
		--pad-oper --pad-header --unpad-paren --add-brackets \
		--align-pointer=name --indent-preprocessor \
		--indent-labels "$$@" \
		--indent=spaces=2 *.c *.h



upscale: upscale.o libnnedi.a
	$(CXX) -o $@ $+ $(LDFLAGS) -lpng -lz -lpthread

libnnedi.a: cpu.o cpu-a.o nnedi.o nnedi-a.o nnedi_cimg.o tables.o
	$(AR) rc $@ $+

%.o: %.asm
	yasm -f elf64 -DARCH_X86_64=1 -o $@ $<
	strip -x $@

nnedi.o: nnedi_asm.c
nnedi-a.o: x86inc.asm

.PHONY: clean
clean:
	rm -f upscale libnnedi.a *.o
