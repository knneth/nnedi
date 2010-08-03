CFLAGS=-O2 -g -Wall -std=gnu99 -msse2 -mfpmath=sse -fomit-frame-pointer
CXXFLAGS=-O2 -g -Wall
LDFLAGS=-L/usr/x11r6/lib

upscale: upscale.o nnedi.o tables.o
	$(CXX) -o $@ $+ $(LDFLAGS) -lpng -ljpeg -lz -lX11 -lpthread

%.o: %.asm
	yasm -f win32 -DPREFIX -o $@ $+
