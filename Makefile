CFLAGS=-O2 -g -Wall -std=gnu99
CXXFLAGS=-O2 -g -Wall
LDFLAGS=-L/usr/x11r6/lib

upscale: upscale.o nnedi.o nnedi3.dll
	$(CXX) -o $@ $+ $(LDFLAGS) -lpng -ljpeg -lz -lX11

%.o: %.asm
	yasm -f win32 -DPREFIX -o $@ $+
