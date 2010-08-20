#define cimg_use_png
#define cimg_display 0
#include <CImg.h>
extern "C" {
#include "nnedi.h"
}
using namespace cimg_library;

int main(int argc, char **argv)
{
    int nns = 0;
    if(argc > 1 && sscanf(argv[1], "-%d", &nns))
        argc--, argv++;
    if(argc != 3) {
        printf("usage: upscale [-#] in.png out.png\n");
        return 1;
    }
    nnedi_config(nns);
    CImg<uint8_t> src(argv[1]);
    CImg<uint8_t> dst(src._width*2, src._height*2, 1, src._spectrum);
    cimg_forC(src, c)
        nnedi_upscale_2x(dst.data(0,0,0,c), src.data(0,0,0,c), src._width, src._height, dst._width, src._width);
    dst.save(argv[2]);
    return 0;
}
