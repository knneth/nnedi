#define cimg_use_png
#define cimg_display 0
#include <CImg.h>
extern "C" {
#include "nnedi.h"
#include <bench.h>
}
using namespace cimg_library;

int main(int argc, char **argv)
{
    if(argc != 3) {
        printf("usage: upscale in.png out.png\n");
        return 1;
    }
    CImg<uint8_t> src(argv[1]);
    CImg<uint8_t> dst(src._width*2, src._height*2, 1, src._spectrum);
    uint64_t t0 = read_time();
    cimg_forC(src, c)
        upscale_2x(dst.data(0,0,0,c), src.data(0,0,0,c), src._width, src._height, dst._width, src._width);
    uint64_t t1 = read_time();
    printf("%d Mcycles\n", (int)((t1-t0)/1000000));
    dst.save(argv[2]);
    return 0;
}
