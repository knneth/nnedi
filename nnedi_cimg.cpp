#define cimg_display 0
#include <CImg.h>
#include "nnedi.h"
using namespace cimg_library;

CImg<uint8_t> nnedi_upscale_2x_cimg(CImg<uint8_t> src, int nns, int threads)
{
    nnedi_t *ctx = nnedi_config(nns, threads);
    CImg<uint8_t> dst(src._width*2, src._height*2, src._depth, src._spectrum);
    cimg_forZC(src,z,c)
        nnedi_upscale_2x(ctx, dst.data(0,0,z,c), src.data(0,0,z,c), src._width, src._height, dst._width, src._width);
    free(ctx);
    return dst;
}
