#define cimg_use_png
#define cimg_display 0
#include <CImg.h>
extern "C" {
#include "nnedi.h"
}
using namespace cimg_library;

static void upscale(nnedi_t *ctx, CImg<uint8_t> &dst, CImg<uint8_t> &src)
{
    cimg_forC(src, c)
        nnedi_upscale_2x(ctx, dst.data(0,0,0,c), src.data(0,0,0,c), src._width, src._height, dst._width, src._width);
}

int main(int argc, char **argv)
{
    int nns = 4;
    int align = 0;
    if(argc > 1 && sscanf(argv[1], "-%d", &nns))
        argc--, argv++;
    if(argc > 1 && !strcmp(argv[1], "--align"))
        argc--, argv++, align=1;
    if(argc != 3) {
        printf("usage: upscale [-#] [--align] in.png out.png\n");
        return 1;
    }

    nnedi_t *ctx = nnedi_config(nns, 0);
    CImg<uint8_t> src(argv[1]);
    CImg<uint8_t> d2(src._width*2, src._height*2, 1, src._spectrum);
    upscale(ctx, d2, src);

    if(align) {
        d2.rotate(180);
        CImg<uint8_t> d4(d2._width*2, d2._height*2, 1, d2._spectrum);
        upscale(ctx, d4, d2);
        cimg_forXYC(d2,x,y,c)
            d2(x,y,0,c) = d4(d4._width-x*2-1, d4._height-y*2-1, 0,c);
    }

    d2.save(argv[2]);
    free(ctx);
    return 0;
}
