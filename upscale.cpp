#define cimg_use_png
#define cimg_display 0
#include <CImg.h>
#include "nnedi.h"
using namespace cimg_library;

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
    CImg<uint8_t> d2 = nnedi_upscale_2x_cimg(src, nns);

    if(align) {
        d2.rotate(180);
        CImg<uint8_t> d4 = nnedi_upscale_2x_cimg(d2, nns);
        cimg_forXYC(d2,x,y,c)
            d2(x,y,0,c) = d4(d4._width-x*2-1, d4._height-y*2-1, 0,c);
    }

    d2.save(argv[2]);
    free(ctx);
    return 0;
}
