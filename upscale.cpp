#define cimg_use_jpeg
#define cimg_use_png
#define cimg_verbosity 1
#include <CImg.h>
extern "C" {
#include "nnedi.h"
}
using namespace cimg_library;

int main(int argc, char **argv)
{
    if(argc != 3) {
        printf("usage: upscale in.png out.png\n");
        return 1;
    }
    CImg<uint8_t> src(argv[1]);
    CImg<uint8_t> tmp(src._width, src._height*2, 1, src._spectrum);
    for(int c=0; c<src._spectrum; c++)
        upscale_v(tmp.data(0,0,0,c), src.data(0,0,0,c), src._width, src._height, src._width, src._width);
    tmp.transpose();
    CImg<uint8_t> dst(tmp._width, tmp._height*2, 1, tmp._spectrum);
    for(int c=0; c<tmp._spectrum; c++)
        upscale_v(dst.data(0,0,0,c), tmp.data(0,0,0,c), tmp._width, tmp._height, tmp._width, tmp._width);
    dst.transpose();
    dst.save(argv[2]);
    return 0;
}
