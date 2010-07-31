#include <dlfcn.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <libavutil/avutil.h>
#include "nnedi.h"

#define ALIGNED_16(x) __attribute__((aligned(16))) x

void *dll = NULL;

static int test_net(const float *weights, const float *pix, float *tmp)
{
    int ret;
    asm("push %1 \n"
        "call *%4 \n"
        "pop %1 \n"
        :"=a"(ret), "+&d"(tmp)
        :"c"(weights), "a"(pix), "r"(dll+0x4380)
        :"memory"
    );
    return ret;
}

void upscale_v(uint8_t *dst, uint8_t *src, int width, int height, int dstride, int sstride)
{
    if(!dll) dll = dlopen("nnedi3.dll", 0);
    if(!dll) abort();
    int twidth = width+11;
    int theight = height*2+10;
    int tstride = (twidth+15)&~15;
    uint8_t *tbuf = memalign(16, tstride*theight+16);
    uint8_t *tpix = tbuf + tstride*4 + 16;
    void (*cast_pixels)(uint8_t *src, int stride2, float *dst, float *mean) = dll+0x5000;
    ALIGNED_16(float test_weights[252]);
    memcpy(test_weights, dll+0x240f8, sizeof(test_weights));

    for(int y=-2; y<height+3; y++) {
        memcpy(tpix+y*2*tstride, src+av_clip(y,0,height-1)*sstride, width);
        memset(tpix+y*2*tstride-5, tpix[y*2*tstride], 5);
        memset(tpix+y*2*tstride+width, tpix[y*2*tstride+width-1], 6);
    }
    for(int y=0; y<height; y++) {
        for(int x=0; x<width; x++) {
            ALIGNED_16(float fbuf[48]);
            ALIGNED_16(float ftmp[36]); // actually size 12, but test_net fills it sparsely
            float mean;
            cast_pixels(tpix+(y-1)*2*tstride+x-5, tstride, fbuf, &mean);
            int t = test_net(test_weights, fbuf, ftmp);
            tpix[(y*2+1)*tstride+x] = t*255;
        }
    }
    for(int y=0; y<height*2; y++)
        memcpy(dst+y*dstride, tpix+y*tstride, width);
    free(tbuf);
}
