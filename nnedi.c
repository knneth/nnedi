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
    asm(
        "push %1 \n"
        "call *%4 \n"
        "pop %1 \n"
        :"=a"(ret), "+d"(tmp), "+c"(weights)
        :"a"(pix), "r"(dll+0x4380)
        :"memory"
    );
    return ret;
}

static float scale_net(int ninputs, int nneurons, const float *weights, const float *pix, float *tmp, float bias, float scale)
{
    static float ret; // static because otherwise we run out of regs
    ret = 0;
    asm(
        "push %8 \n"
        "sub  $8, %%esp \n"
        "movss %5, 4(%%esp) \n"
        "movss %4,  (%%esp) \n"
        "push %3 \n"
        "push %2 \n"
        "call *%9 \n"
        "add $20, %%esp \n"
        // eax and edx are specified just as a clobber, not because it matters which variables go in them.
        :"+m"(ret), "+c"(ninputs), "+a"(pix), "+d"(weights), "+x"(bias), "+x"(scale)
        :"S"(nneurons), "D"(tmp), "i"(&ret), "r"(dll+0x1bf0)
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
    void (*cast_pixels_12x4)(const uint8_t *src, int stride2, float *dst, float *mean) = dll+0x5000;
    void (*cast_pixels_general)(const uint8_t *src, int stride2, int width, int height, float *bias, float *scale, float *dst) = dll+0x5bf0;
    ALIGNED_16(float test_weights[252]);
    ALIGNED_16(float scale_weights[49*32]);
    memcpy(test_weights, dll+0x240f8, sizeof(test_weights));
    memcpy(scale_weights, dll+0x244e8, sizeof(scale_weights));

    // L/R mirroring ends up with only 1 copy of the last column.
    // T/B mirroring ends up with 2 copies of the last row.
    // this is inconsistent, but matches the way nnedi3 does it. (just for testing)
    for(int y=0; y<height; y++) {
        memcpy(tpix+y*2*tstride, src+av_clip(y,0,height-1)*sstride, width);
        for(int x=1; x<=5; x++)
            tpix[y*2*tstride-x] = tpix[y*2*tstride+x];
        for(int x=1; x<=6; x++)
            tpix[y*2*tstride+width-1+x] = tpix[y*2*tstride+width-1-x];
    }
    for(int y=0; y<2; y++)
        memcpy(tpix-(y+1)*2*tstride-5, tpix+y*2*tstride-5, twidth);
    for(int y=0; y<3; y++)
        memcpy(tpix+(height+y)*2*tstride-5, tpix+(height-1-y)*2*tstride-5, twidth);
    for(int y=0; y<height; y++) {
        for(int x=0; x<width; x++) {
            uint8_t *pix = tpix+(y*2+1)*tstride+x;
            ALIGNED_16(float fbuf[48]);
            ALIGNED_16(float ftmp[36]); // test uses 36, scale uses 32
            float mean, scale;
            cast_pixels_12x4(pix-3*tstride-5, tstride, fbuf, &mean);
            int t = test_net(test_weights, fbuf, ftmp);
            if(t) {
                *pix = av_clip_uint8(((pix[-tstride]+pix[tstride])*6-(pix[-tstride*3]+pix[tstride*3])+5)/10);
            } else {
                cast_pixels_general(pix-5*tstride-3, tstride, 8, 6, &mean, &scale, fbuf);
                float v = scale_net(48, 16, scale_weights, fbuf, ftmp, mean, scale);
                *pix = av_clip_uint8(v+.5f);
            }
        }
    }
    for(int y=0; y<height*2; y++)
        memcpy(dst+y*dstride, tpix+y*tstride, width);
    free(tbuf);
}
