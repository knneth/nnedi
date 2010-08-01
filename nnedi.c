#include <dlfcn.h>
#include <float.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <libavutil/avutil.h>
#include <bench.h>
#include "nnedi.h"

typedef float __attribute__((vector_size(16))) v4f;
typedef int   __attribute__((vector_size(16))) v4si;

static const v4f ps_1 = { 1.0, 1.0, 1.0, 1.0 };
static const v4si ps_abs = { ~(1<<31), ~(1<<31), ~(1<<31), ~(1<<31) };

static void cast_pixels_12x4(const uint8_t *src, int stride2, float *dst, float *mean)
{
    int sum = 0;
    for(int y=0; y<4; y++)
        for(int x=0; x<12; x++)
            sum += src[y*stride2*2+x];
    float bias = *mean = sum * (1/48.f);
    for(int y=0; y<4; y++)
        for(int x=0; x<12; x++)
            *dst++ = (src[y*stride2*2+x] - bias) * (1/127.5f);
}

static void cast_pixels_general(const uint8_t *src, int stride2, int width, int height, float *mean, float *stddev, float *dst)
{
    int sum = 0, sum2 = 0;
    for(int y=0; y<height; y++)
        for(int x=0; x<width; x++) {
            int v = src[y*stride2*2+x];
            sum += v;
            sum2 += v*v;
        }
    float norm = 1.f / (width*height);
    float bias = *mean = sum*norm;
    float var = sum2*norm - bias*bias;
    float scale;
    if(var > FLT_EPSILON) {
        *stddev = sqrt(var);
        scale = 1 / *stddev;
    } else {
        *stddev = 0;
        scale = 0;
    }
    for(int y=0; y<height; y++)
        for(int x=0; x<width; x++)
            *dst++ = (src[y*stride2*2+x] - bias) * scale;
}

static inline v4f splatps(float x)
{
    return (v4f){x,x,x,x};
}

static inline float haddps(v4f x)
{
    float ret, tmp;
    asm("movhlps %2, %1 \n"
        "addps   %2, %1 \n"
        "pshuflw $14, %1, %0 \n"
        "addss   %1, %0 \n"
        :"=x"(ret), "=&x"(tmp)
        :"x"(x)
    );
    return ret;
}

static inline v4f haddps_x2(v4f x0, v4f x1)
{
    v4f ret;
    asm("movaps   %1, %0 \n"
        "unpcklps %2, %1 \n"
        "unpckhps %2, %0 \n"
        "addps    %1, %0 \n"
        "movhlps  %0, %1 \n"
        "addps    %1, %0 \n"
        :"=&x"(ret), "+&x"(x0)
        :"x"(x1)
    );
    return ret;
}

static inline v4f haddps_x4(v4f x0, v4f x1, v4f x2, v4f x3)
{
    v4f ret;
    asm("movaps   %1, %0 \n"
        "unpcklps %3, %1 \n"
        "unpckhps %3, %0 \n"
        "addps    %1, %0 \n"
        "movaps   %2, %1 \n"
        "unpcklps %4, %2 \n"
        "unpckhps %4, %1 \n"
        "addps    %2, %1 \n"
        "movaps   %0, %2 \n"
        "movlhps  %1, %0 \n"
        "movhlps  %2, %1 \n"
        "addps    %1, %0 \n"
        :"=&x"(ret), "+&x"(x0), "+&x"(x2)
        :"x"(x1), "x"(x3)
    );
    return ret;
}

static inline v4f dotproduct_x4(const float *weights, const float *inputs, int n, int stride)
{
    v4f in = *(v4f*)inputs;
    v4f s0 = *(v4f*)weights * in;
    v4f s1 = *(v4f*)(weights+stride) * in;
    v4f s2 = *(v4f*)(weights+stride*2) * in;
    v4f s3 = *(v4f*)(weights+stride*3) * in;
    for(int i=4; i<n; i+=4) {
        in = *(v4f*)(inputs+i);
        s0 += *(v4f*)(weights+i) * in;
        s1 += *(v4f*)(weights+i+stride) * in;
        s2 += *(v4f*)(weights+i+stride*2) * in;
        s3 += *(v4f*)(weights+i+stride*3) * in;
    }
    return haddps_x4(s0, s1, s2, s3);
}

static inline v4f sigmoid_x4(v4f x)
{
    v4f t = x;
    asm("andps %3, %0 \n"
        "addps %2, %0 \n"
        "rcpps %0, %0 \n"
        "mulps %1, %0 \n"
        :"+&x"(x)
        :"x"(t), "m"(ps_1), "m"(ps_abs)
    );
    return x;
}

static inline void softmax(float *x, int n)
{
    float max = x[0];
    for(int i=1; i<n; i++)
        max = fmaxf(max, x[i]);
    for(int i=0; i<n; i++)
        x[i] = exp(x[i]-max);
}

static inline float weighted_average(float *weights, float *x, int n)
{
    v4f sum = splatps(0);
    v4f dot = splatps(0);
    for(int i=0; i<n; i+=4) {
        sum += *(v4f*)(weights+i);
        dot += *(v4f*)(weights+i) * *(v4f*)(x+i);
    }
    return haddps(dot)/haddps(sum);
}

static int test_net(const float *weights, const float *pix)
{
    ALIGNED_16(float tmp[8]);
    *(v4f*)tmp = sigmoid_x4(dotproduct_x4(weights, pix, 48, 48) + *(v4f*)(weights+48*4));
    weights += 49*4;
    *(v4f*)(tmp+4) = sigmoid_x4(dotproduct_x4(weights, tmp, 4, 4) + *(v4f*)(weights+4*4));
    weights += 5*4;
    v4f x = sigmoid_x4(dotproduct_x4(weights, tmp, 8, 8) + *(v4f*)(weights+8*4));
    v4f y;
    int ret = 0;
    asm("pshufd $0xa0, %1, %2 \n" // could be a pshuflw if I reordered the weights
        "maxps   %2, %1 \n"
        "movhlps %1, %2 \n"
        "comiss  %2, %1 \n"
        "seta    %b0 \n" // could directly use the flags in the branch
        :"+r"(ret), "+x"(x), "=x"(y)
    );
    return ret;
}

static float scale_net(int ninputs, int nneurons, const float *weights, const float *pix, float *tmp, float bias, float scale)
{
    const float *biases = weights+ninputs*nneurons*2;
    for(int i=0; i<nneurons*2; i+=4, weights+=ninputs*4)
        *(v4f*)(tmp+i) = dotproduct_x4(weights, pix, ninputs, ninputs) + *(v4f*)(biases+i);
    softmax(tmp, nneurons);
    for(int i=nneurons; i<nneurons*2; i+=4)
        *(v4f*)(tmp+i) = sigmoid_x4(*(v4f*)(tmp+i));
    return bias+5*scale*weighted_average(tmp, tmp+nneurons, nneurons);
}

void upscale_v(uint8_t *dst, uint8_t *src, int width, int height, int dstride, int sstride)
{
    int twidth = width+11;
    int theight = height*2+10;
    int tstride = (twidth+15)&~15;
    uint8_t *tbuf = memalign(16, tstride*theight+16);
    uint8_t *tpix = tbuf + tstride*4 + 16;

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
            ALIGNED_16(float ftmp[128]);
            float mean, scale;
            cast_pixels_12x4(pix-3*tstride-5, tstride, fbuf, &mean);
            int t = test_net(test_weights, fbuf);
            if(t) {
                *pix = av_clip_uint8(((pix[-tstride]+pix[tstride])*6-(pix[-tstride*3]+pix[tstride*3])+5)/10);
            } else {
                cast_pixels_general(pix-5*tstride-3, tstride, 8, 6, &mean, &scale, fbuf);
                float v = scale_net(48, 64, scale_weights_8x6x64, fbuf, ftmp, mean, scale);
                *pix = av_clip_uint8(v+.5f);
            }
        }
    }
    for(int y=0; y<height*2; y++)
        memcpy(dst+y*dstride, tpix+y*tstride, width);
    free(tbuf);
}
