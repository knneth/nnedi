#include <dlfcn.h>
#include <float.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <libavutil/avutil.h>
#include <bench.h>
#include "nnedi.h"

#define NNS 64

typedef float __attribute__((vector_size(16))) v4f;
typedef int   __attribute__((vector_size(16))) v4si;

static const v4f ps_1 = { 1.0, 1.0, 1.0, 1.0 };
static const v4si ps_abs = { ~(1<<31), ~(1<<31), ~(1<<31), ~(1<<31) };
static const v4si unpackbd_shuf = { 0xffffff00, 0xffffff01, 0xffffff02, 0xffffff03 };

static inline v4si splatpi(int x)
{
    return (v4si){x,x,x,x};
}

static inline v4f splatps(float x)
{
    return (v4f){x,x,x,x};
}

#define SPLATPS(x) {x,x,x,x}

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

static inline float rcpss(float x)
{
    asm("rcpss %0, %0 \n" :"+x"(x));
    return x;
}

static inline float rsqrtss(float x)
{
    asm("rsqrtss %0, %0 \n" :"+x"(x));
    return x;
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

static inline float vec_max(float *x, int n)
{
//  float max = x[0];
//  for(int i=1; i<n; i++)
//      max = fmaxf(max, x[i]);
    intptr_t i = n*4-32;
    float max, tmp;
    asm("movaps 16(%3,%2), %0 \n"
        "maxps    (%3,%2), %0 \n"
        "1: \n"
        "sub          $32, %2 \n"
        "maxps  16(%3,%2), %0 \n"
        "maxps    (%3,%2), %0 \n"
        "jg 1b \n"
        "movhlps       %0, %1 \n"
        "maxps         %1, %0 \n"
        "pshuflw  $14, %0, %1 \n"
        "maxss         %1, %0 \n"
        :"=x"(max), "=x"(tmp), "+&r"(i)
        :"r"(x)
    );
    return max;
}

static inline v4f exp2ps(v4f x)
{
    // this might be excessively precise.
    static const v4f ss_bias   = SPLATPS(3<<22);
    static const v4f ss_1      = SPLATPS(1.0);
    static const v4f ss_ln2    = SPLATPS(M_LN2);
    static const v4f ss_0_5035 = SPLATPS(0.5035*M_LN2*M_LN2);
    static const v4f ss_0_1667 = SPLATPS(0.1667*M_LN2*M_LN2*M_LN2);
    v4f t, u, v;
    asm volatile (
        "movaps %0, %2 \n\t"
        "addps  %4, %0 \n\t"
        "movaps %0, %3 \n\t"
        "subps  %4, %0 \n\t" // round(x)
        "pslld $23, %3 \n\t"
        "subps  %0, %2 \n\t"
        "movaps %2, %0 \n\t"
        "mulps  %2, %2 \n\t"
        "movaps %2, %1 \n\t"
        "mulps  %0, %2 \n\t"
        "mulps  %6, %0 \n\t"
        "mulps  %7, %1 \n\t"
        "mulps  %8, %2 \n\t"
        "addps  %5, %0 \n\t"
        "addps  %1, %0 \n\t"
        "addps  %2, %0 \n\t"
        "paddd  %3, %0 \n\t"
        :"+x"(x), "=x"(t), "=x"(u), "=x"(v)
        :"m"(ss_bias), "m"(ss_1), "m"(ss_ln2), "m"(ss_0_5035), "m"(ss_0_1667)
    );
    return x;
}

static inline void softmax(float *x, int n)
{
    v4f max = splatps(vec_max(x,n));
    for(int i=0; i<n; i+=4)
        *(v4f*)(x+i) = exp2ps(*(v4f*)(x+i) - max);
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

static float scale_net(int ninputs, int nneurons, const float *weights, const float *pix)
{
    ALIGNED_16(float tmp[128]);
    const float *biases = weights+ninputs*nneurons*2;
    for(int i=0; i<nneurons*2; i+=4, weights+=ninputs*4)
        *(v4f*)(tmp+i) = dotproduct_x4(weights, pix, ninputs, ninputs) + *(v4f*)(biases+i);
    softmax(tmp, nneurons);
    for(int i=nneurons; i<nneurons*2; i+=4)
        *(v4f*)(tmp+i) = sigmoid_x4(*(v4f*)(tmp+i));
    return 5*weighted_average(tmp, tmp+nneurons, nneurons);
}

static void cast_pixels_12x4(const uint8_t *src, int stride, float *dst, float mean)
{
    v4f biasv = splatps(mean);
    for(int y=0; y<4; y++)
        for(int x=0; x<12; x+=4, dst+=4)
            asm("movd         %1, %%xmm0 \n"
                "pshufb       %2, %%xmm0 \n"
                "cvtdq2ps %%xmm0, %%xmm0 \n"
                "subps        %3, %%xmm0 \n"
                "movaps   %%xmm0, %0 \n"
                :"=m"(*(v4f*)dst)
                :"m"(src[y*stride+x]),
                 "x"(unpackbd_shuf), "x"(biasv)
                :"xmm0"
            );
}

static void cast_pixels_general(const uint8_t *src, int stride, int width, int height, float *mean, float *stddev, float *dst)
{
    int sum = 0, sum2 = 0;
#if 0
    for(int y=0; y<height; y++)
        for(int x=0; x<width; x++) {
            int v = src[y*stride+x];
            sum += v;
            sum2 += v*v;
        }
#else
    asm("pxor       %%xmm0, %%xmm0 \n"
        "movq    (%2),      %%xmm1 \n"
        "movq    (%2,%3),   %%xmm3 \n"
        "movdqa     %%xmm1, %%xmm2 \n"
        "punpcklqdq %%xmm3, %%xmm2 \n"
        "punpcklbw  %%xmm0, %%xmm1 \n"
        "punpcklbw  %%xmm0, %%xmm3 \n"
        "pmaddwd    %%xmm1, %%xmm1 \n"
        "pmaddwd    %%xmm3, %%xmm3 \n"
        "psadbw     %%xmm0, %%xmm2 \n"
        "paddd      %%xmm3, %%xmm1 \n"

        "movq    (%2,%3,2), %%xmm3 \n"
        "movq    (%4),      %%xmm5 \n"
        "movdqa     %%xmm3, %%xmm4 \n"
        "punpcklqdq %%xmm5, %%xmm4 \n"
        "punpcklbw  %%xmm0, %%xmm3 \n"
        "punpcklbw  %%xmm0, %%xmm5 \n"
        "pmaddwd    %%xmm3, %%xmm3 \n"
        "pmaddwd    %%xmm5, %%xmm5 \n"
        "psadbw     %%xmm0, %%xmm4 \n"
        "paddd      %%xmm5, %%xmm3 \n"
        "paddd      %%xmm4, %%xmm2 \n"
        "paddd      %%xmm3, %%xmm1 \n"

        "movq    (%4,%3),   %%xmm3 \n"
        "movq    (%4,%3,2), %%xmm5 \n"
        "movdqa     %%xmm3, %%xmm4 \n"
        "punpcklqdq %%xmm5, %%xmm4 \n"
        "punpcklbw  %%xmm0, %%xmm3 \n"
        "punpcklbw  %%xmm0, %%xmm5 \n"
        "pmaddwd    %%xmm3, %%xmm3 \n"
        "pmaddwd    %%xmm5, %%xmm5 \n"
        "psadbw     %%xmm0, %%xmm4 \n"
        "paddd      %%xmm5, %%xmm3 \n"
        "paddd      %%xmm4, %%xmm2 \n"
        "paddd      %%xmm3, %%xmm1 \n"

        "movhlps    %%xmm2, %%xmm4 \n"
        "movhlps    %%xmm1, %%xmm3 \n"
        "paddd      %%xmm4, %%xmm2 \n"
        "paddd      %%xmm3, %%xmm1 \n"
        "pshuflw $14,%%xmm1, %%xmm3 \n"
        "paddd      %%xmm3, %%xmm1 \n"
        "movd       %%xmm2, %0 \n"
        "movd       %%xmm1, %1 \n"
        :"=r"(sum), "=r"(sum2)
        :"r"(src), "r"(stride), "r"(src+stride*3)
        :"xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
    );
#endif
    float norm = 1.f / (width*height);
    float bias = *mean = sum*norm;
    float var = sum2*norm - bias*bias;
    float scale;
    if(var > FLT_EPSILON) {
        scale = rsqrtss(var);
        *stddev = rcpss(scale);
    } else {
        *stddev = 0;
        scale = 0;
    }
    v4f biasv = splatps(bias);
    v4f scalev = splatps(scale);
    for(int y=0; y<height; y++)
        for(int x=0; x<width; x+=4, dst+=4)
            asm("movd         %1, %%xmm0 \n"
                "pshufb       %2, %%xmm0 \n"
                "cvtdq2ps %%xmm0, %%xmm0 \n"
                "subps        %3, %%xmm0 \n"
                "mulps        %4, %%xmm0 \n"
                "movaps   %%xmm0, %0 \n"
                :"=m"(*(v4f*)dst)
                :"m"(src[y*stride+x]),
                 "x"(unpackbd_shuf), "x"(biasv), "x"(scalev)
                :"xmm0"
            );
}

static void munge_test_weights(float *dst, const float *src)
{
    memcpy(dst, src, 252*sizeof(float));
    for(int i=0; i<48*4; i++)
        dst[i] *= 1/127.5f;
}

static void munge_scale_weights(float *dst, const float *src)
{
    memcpy(dst, src, 49*2*NNS*sizeof(float));
    for(int i=0; i<48*NNS; i++)
        dst[i] *= M_LOG2E;
    for(int i=96*NNS; i<97*NNS; i++)
        dst[i] *= M_LOG2E;
}

static void block_sums(uint16_t *dst, uint8_t *src, int n, int width)
{
    int sum = 0;
    for(int i=0; i<width; i++)
        sum += src[i];
    dst[0] = sum;
    for(int i=1; i<n; i++)
        dst[i] = sum += src[i+width-1] - src[i-1];
}

void upscale_v(uint8_t *dst, uint8_t *src, int width, int height, int dstride, int sstride)
{
    int twidth = width+11;
    int theight = height*2+10;
    int tstride = (twidth+15)&~15;
    uint8_t *tbuf = memalign(16, tstride*theight+16);
    uint8_t *tpix = tbuf + tstride*4 + 16;
    uint16_t *sum_w12 = memalign(16, 4*tstride*sizeof(uint16_t));
    ALIGNED_16(float test_weights2[252]);
    ALIGNED_16(float scale_weights2[49*2*NNS]);
    munge_test_weights(test_weights2, test_weights);
    munge_scale_weights(scale_weights2, NNS==16 ? scale_weights_8x6x16 : NNS==32 ? scale_weights_8x6x32 : scale_weights_8x6x64);

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
    uint64_t t0 = read_time();
    for(int y=0; y<3; y++)
        block_sums(sum_w12+y*tstride, tpix+(y-1)*2*tstride-5, width, 12);
    for(int y=0; y<height; y++) {
        block_sums(sum_w12+((y+3)&3)*tstride, tpix+(y+2)*2*tstride-5, width, 12);
        for(int x=0; x<width; x++) {
            uint8_t *pix = tpix+(y*2+1)*tstride+x;
            ALIGNED_16(float fbuf[48]);
            float mean = (sum_w12[x] + sum_w12[x+tstride] + sum_w12[x+tstride*2] + sum_w12[x+tstride*3])*(1.f/48);
            float stddev;
            cast_pixels_12x4(pix-3*tstride-5, tstride*2, fbuf, mean);
            int t = test_net(test_weights2, fbuf);
            if(t) {
                *pix = av_clip_uint8(((pix[-tstride]+pix[tstride])*6-(pix[-tstride*3]+pix[tstride*3])+5)/10);
            } else {
                cast_pixels_general(pix-5*tstride-3, tstride*2, 8, 6, &mean, &stddev, fbuf);
                float v = scale_net(48, NNS, scale_weights2, fbuf)*stddev+mean;
                *pix = av_clip_uint8(v+.5f);
            }
        }
    }
    uint64_t t1 = read_time();
    printf("%d Mcycles\n", (int)((t1-t0)/1000000));
    for(int y=0; y<height*2; y++)
        memcpy(dst+y*dstride, tpix+y*tstride, width);
    free(tbuf);
    free(sum_w12);
}
