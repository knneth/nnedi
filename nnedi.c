#include <dlfcn.h>
#include <float.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <libavutil/avutil.h>
#include <bench.h>
#include "nnedi.h"

#define NNS 16

typedef float __attribute__((vector_size(16))) v4f;
typedef int32_t __attribute__((vector_size(16))) v4si;
typedef int16_t __attribute__((vector_size(16))) v8si;

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

static inline v4si haddpi_x4(v4si x0, v4si x1, v4si x2, v4si x3)
{
    v4si ret;
    asm("movdqa     %1, %0 \n"
        "punpckldq  %3, %1 \n"
        "punpckhdq  %3, %0 \n"
        "paddd      %1, %0 \n"
        "movdqa     %2, %1 \n"
        "punpckldq  %4, %2 \n"
        "punpckhdq  %4, %1 \n"
        "paddd      %2, %1 \n"
        "movdqa     %0, %2 \n"
        "punpcklqdq %1, %0 \n"
        "punpckhqdq %1, %2 \n"
        "paddd      %2, %0 \n"
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

static inline v4f cvtdq2ps(v4si x)
{
    v4f ret;
    asm("cvtdq2ps %1, %0 \n" :"=x"(ret) :"x"(x));
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

static inline v4si dotproduct_x4i(const int16_t *weights, const int16_t *inputs, int n, int stride)
{
    v4si s0,s1,s2,s3,t4,t5,t6,t7;
    asm("movdqa %4, %0 \n"
        "movdqa %0, %1 \n"
        "movdqa %0, %2 \n"
        "movdqa %0, %3 \n"
        "pmaddwd %5, %0 \n"
        "pmaddwd %6, %1 \n"
        "pmaddwd %7, %2 \n"
        "pmaddwd %8, %3 \n"
        :"=&x"(s0), "=&x"(s1), "=&x"(s2), "=&x"(s3)
        :"m"(inputs[0]), "m"(weights[0]), "m"(weights[stride]), "m"(weights[stride*2]), "m"(weights[stride*3])
    );
    for(int i=8; i<n; i+=8)
        asm("movdqa %8, %4 \n"
            "movdqa %4, %5 \n"
            "movdqa %4, %6 \n"
            "movdqa %4, %7 \n"
            "pmaddwd %9, %4 \n"
            "pmaddwd %10, %5 \n"
            "pmaddwd %11, %6 \n"
            "pmaddwd %12, %7 \n"
            "paddd   %4, %0 \n"
            "paddd   %5, %1 \n"
            "paddd   %6, %2 \n"
            "paddd   %7, %3 \n"
            :"+&x"(s0), "+&x"(s1), "+&x"(s2), "+&x"(s3), "=&x"(t4), "=&x"(t5), "=&x"(t6), "=&x"(t7)
            :"m"(inputs[i]), "m"(weights[i]), "m"(weights[i+stride]), "m"(weights[i+stride*2]), "m"(weights[i+stride*3])
        );
    return haddpi_x4(s0, s1, s2, s3);
}

static inline v4f sigmoid_x4(v4f x)
{
    static const v4f ps_1 = { 1.0, 1.0, 1.0, 1.0 };
    static const v4si ps_abs = { ~(1<<31), ~(1<<31), ~(1<<31), ~(1<<31) };
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
    static const v4f bias = SPLATPS(3<<22);
    // Taylor coefs are 1, 1*ln2, .5*ln2*ln2. But if I'm going to truncate the Taylor
    // series and care only about [-.5,.5], these modified coefs are a better fit.
    static const v4f c0 = SPLATPS(1.00035);
    static const v4f c1 = SPLATPS(1.01173*M_LN2);
    static const v4f c2 = SPLATPS(0.49401*M_LN2*M_LN2);
    v4f t, u, v; // gcc pessimizes this if I remove the unused variable
    asm volatile (
        "movaps %0, %2 \n\t"
        "addps  %4, %0 \n\t"
        "movaps %0, %3 \n\t"
        "subps  %4, %0 \n\t" // round(x)
        "pslld $23, %3 \n\t"
        "subps  %0, %2 \n\t"
        "movaps %2, %0 \n\t"
        "mulps  %2, %2 \n\t"
        "mulps  %6, %0 \n\t"
        "mulps  %7, %2 \n\t"
        "addps  %5, %0 \n\t"
        "addps  %2, %0 \n\t"
        "paddd  %3, %0 \n\t"
        :"+x"(x), "=x"(t), "=x"(u), "=x"(v)
        :"m"(bias), "m"(c0), "m"(c1), "m"(c2)
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

static int test_net(const int16_t *weightsi, const float *weightsf, const int16_t *pix, float mean)
{
    v4f tmp[2];
    const v4f *weightsv = (v4f*)weightsf;
    tmp[0] = cvtdq2ps(dotproduct_x4i(weightsi, pix, 48, 48))*weightsv[0] - splatps(mean)*weightsv[1];
    tmp[0] = sigmoid_x4(tmp[0] + weightsv[2]);
    tmp[1] = sigmoid_x4(dotproduct_x4(weightsf+12, (float*)tmp, 4, 4) + weightsv[7]);
    v4f x = sigmoid_x4(dotproduct_x4(weightsf+32, (float*)tmp, 8, 8) + weightsv[16]);
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

__attribute__((noinline))
static float scale_net(const int16_t *weightsi, const float *weightsf, const int16_t *pix, float invstddev)
{
    const int ninputs = 48;
    v4f tmp[NNS/2];
    const v4f *biases = (v4f*)weightsf;
    v4f scalev = splatps(invstddev);
    for(int i=0; i<NNS/2; i++, weightsi+=ninputs*4)
        tmp[i] = cvtdq2ps(dotproduct_x4i(weightsi, pix, ninputs, ninputs)) * scalev * biases[i*2] + biases[i*2+1];
    softmax((float*)tmp, NNS);
    for(int i=NNS/4; i<NNS/2; i++)
        tmp[i] = sigmoid_x4(tmp[i]);
    return 5*weighted_average((float*)tmp, (float*)tmp+NNS, NNS);
}

static void init_testblock(int16_t *block, uint8_t *src, int stride)
{
    for(int y=0; y<4; y++)
        for(int x=2; x<12; x++)
            block[x*4+y] = src[x+y*stride];
}

static void shift_testblock(int16_t *block, uint8_t *src, int stride)
{
//  memcpy(block, block+8, 40*sizeof(*block));
    asm("movdqa      16(%0), %%xmm0 \n"
        "movdqa      32(%0), %%xmm1 \n"
        "movdqa      48(%0), %%xmm2 \n"
        "movdqa      64(%0), %%xmm3 \n"
        "movdqa      80(%0), %%xmm4 \n"
        "movdqa      %%xmm0,   (%0) \n"
        "movdqa      %%xmm1, 16(%0) \n"
        "movdqa      %%xmm2, 32(%0) \n"
        "movdqa      %%xmm3, 48(%0) \n"
        "movdqa      %%xmm4, 64(%0) \n"
        ::"r"(block)
    );
    for(int i=0; i<4; i++) {
        block[40+i] = src[i*stride];
        block[44+i] = src[i*stride+1];
    }
}

static void cast_pixels_test(const uint8_t *src, intptr_t stride, int16_t *dst)
{
//  for(int y=0; y<4; y++)
//      for(int x=0; x<12; x++)
//          dst[y*12+x] = src[y*stride+x];

#define ROW(dst, src0, src1)\
        "movq      "src0", %%xmm0 \n"\
        "movd    8+"src0", %%mm1 \n"\
        "movd      "src1", %%mm2 \n"\
        "movq    4+"src1", %%xmm3 \n"\
        "punpcklbw %%xmm4, %%xmm0 \n"\
        "punpcklbw  %%mm4, %%mm1 \n"\
        "punpcklbw  %%mm4, %%mm2 \n"\
        "punpcklbw %%xmm4, %%xmm3 \n"\
        "movdqa %%xmm0,    "dst" \n"\
        "movq    %%mm1, 16+"dst" \n"\
        "movq    %%mm2, 24+"dst" \n"\
        "movdqa %%xmm3, 32+"dst" \n"\

    asm("pxor %%xmm4, %%xmm4 \n"
        "pxor  %%mm4, %%mm4 \n"
        ROW( "0(%1)", "0(%2)", "0(%2,%3)")
        ROW("48(%1)", "0(%2,%3,2)", "0(%2,%4)")
        "emms \n"
        :"=m"(*(struct {int16_t x[48];}*)dst)
        :"r"(dst), "r"(src), "r"(stride), "r"(stride*3)
        :"xmm0", "mm1", "mm2", "xmm3"
    );
#undef ROW
}

static void cast_pixels_general(const uint8_t *src, intptr_t stride, int width, int height, float *mean, float *stddev, float *invstddev, int16_t *dst)
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
    if(var > FLT_EPSILON) {
        *invstddev = rsqrtss(var);
        *stddev = rcpss(*invstddev);
    } else {
        *invstddev = 0;
        *stddev = 0;
    }

#define ROW(dst, src)\
        "movq       "src", %%xmm0 \n"\
        "punpcklbw %%xmm3, %%xmm0 \n"\
        "psllw         $4, %%xmm0 \n"\
        "psubw     %%xmm2, %%xmm0 \n"\
        "movdqa    %%xmm0, "dst"  \n"\

    asm("movd            %5, %%xmm2 \n"
        "pshuflw $0, %%xmm2, %%xmm2 \n"
        "punpcklqdq  %%xmm2, %%xmm2 \n"
        "pxor        %%xmm3, %%xmm3 \n"
        ROW( "0(%1)", "0(%2)")
        ROW("16(%1)", "0(%2,%4)")
        ROW("32(%1)", "0(%2,%4,2)")
        ROW("48(%1)", "0(%3)")
        ROW("64(%1)", "0(%3,%4)")
        ROW("80(%1)", "0(%3,%4,2)")
        :"=m"(*(struct {int16_t x[48];}*)dst)
        :"r"(dst), "r"(src), "r"(src+stride*3), "r"(stride), "r"((sum+1)/3)
        :"xmm0", "xmm1", "xmm2", "xmm3"
    );
#undef ROW
}

// FIXME cap scaling factors so that intermediate sums don't overflow; or allow 7fff if that works.
static void munge_test_weights(int16_t *dsti, int16_t *dsti_transpose, float *dstf, const float *src)
{
    for(int j=0; j<4; j++, src+=48, dsti+=48, dsti_transpose+=48) {
        float max = 0;
        int sum = 0;
        for(int i=0; i<48; i++)
            max = fmaxf(max, fabsf(src[i]));
        float scale = 0x3fff/max;
        for(int i=0; i<48; i++)
            sum += dsti[i] = dsti_transpose[i/12+i%12*4] = roundf(src[i]*scale);
        dstf[j] = max/(0x3fff*127.5f);
        dstf[j+4] = sum*dstf[j]/48;
    }
    memcpy(dstf+8, src, 60*sizeof(float));
}

static void munge_scale_weights(int16_t *dsti, float *dstf, const float *src)
{
    float scales[2*NNS];
    for(int j=0; j<2*NNS; j++, src+=48, dsti+=48) {
        float max = 0;
        for(int i=0; i<48; i++)
            max = fmaxf(max, fabsf(src[i]));
        float scale = 0x3fff/max;
        for(int i=0; i<48; i++)
            dsti[i] = roundf(src[i]*scale);
        scales[j] = max/(0x3fff*16);
    }
    for(int j=0; j<2*NNS; j+=4) {
        memcpy(dstf+j*2, scales+j, 4*sizeof(float));
        memcpy(dstf+j*2+4, src+j, 4*sizeof(float));
    }
    for(int j=0; j<2*NNS; j++)
        dstf[j] *= M_LOG2E;
}

static void block_sums(float *blocks, uint16_t *dst, uint8_t *src, int n, int width, int y, intptr_t stride)
{
    int sum = 0;
    for(int i=0; i<width; i++)
        sum += src[i];
    dst += y*stride;
    dst[0] = sum;
    for(int i=1; i<n; i++)
        dst[i] = sum += src[i+width-1] - src[i-1];
    dst -= y*stride;
    if(blocks)
        for(int i=0; i<n; i+=8)
            asm("movdqa (%3), %0 \n"
                "paddw  (%3,%4,2), %0 \n"
                "paddw  (%3,%4,4), %0 \n"
                "paddw  (%3,%5,2), %0 \n"
                "movdqa    %0, %1 \n"
                "punpcklwd %2, %0 \n"
                "punpckhwd %2, %1 \n"
                "cvtdq2ps  %0, %0 \n"
                "cvtdq2ps  %1, %1 \n"
                :"=&x"(*(v4f*)&blocks[i]), "=&x"(*(v4f*)&blocks[i+4])
                :"x"(splatpi(0)), "r"(dst+i), "r"(stride), "r"(stride*3)
            );
}

static int merge_test_neighbors(uint8_t *dst, uint16_t *retest, uint8_t *row0, uint8_t *row1, uint8_t *row2, int n, int parity)
{
    uint16_t *pretest = retest;
    int n2 = (n+1)>>1;
    int n32 = (n+31)>>5;
    int n64 = (n+63)>>6;
#if 0
    uint8_t *row1offset = row1-1+2*parity;
    for(int x=0; x<n2; x++) {
        dst[2*x] = row1[x];
        dst[2*x+1] = row1[x];
        // FIXME check that this is ok for odd width
        int a = row0[x], b = row1[x], c = row2[x], d = row1offset[x];
        if((a^b)|(b^c)|(c^d))
            *pretest++ = 2*x+parity;
    }
#else
    uint16_t *masks = retest+n2-n32;
#define MERGE(load_row1offset) {\
        intptr_t x = -n2;\
        asm("1: \n"\
            "movdqa   (%3,%1), %%xmm0 \n"\
            load_row1offset\
            "movdqa   (%4,%1), %%xmm2 \n"\
            "movdqa   (%5,%1), %%xmm3 \n"\
            "pcmpeqb   %%xmm0, %%xmm1 \n"\
            "pcmpeqb   %%xmm0, %%xmm2 \n"\
            "pcmpeqb   %%xmm0, %%xmm3 \n"\
            "pand      %%xmm2, %%xmm1 \n"\
            "pand      %%xmm3, %%xmm1 \n"\
            "movdqa    %%xmm0, %%xmm4 \n"\
            "punpcklbw %%xmm0, %%xmm0 \n"\
            "punpckhbw %%xmm4, %%xmm4 \n"\
            "pmovmskb  %%xmm1,  %%eax \n"\
            "movdqa    %%xmm0,   (%2,%1,2) \n"\
            "movdqa    %%xmm4, 16(%2,%1,2) \n"\
            "mov %%ax, (%0) \n"\
            "add   $2,  %0 \n"\
            "add  $16,  %1 \n"\
            "jl 1b \n"\
            :"+&r"(masks), "+&r"(x)\
            :"r"(dst+n2*2), "r"(row1+n2), "r"(row0+n2), "r"(row2+n2)\
            :"eax", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "memory"\
        );\
        masks -= n32;\
    }
    if(parity) {
        MERGE("movdqa 16(%3,%1), %%xmm1 \n"
              "palignr $1, %%xmm0, %%xmm1 \n");
    } else {
        MERGE("movdqa %%xmm0, %%xmm1 \n"
              "palignr $15, -16(%3,%1), %%xmm1 \n");
    }
#undef MERGE
    masks[n32] = 0xffff;
    uint32_t *masks2 = (uint32_t*)masks;
    for(int x=0; x<n64; x++) {
        int x2 = x*64-2+parity;
        uint32_t mask = ~masks2[x];
        while(mask) {
            int tz = __builtin_ctz(mask);
            x2 += (tz+1)*2;
            *pretest++ = x2;
            mask >>= tz;
            mask >>= 1;
        }
    }
#endif
    return pretest - retest;
}

static void bicubic(uint8_t *dst, uint8_t *src, int stride, int n)
{
//  for(int x=0; x<n; x++)
//      dst[x] = av_clip_uint8(((src[x+stride]+src[x+stride*2])*38-(src[x]+src[x+stride*3])*6+32)>>6);

    ALIGNED_16(static const int8_t coef[16]) = {38,-6,38,-6,38,-6,38,-6,38,-6,38,-6,38,-6,38,-6};
    ALIGNED_16(static const int16_t round[8]) = {32,32,32,32,32,32,32,32};
    intptr_t i = -n;
    asm volatile(
        "movdqa        %6, %%xmm6 \n"
        "movdqa        %7, %%xmm7 \n"
        "1: \n"
        "movdqa   (%3,%0), %%xmm0 \n"
        "movdqa   (%4,%0), %%xmm1 \n"
        "movdqa    %%xmm0, %%xmm2 \n"
        "movdqa    %%xmm1, %%xmm3 \n"
        "punpcklbw (%2,%0),%%xmm0 \n"
        "punpcklbw (%5,%0),%%xmm1 \n"
        "punpckhbw (%2,%0),%%xmm2 \n"
        "punpckhbw (%5,%0),%%xmm3 \n"
        "pmaddubsw %%xmm6, %%xmm0 \n"
        "pmaddubsw %%xmm6, %%xmm1 \n"
        "pmaddubsw %%xmm6, %%xmm2 \n"
        "pmaddubsw %%xmm6, %%xmm3 \n"
        "paddw     %%xmm7, %%xmm0 \n"
        "paddw     %%xmm7, %%xmm2 \n"
        "paddw     %%xmm1, %%xmm0 \n"
        "paddw     %%xmm3, %%xmm2 \n"
        "psraw         $6, %%xmm0 \n"
        "psraw         $6, %%xmm2 \n"
        "packuswb  %%xmm2, %%xmm0 \n"
        "movdqa    %%xmm0, (%1,%0) \n"
        "add $16, %0 \n"
        "jl 1b \n"
        :"+&r"(i)
        :"r"(dst+n), "r"(src+n), "r"(src+stride+n), "r"(src+stride*2+n), "r"(src+stride*3+n),
         "m"(*coef), "m"(*round)
        :"xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "memory"
    );
}

void upscale_v(uint8_t *dst, uint8_t *src, int width, int height, int dstride, int sstride)
{
    int twidth = width+11;
    int theight = height*2+10;
    int tstride = (twidth+15)&~15;
    uint8_t *tbuf = memalign(16, tstride*theight+16);
    uint8_t *tpix = tbuf + tstride*4 + 16;
    uint16_t *sum_w12 = memalign(16, 4*tstride*sizeof(uint16_t));
    float *sum_12x4[2] = { memalign(16, tstride*sizeof(float)), memalign(16, tstride*sizeof(float)) };
    uint8_t *tested = memalign(16, 3*tstride+16); // FIXME only needs stride=align(tstride/2+2)
    uint8_t *tested2 = memalign(16, tstride+16);
    uint16_t *retest = memalign(16, (tstride+16)/2*sizeof(uint16_t));
    memset(tested, 0, 3*tstride+16);
    tested += 16;
    ALIGNED_16(int16_t test_weights_i[48*4]);
    ALIGNED_16(int16_t test_weights_i_transpose[48*4]);
    ALIGNED_16(float test_weights_f[68]);
    ALIGNED_16(int16_t scale_weights_i[48*2*NNS]);
    ALIGNED_16(float scale_weights_f[4*NNS]);
    ALIGNED_16(int16_t ibuf[48]);
    ALIGNED_16(int16_t ibuf2[48]);
    munge_test_weights(test_weights_i, test_weights_i_transpose, test_weights_f, test_weights);
    munge_scale_weights(scale_weights_i, scale_weights_f,
        NNS==16 ? scale_weights_8x6x16 : NNS==32 ? scale_weights_8x6x32 : scale_weights_8x6x64);

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
        block_sums(NULL, sum_w12, tpix+(y-1)*2*tstride-5, width, 12, y+1, tstride);
    for(int y=0, testy=0; y<height; y++) {
        for(; testy<=y+1 && testy<height; testy++) {
            block_sums(sum_12x4[testy&1], sum_w12, tpix+(testy+2)*2*tstride-5, width, 12, testy&3, tstride);
            uint8_t *pt = tested+(testy%3)*tstride;
            int x = !(testy&1);
            init_testblock(ibuf, tpix+(testy-1)*2*tstride+x-7, 2*tstride);
            for(; x<width; x+=2) {
                uint8_t *pix = tpix+(testy*2+1)*tstride+x;
                shift_testblock(ibuf, pix-3*tstride+5, 2*tstride);
                pt[x/2] = test_net(test_weights_i_transpose, test_weights_f, ibuf, sum_12x4[testy&1][x]);
            }
        }
        if(y==height-1) memset(tested+(y+1)%3*tstride, 0, tstride);
        int nretest = merge_test_neighbors(tested2, retest, tested+(y+2)%3*tstride, tested+y%3*tstride, tested+(y+1)%3*tstride, width, y&1);
        uint8_t *pix = tpix+(y-1)*2*tstride-5;
        for(int i=0; i<nretest; i++) {
            int x = retest[i];
            cast_pixels_test(pix+x, 2*tstride, ibuf);
            tested2[x] = test_net(test_weights_i, test_weights_f, ibuf, sum_12x4[y&1][x]);
        }
        bicubic(tpix+(y*2+1)*tstride, tpix+(y*2-2)*tstride, tstride*2, width);
        for(int x=0; x<width; x++) {
            uint8_t *pix = tpix+(y*2+1)*tstride+x;
            if(!tested2[x]) {
                float mean, stddev, invstddev;
                cast_pixels_general(pix-5*tstride-3, tstride*2, 8, 6, &mean, &stddev, &invstddev, ibuf2);
                float v = scale_net(scale_weights_i, scale_weights_f, ibuf2, invstddev)*stddev+mean;
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
    free(sum_12x4[0]);
    free(sum_12x4[1]);
    free(tested-16);
    free(tested2);
    free(retest);
}
