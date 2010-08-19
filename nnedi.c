#include <dlfcn.h>
#include <float.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <libavutil/avutil.h>
#include <bench.h>
#include "nnedi.h"

#define NNS 16
#define USE_ASM

#ifndef USE_ASM
static inline int dotproduct(const int16_t *weights, const int16_t *inputs, int n)
{
    int s = 0;
    for(int i=0; i<n; i++)
        s += weights[i]*inputs[i];
    return s;
}

static inline float dotproductf(const float *weights, const float *inputs, int n)
{
    float s = 0;
    for(int i=0; i<n; i++)
        s += weights[i]*inputs[i];
    return s;
}

static inline float sigmoid(float x)
{
    return x / (fabsf(x) + 1.f);
}

static inline float fast_exp2(float x)
{
    // Taylor coefs are 1, 1*ln2, .5*ln2*ln2. But if I'm going to truncate the Taylor
    // series and care only about [-.5,.5], these modified coefs are a better fit.
    float c0 = 1.00035;
    float c1 = 1.01173*M_LN2;
    float c2 = 0.49401*M_LN2*M_LN2;
    int i = (int)(x + 128.5f) - 128;
    x -= i;
    x = c0 + c1*x + c2*x*x;
    union { float f; int i; } u;
    u.i = (i+128)<<23;
    return x * u.f;
}

static inline float weighted_average(float *weights, float *x, int n)
{
    float sum = 0, den = 0;
    for(int i=0; i<n; i++) {
        sum += weights[i] * x[i];
        den += weights[i];
    }
    return sum/den;
}

static void cast_pixels_test(int16_t *dst, const uint8_t *src, intptr_t stride)
{
    for(int y=0; y<4; y++)
        for(int x=0; x<12; x++)
            dst[y*12+x] = src[y*stride+x];
}

static void cast_pixels_scale(int16_t *dst, const uint8_t *src, intptr_t stride, float *mean, float *stddev, float *invstddev)
{
    int sum = 0, sum2 = 0;
    for(int y=0; y<6; y++)
        for(int x=0; x<8; x++) {
            int v = src[y*stride+x];
            sum += v;
            sum2 += v*v;
        }
    int var = sum2*48-sum*sum;
    *mean = sum*(1/48.f);
    if(var > 0) {
        *stddev = (1/48.f) * sqrtf(var);
        *invstddev = 1.f / *stddev;
    } else {
        *stddev = 0;
        *invstddev = 0;
    }
    for(int y=0; y<6; y++)
        for(int x=0; x<8; x++)
            *dst++ = src[y*stride+x]*16 - (sum+1)/3;
}

void test_dotproduct(const int16_t *weightsi, int *dst, const uint8_t *pix, int stride)
{
    int16_t in[48];
    cast_pixels_test(in, pix, stride);
    dst[0] = dotproduct(weightsi, in, 48);
    dst[1] = dotproduct(weightsi+48, in, 48);
    dst[2] = dotproduct(weightsi+48*2, in, 48);
    dst[3] = dotproduct(weightsi+48*3, in, 48);
}

void test_dotproducts(const int16_t *weightsi, int (*dst)[4], const uint8_t *pix, int stride, int width)
{
    int16_t in[48];
    for(int x=5; x<width; x++) {
        cast_pixels_test(in, pix+2*x-10, stride);
        for(int i=0; i<4; i++)
            dst[x][i] = dotproduct(weightsi+48*i, in, 48);
    }
}

__attribute__((noinline))
static int scale_net(const int16_t *weightsi, const float *weightsf, const uint8_t *pix, int stride)
{
    int16_t in[48];
    float mean, stddev, invstddev;
    cast_pixels_scale(in, pix, stride, &mean, &stddev, &invstddev);
    float tmp[NNS*2];
    for(int i=0; i<NNS*2; i++, weightsi+=48)
        tmp[i] = dotproduct(weightsi, in, 48) * invstddev * weightsf[(i&~3)*2+(i&3)] + weightsf[(i&~3)*2+(i&3)+4];
    for(int i=0; i<NNS; i++)
        tmp[i] = fast_exp2(tmp[i]);
    for(int i=NNS; i<NNS*2; i++)
        tmp[i] = sigmoid(tmp[i]);
    return weighted_average(tmp, tmp+NNS, NNS)*5*stddev+mean+.5f;
}

__attribute__((noinline))
static int test_net(const float *weightsf, const int *dotp, float dc)
{
    float tmp[12];
    for(int i=0; i<4; i++)
        tmp[i] = sigmoid(dotp[i]*weightsf[i] - dc*weightsf[4+i] + weightsf[8+i]);
    for(int i=0; i<4; i++)
        tmp[4+i] = sigmoid(dotproductf(weightsf+12+i*4, tmp, 4) + weightsf[28+i]);
    for(int i=0; i<4; i++)
        tmp[8+i] = dotproductf(weightsf+32+i*8, tmp, 8) + weightsf[64+i];
    return fabsf(fmaxf(tmp[8],tmp[9])) > fabsf(fmaxf(tmp[10],tmp[11]));
}

int test_net_x4(const float *weightsf, int (*dotp)[4], float dc0, float dc1, float dc2, float dc3)
{
    return test_net(weightsf, dotp[0], dc0)
         | test_net(weightsf, dotp[1], dc1)<<8
         | test_net(weightsf, dotp[2], dc2)<<16
         | test_net(weightsf, dotp[3], dc3)<<24;
}

static int merge_test_neighbors(uint8_t *dst, uint16_t *retest, uint8_t *row0, uint8_t *row1, uint8_t *row2, int n, int parity)
{
    uint16_t *pretest = retest;
    int n2 = (n+1)>>1;
    uint8_t *row1offset = row1-1+2*parity;
    for(int x=0; x<n2; x++) {
        dst[2*x] = row1[x];
        dst[2*x+1] = row1[x];
        // FIXME check that this is ok for odd width
        int a = row0[x], b = row1[x], c = row2[x], d = row1offset[x];
        if((a^b)|(b^c)|(c^d))
            *pretest++ = 2*x+parity;
    }
    return pretest - retest;
}

static int merge_test_runlength(uint16_t *retest, uint8_t *src, int n)
{
    uint16_t *pretest = retest;
    for(int x=0; x<n; x++)
        if(!src[x])
            *pretest++ = x;
    return pretest - retest;
}

static void block_sums_core(float *dst, uint16_t *src, int stride, int width)
{
    for(int x=0; x<width; x++)
        dst[x] = src[x] + src[x+stride] + src[x+stride*2] + src[x+stride*3];
}

static void bicubic(uint8_t *dst, uint8_t *src, int stride, int n)
{
    for(int x=0; x<n; x++)
        dst[x] = av_clip_uint8(((src[x+stride]+src[x+stride*2])*38-(src[x]+src[x+stride*3])*6+32)>>6);
}

static void transpose(uint8_t *dst, uint8_t *src, int width, int height, int dstride, int sstride)
{
    for(int x=0; x<width; x++)
        for(int y=0; y<height; y++)
            dst[x*dstride+y] = src[y*sstride+x];
}

#else // USE_ASM
#include "nnedi_dsp.c"
void nnedi_test_dotproduct_sse2(const int16_t *weightsi, int *dst, const uint8_t *pix, int stride);
void nnedi_test_dotproducts_sse2(const int16_t *weightsi, int (*dst)[4], const uint8_t *pix, int stride, int width);
int nnedi_test_net_sse2(const float *weightsf, const int *dotp, float dc);
int nnedi_test_net_x4_ssse3(const float *weightsf, int (*dotp)[4], float dc0, float dc1, float dc2, float dc3);
int nnedi_scale_net_sse2(const int16_t *weightsi, const float *weightsf, const uint8_t *pix, int stride);
void nnedi_block_sums_core_sse2(float *dst, uint16_t *src, int stride, int width);
void nnedi_bicubic_ssse3(uint8_t *dst, uint8_t *src, int stride, int width);
#define test_dotproduct nnedi_test_dotproduct_sse2
#define test_dotproducts nnedi_test_dotproducts_sse2
#define scale_net nnedi_scale_net_sse2
#define test_net_x4 nnedi_test_net_x4_ssse3
#define merge_test_neighbors merge_test_neighbors_ssse3
#define merge_test_runlength merge_test_runlength_sse2
#define block_sums_core nnedi_block_sums_core_sse2
#define bicubic nnedi_bicubic_ssse3
#define transpose transpose_sse2
#endif

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
        block_sums_core(blocks, dst, stride, n);
}

static void pad_row(uint8_t *src, int width, int height, int stride, int y)
{
    if(y<0)
        memcpy(src+y*stride, src+(-1-y)*stride, width);
    if(y>=height)
        memcpy(src+y*stride, src+(2*height-1-y)*stride, width);
    for(int x=1; x<=5; x++)
        src[y*stride-x] = src[y*stride-1+x];
    for(int x=1; x<=6; x++)
        src[y*stride+width-1+x] = src[y*stride+width-x];
}

static struct {
    int initted;
} dsp;

static void init_dsp(void)
{
    if(dsp.initted)
        return;
    dsp.initted = 1;
    uint32_t cpu = 1;
    if(getenv("noasm"))
        cpu = 0;
}

// FIXME cap scaling factors so that intermediate sums don't overflow; or allow 7fff if that works.
// FIXME try multiple scaling factors and mean removals to minimize total quantization error.
// or try forcing everything to the same scaling factor to eliminate a few multiplies from scale_net.
static void munge_test_weights(int16_t *dsti, int16_t *dsti_transpose, float *dstf, const float *src)
{
    for(int j=0; j<4; j++, src+=48) {
        float max = 0;
        int sum = 0;
        for(int i=0; i<48; i++)
            max = fmaxf(max, fabsf(src[i]));
        float scale = 0x3fff/max;
        for(int i=0; i<48; i++)
            sum += dsti[48*j+i] = dsti_transpose[48*j+i/12+i%12*4] = roundf(src[i]*scale);
        dstf[j] = max/(0x3fff*127.5f);
        dstf[j+4] = sum*dstf[j]/48;
    }
    memcpy(dstf+8, src, 60*sizeof(float));

#ifdef USE_ASM
    // transpose weights into the order that asm wants
    int16_t b[48*4];
    int32_t c[24*4];
    float d[16], e[32];
    for(int i=0; i<48*4; i++)
        b[i] = dsti[((i>>3)&3)*48 + (i>>5)*8 + (i&7)];
    for(int i=0; i<24*4; i++)
        c[i] = ((int32_t*)dsti_transpose)[(i&3)*24 + ((i%24)&~3) + ((i+i/24)&3)];
    for(int i=40; i<48; i++)
        FFSWAP(float, dstf[i], dstf[i+8]);
    FFSWAP(float, dstf[65], dstf[66]);
    for(int i=0; i<16; i++)
        d[i] = dstf[12 + (i&3)*4 + ((i+(i>>2))&3)];
    for(int i=0; i<32; i++)
        e[i] = dstf[32 + ((i>>2)&4) + (i&3)*8 + ((i+(i>>2))&3)];
    memcpy(dsti, b, sizeof(b));
    memcpy(dsti_transpose, c, sizeof(c));
    memcpy(dstf+12, d, sizeof(d));
    memcpy(dstf+32, e, sizeof(e));
#else
    memcpy(dsti_transpose, dsti, 48*4*sizeof(*dsti));
#endif
}

static void munge_scale_weights(int16_t *dsti, float *dstf, const float *src)
{
    // the first half of the dotps go through a softmax filter, and thus are invariant to global offsets.
    // remove any offsets they already have, so that the remaining smaller coefs can be represented with higher precision.
    float mean[49] = {0};
    for(int j=0; j<NNS; j++) {
        for(int i=0; i<48; i++)
            mean[i] += src[48*j+i];
        mean[48] += src[48*2*NNS+j];
    }
    for(int i=0; i<49; i++)
        mean[i] /= NNS;
    float src2[49*2*NNS];
    memcpy(src2, src, 49*2*NNS*sizeof(float));
    for(int j=0; j<NNS; j++) {
        for(int i=0; i<48; i++)
            src2[48*j+i] -= mean[i];
        src2[48*2*NNS+j] -= mean[48];
    }
    src = src2;

    // cast input weights to int, scaling them by the largest factor that fits.
    // record that factor so they can be converted back after dotproduct.
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

#ifdef USE_ASM
    // transpose weights into the order that asm wants
    dsti -= 48*2*NNS;
    for(int j=0; j<2*NNS; j+=16) {
        int32_t *a = (int32_t*)(dsti+48*j);
        int32_t b[24*16];
        for(int i=0; i<24*16; i++)
            b[i] = a[96*((i>>2)&3) + 24*(i&3) + 4*(i>>6) + ((i+(i>>4))&3)];
        memcpy(a, b, sizeof(b));
    }
#endif
}

static void upscale_v(uint8_t *dst, uint8_t *src, int width, int height, int dstride, int sstride)
{
    int twidth = width+11;
    int tstride = ALIGN(twidth);
    uint16_t *sum_w12 = memalign(16, 4*tstride*sizeof(uint16_t));
    float *sum_12x4[2] = { memalign(16, tstride*sizeof(float)), memalign(16, tstride*sizeof(float)) };
    uint8_t *tested = memalign(16, 3*tstride+16); // FIXME only needs stride=align(tstride/2+2)
    uint8_t *tested2 = memalign(16, tstride+16);
    uint16_t *retest = memalign(16, (tstride+32)*sizeof(uint16_t));
    int (*test_dotp)[4] = memalign(16, tstride*2*sizeof(int));
    memset(tested, 0, 3*tstride+16);
    tested += 16;
    ALIGNED_16(int16_t test_weights_i[48*4]);
    ALIGNED_16(int16_t test_weights_i_transpose[48*4]);
    ALIGNED_16(float test_weights_f[68]);
    ALIGNED_16(int16_t scale_weights_i[48*2*NNS]);
    ALIGNED_16(float scale_weights_f[4*NNS]);
    munge_test_weights(test_weights_i, test_weights_i_transpose, test_weights_f, test_weights);
    munge_scale_weights(scale_weights_i, scale_weights_f,
        NNS==16 ? scale_weights_8x6x16 : NNS==32 ? scale_weights_8x6x32 : scale_weights_8x6x64);

    for(int y=-2; y<3; y++)
        pad_row(src, width, height, sstride, y);
    uint64_t t0 = read_time();
    for(int y=0; y<3; y++)
        block_sums(NULL, sum_w12, src+(y-1)*sstride-5, width, 12, y+1, tstride);
    for(int y=0, testy=0; y<height; y++) {
        pad_row(src, width, height, sstride, y+3);
        for(; testy<=y+1 && testy<height; testy++) {
            block_sums(sum_12x4[testy&1], sum_w12, src+(testy+2)*sstride-5, width, 12, testy&3, tstride);
            uint8_t *pix = src+(testy-1)*sstride-5+!(testy&1);
            int end = (width+(testy&1))>>1;
            test_dotproducts(test_weights_i_transpose, test_dotp, pix, sstride, end+5);
            uint8_t *pt = tested+(testy%3)*tstride;
            float *dc = sum_12x4[testy&1]+!(testy&1);
            for(int x=0; x<end; x+=4)
                *(uint32_t*)(pt+x) = test_net_x4(test_weights_f, test_dotp+x+5, dc[x*2], dc[x*2+2], dc[x*2+4], dc[x*2+6]);
            pt[end] = 0;
        }
        if(y==height-1) memset(tested+(y+1)%3*tstride, 0, tstride);
        int nretest = merge_test_neighbors(tested2, retest, tested+(y+2)%3*tstride, tested+y%3*tstride, tested+(y+1)%3*tstride, width, y&1);
        uint8_t *pix = src+(y-1)*sstride-5;
        for(int i=0; i<nretest; i++)
            test_dotproduct(test_weights_i, test_dotp[i], pix+retest[i], sstride);
        float *dc = sum_12x4[y&1];
        retest[nretest] = retest[nretest+1] = retest[nretest+2] = width+1;
        for(int i=0; i<nretest; i+=4) {
            uint32_t v = test_net_x4(test_weights_f, test_dotp+i, dc[retest[i+0]], dc[retest[i+1]], dc[retest[i+2]], dc[retest[i+3]]);
            tested2[retest[i+0]] = v;
            tested2[retest[i+1]] = v>>8;
            tested2[retest[i+2]] = v>>16;
            tested2[retest[i+3]] = v>>24;
        }
        if(dst != src)
            memcpy(dst+y*2*dstride, src+y*sstride, width);
        bicubic(dst+(y*2+1)*dstride, src+(y-1)*sstride, sstride, width);
        pix = src+(y-2)*sstride-3;
        uint8_t *dpix = dst+(y*2+1)*dstride;
        nretest = merge_test_runlength(retest, tested2, width);
        for(int i=0; i<nretest; i++) {
            int x = retest[i];
            int v = scale_net(scale_weights_i, scale_weights_f, pix+x, sstride);
            dpix[x] = av_clip_uint8(v);
        }
    }
    uint64_t t1 = read_time();
    printf("%d Mcycles\n", (int)((t1-t0)/1000000));
    free(sum_w12);
    free(sum_12x4[0]);
    free(sum_12x4[1]);
    free(tested-16);
    free(tested2);
    free(retest);
    free(test_dotp);
}

void upscale_2x(uint8_t *dst, uint8_t *src, int width, int height, int dstride, int sstride)
{
    init_dsp();
    int h1 = width;
    int w1 = height;
    int s1 = ALIGN(w1+11)*2;
    uint8_t *b1 = memalign(16, (h1+5)*s1+16);
    uint8_t *p1 = b1+s1*2+16;
    transpose(p1, src, width, height, s1, sstride);
    upscale_v(p1, p1, w1, h1, s1/2, s1);
    int h2 = w1;
    int w2 = h1*2;
    int s2 = ALIGN(w2+11);
    uint8_t *b2 = memalign(16, (h2+5)*s2+16);
    uint8_t *p2 = b2+s2*2+16;
    transpose(p2, p1, h2, w2, s2, s1/2);
    upscale_v(dst, p2, w2, h2, dstride, s2);
    free(b1);
    free(b2);
}
