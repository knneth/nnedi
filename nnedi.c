/**
 * NNEDI - neural net edge directed interpolation
 *
 * Copyright (C) 2010 Loren Merritt
 * Algorithm designed by Tritical
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <malloc.h>
#include <pthread.h>
#include <string.h>
#include <libavutil/avutil.h>
#include "nnedi.h"

#define MAX_NNS 256

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
            *dst++ = src[y*stride+x];
}

static void test_dotproduct_c(const int16_t *weightsi, int *dst, const uint8_t *pix, intptr_t stride)
{
    int16_t in[48];
    cast_pixels_test(in, pix, stride);
    dst[0] = dotproduct(weightsi, in, 48);
    dst[1] = dotproduct(weightsi+48, in, 48);
    dst[2] = dotproduct(weightsi+48*2, in, 48);
    dst[3] = dotproduct(weightsi+48*3, in, 48);
}

static void test_dotproducts_c(const int16_t *weightsi, int (*dst)[4], const uint8_t *pix, intptr_t stride, int width)
{
    int16_t in[48];
    for(int x=5; x<width; x++) {
        cast_pixels_test(in, pix+2*x-10, stride);
        for(int i=0; i<4; i++)
            dst[x][i] = dotproduct(weightsi+48*i, in, 48);
    }
}

#define SCALE_NET_C(NNSI) \
static int scale_net##NNSI##_c(const int16_t *weights, const uint8_t *pix, intptr_t stride)\
{\
    int nns = 16<<NNSI;\
    const float *weightsf = (const float*)(weights+48*2*nns);\
    int16_t in[48];\
    float mean, stddev, invstddev;\
    cast_pixels_scale(in, pix, stride, &mean, &stddev, &invstddev);\
    float tmp[nns*2];\
    for(int i=0; i<nns*2; i++, weights+=48)\
        tmp[i] = dotproduct(weights, in, 48) * invstddev * weightsf[(i&~3)*2+(i&3)] + weightsf[(i&~3)*2+(i&3)+4];\
    for(int i=0; i<nns; i++)\
        tmp[i] = fast_exp2(tmp[i]);\
    for(int i=nns; i<nns*2; i++)\
        tmp[i] = sigmoid(tmp[i]);\
    return av_clip_uint8(weighted_average(tmp, tmp+nns, nns)*5*stddev+mean+.5f);\
}\
\
static void scale_nets##NNSI##_c(const int16_t *weights, const uint8_t *pix, intptr_t stride, uint8_t *dst, const uint16_t *offsets, int n)\
{\
    for(int i=0; i<n; i++)\
        dst[offsets[i]] = scale_net##NNSI##_c(weights, pix+offsets[i], stride);\
}

SCALE_NET_C(0)
SCALE_NET_C(1)
SCALE_NET_C(2)
SCALE_NET_C(3)
SCALE_NET_C(4)
static void (*scale_nets_tab_c[])(const int16_t *weights, const uint8_t *pix, intptr_t stride, uint8_t *dst, const uint16_t *offsets, int n) = {
    scale_nets0_c, scale_nets1_c, scale_nets2_c, scale_nets3_c, scale_nets4_c
};

static int test_net_c(const float *weightsf, const int *dotp)
{
    float tmp[12];
    for(int i=0; i<4; i++)
        tmp[i] = sigmoid(dotp[i]*weightsf[i] + weightsf[4+i]);
    for(int i=0; i<4; i++)
        tmp[4+i] = sigmoid(dotproductf(weightsf+8+i*4, tmp, 4) + weightsf[24+i]);
    for(int i=0; i<4; i++)
        tmp[8+i] = dotproductf(weightsf+28+i*8, tmp, 8) + weightsf[60+i];
    return fabsf(fmaxf(tmp[8],tmp[9])) > fabsf(fmaxf(tmp[10],tmp[11]));
}

static int test_net_x4_c(const float *weightsf, int (*dotp)[4])
{
    return test_net_c(weightsf, dotp[0])
         | test_net_c(weightsf, dotp[1])<<8
         | test_net_c(weightsf, dotp[2])<<16
         | test_net_c(weightsf, dotp[3])<<24;
}

static int merge_test_neighbors_c(uint8_t *dst, uint16_t *retest, uint8_t *row0, uint8_t *row1, uint8_t *row2, int n, int parity)
{
    uint16_t *pretest = retest;
    int n2 = (n+1)>>1;
    uint8_t *row1offset = row1-1+2*parity;
    for(int x=0; x<n2; x++) {
        dst[2*x] = row1[x];
        dst[2*x+1] = row1[x];
        int a = row0[x], b = row1[x], c = row2[x], d = row1offset[x];
        if((a^b)|(b^c)|(c^d))
            *pretest++ = 2*x+parity;
    }
    return pretest - retest;
}

static int merge_test_runlength_c(uint16_t *retest, uint8_t *src, int n)
{
    uint16_t *pretest = retest;
    for(int x=0; x<n; x++)
        if(!src[x])
            *pretest++ = x;
    return pretest - retest;
}

static void bicubic_c(uint8_t *dst, uint8_t *src, intptr_t stride, int n)
{
    for(int x=0; x<n; x++)
        dst[x] = av_clip_uint8(((src[x+stride]+src[x+stride*2])*38-(src[x]+src[x+stride*3])*6+32)>>6);
}

static void transpose_c(uint8_t *dst, uint8_t *src, int width, int height, int dstride, int sstride)
{
    for(int x=0; x<width; x++)
        for(int y=0; y<height; y++)
            dst[x*dstride+y] = src[y*sstride+x];
}

static void munge_test_weights(int16_t *dsti, int16_t *dsti_transpose, float *dstf, const float *src);
static void munge_scale_weights(int16_t *dsti, float *dstf, const float *src);

#ifdef ARCH_X86
#include "nnedi_asm.c"
void nnedi_test_dotproduct_sse2(const int16_t *weightsi, int *dst, const uint8_t *pix, intptr_t stride);
void nnedi_test_dotproducts_sse2(const int16_t *weightsi, int (*dst)[4], const uint8_t *pix, intptr_t stride, int width);
int nnedi_test_net_sse2(const float *weightsf, const int *dotp);
int nnedi_test_net_x4_sse2(const float *weightsf, int (*dotp)[4]);
int nnedi_test_net_x4_ssse3(const float *weightsf, int (*dotp)[4]);
extern void (*nnedi_scale_nets_tab_sse2[])(const int16_t *weights, const uint8_t *pix, intptr_t stride, uint8_t *dst, const uint16_t *offsets, int n);
void nnedi_bicubic_sse2(uint8_t *dst, uint8_t *src, intptr_t stride, int width);
void nnedi_bicubic_ssse3(uint8_t *dst, uint8_t *src, intptr_t stride, int width);
#endif

static struct {
    int cpu;
    int nns, nnsi;

    void (*test_dotproduct)(const int16_t *weightsi, int *dst, const uint8_t *pix, intptr_t stride);
    void (*test_dotproducts)(const int16_t *weightsi, int (*dst)[4], const uint8_t *pix, intptr_t stride, int width);
    int  (*test_net_x4)(const float *weightsf, int (*dotp)[4]);
    void (*scale_nets)(const int16_t *weights, const uint8_t *pix, intptr_t stride, uint8_t *dst, const uint16_t *offsets, int n);
    int  (*merge_test_neighbors)(uint8_t *dst, uint16_t *retest, uint8_t *row0, uint8_t *row1, uint8_t *row2, int n, int parity);
    int  (*merge_test_runlength)(uint16_t *retest, uint8_t *src, int n);
    void (*bicubic)(uint8_t *dst, uint8_t *src, intptr_t stride, int width);
    void (*transpose)(uint8_t *dst, uint8_t *src, int width, int height, int dstride, int sstride);

    ALIGNED_16(int16_t test_weights_i[48*4]);
    ALIGNED_16(int16_t test_weights_i_transpose[48*4]);
    ALIGNED_16(float test_weights_f[64]);
    ALIGNED_16(int16_t scale_weights[(48*2+4*sizeof(float)/sizeof(int16_t))*MAX_NNS]);
} dsp;

void nnedi_config(int nns)
{
    dsp.cpu = 1;
    if(getenv("noasm"))
        dsp.cpu = 0;
    dsp.nnsi = av_clip(nns, 0, 4);
    dsp.nns = 16<<dsp.nnsi;

    dsp.test_dotproduct = test_dotproduct_c;
    dsp.test_dotproducts = test_dotproducts_c;
    dsp.scale_nets = scale_nets_tab_c[dsp.nnsi];
    dsp.test_net_x4 = test_net_x4_c;
    dsp.merge_test_neighbors = merge_test_neighbors_c;
    dsp.merge_test_runlength = merge_test_runlength_c;
    dsp.bicubic = bicubic_c;
    dsp.transpose = transpose_c;

#ifdef ARCH_X86
    if(dsp.cpu) {
        dsp.test_dotproduct = nnedi_test_dotproduct_sse2;
        dsp.test_dotproducts = nnedi_test_dotproducts_sse2;
        dsp.scale_nets = nnedi_scale_nets_tab_sse2[dsp.nnsi];
        dsp.test_net_x4 = nnedi_test_net_x4_sse2;
        dsp.merge_test_neighbors = merge_test_neighbors_sse2;
        dsp.merge_test_runlength = merge_test_runlength_sse2;
        dsp.bicubic = nnedi_bicubic_sse2;
        dsp.transpose = transpose_sse2;
    }

    if(dsp.cpu) {
        dsp.test_net_x4 = nnedi_test_net_x4_ssse3;
        dsp.merge_test_neighbors = merge_test_neighbors_ssse3;
        dsp.bicubic = nnedi_bicubic_ssse3;
    }
#else
    dsp.cpu = 0;
#endif

    munge_test_weights(dsp.test_weights_i, dsp.test_weights_i_transpose, dsp.test_weights_f, nnedi_test_weights);
    munge_scale_weights(dsp.scale_weights, (float*)(dsp.scale_weights+48*2*dsp.nns), nnedi_scale_weights_8x6xN[dsp.nnsi]);
}

static void bicubic(uint8_t *dst, uint8_t *src, intptr_t stride, int n)
{
    if(n >= 16)
        dsp.bicubic(dst, src, stride, n&~15);
    if(n & 15)
        bicubic_c(dst+(n&~15), src+(n&~15), stride, n&15);
}

static void pad_row(uint8_t *src, int width, int height, int stride, int y, int pad_top, int pad_bottom)
{
    if(y<0 && pad_top)
        memcpy(src+y*stride, src+(-1-y)*stride, width);
    if(y>=height && pad_bottom)
        memcpy(src+y*stride, src+(2*height-1-y)*stride, width);
    for(int x=1; x<=5; x++)
        src[y*stride-x] = src[y*stride-1+x];
    for(int x=1; x<=6; x++)
        src[y*stride+width-1+x] = src[y*stride+width-x];
}

static void remove_mean(float *x, int n)
{
    float mean = 0;
    for(int i=0; i<n; i++)
        mean += x[i];
    mean /= n;
    for(int i=0; i<n; i++)
        x[i] -= mean;
}

static float quantize_weights(int16_t *dst, const float *src, int n)
{
    float max = 0;
    for(int i=0; i<n; i++)
        max = fmaxf(max, fabsf(src[i]));
    float scale = 0x7fff/max;
    for(int i=0; i<n; i++)
        dst[i] = roundf(src[i]*scale);
    return 1.f/scale;
}

// FIXME try multiple scaling factors and mean removals to minimize total quantization error.
// or try forcing everything to the same scaling factor to eliminate a few multiplies from scale_net.
static void munge_test_weights(int16_t *dsti, int16_t *dsti_transpose, float *dstf, const float *src)
{
    // convolve pixel-wise mean removal into the weights
    float src2[252];
    memcpy(src2, src, sizeof(src2));
    src = src2;
    for(int j=0; j<4; j++)
        remove_mean(src2+48*j, 48);

    // cast input weights to int, scaling them by the largest factor that fits.
    // record that factor so they can be converted back after dotproduct.
    for(int j=0; j<4; j++, src+=48) {
        dstf[j] = quantize_weights(dsti+48*j, src, 48) * (1/127.f);
        for(int i=0; i<48; i++)
            dsti_transpose[48*j+i/12+i%12*4] = dsti[48*j+i];
    }
    memcpy(dstf+4, src, 60*sizeof(float));

    if(dsp.cpu) {
        // transpose weights into the order that asm wants
        int16_t b[48*4];
        int32_t c[24*4];
        float d[16], e[32];
        for(int i=0; i<48*4; i++)
            b[i] = dsti[48*((i>>3)&3) + 8*(i>>5) + (i&7)];
        for(int i=0; i<24*4; i++)
            c[i] = ((int32_t*)dsti_transpose)[24*(i&3) + ((i%24)&~3) + ((i+i/24)&3)];
        for(int i=36; i<44; i++)
            FFSWAP(float, dstf[i], dstf[i+8]);
        FFSWAP(float, dstf[61], dstf[62]);
        for(int i=0; i<16; i++)
            d[i] = dstf[8 + 4*(i&3) + ((i+(i>>2))&3)];
        for(int i=0; i<32; i++)
            e[i] = dstf[28 + ((i>>2)&4) + 8*(i&3) + ((i+(i>>2))&3)];
        memcpy(dsti, b, sizeof(b));
        memcpy(dsti_transpose, c, sizeof(c));
        memcpy(dstf+8, d, sizeof(d));
        memcpy(dstf+28, e, sizeof(e));
    } else {
        memcpy(dsti_transpose, dsti, 48*4*sizeof(*dsti));
    }
}

static void munge_scale_weights(int16_t *dsti, float *dstf, const float *src)
{
    // the first half of the dotps go through a softmax filter, and thus are invariant to global offsets.
    // remove any offsets they already have, so that the remaining smaller coefs can be represented with higher precision.
    float means[49] = {0};
    for(int j=0; j<dsp.nns; j++) {
        for(int i=0; i<48; i++)
            means[i] += src[48*j+i];
        means[48] += src[48*2*dsp.nns+j];
    }
    for(int i=0; i<49; i++)
        means[i] /= dsp.nns;
    float src2[49*2*MAX_NNS];
    memcpy(src2, src, 49*2*dsp.nns*sizeof(float));
    src = src2;
    for(int j=0; j<dsp.nns; j++) {
        for(int i=0; i<48; i++)
            src2[48*j+i] -= means[i];
        src2[48*2*dsp.nns+j] -= means[48];
    }

    // convolve pixel-wise mean removal into the weights
    for(int j=0; j<2*dsp.nns; j++)
        remove_mean(src2+48*j, 48);

    // cast input weights to int, scaling them by the largest factor that fits.
    // record that factor so they can be converted back after dotproduct.
    for(int j=0; j<2*dsp.nns; j++) {
        dstf[2*(j&~3)+(j&3)] = quantize_weights(dsti+48*j, src+48*j, 48);
        dstf[2*(j&~3)+(j&3)+4] = src[48*2*dsp.nns+j];
    }

    // weights are defined in terms of exp(), but exp2() is easier to calculate
    for(int j=0; j<2*dsp.nns; j++)
        dstf[j] *= (float)M_LOG2E;

    if(dsp.cpu) {
        // transpose weights into the order that asm wants
        for(int j=0; j<2*dsp.nns; j+=16) {
            int32_t *a = (int32_t*)(dsti+48*j);
            int32_t b[24*16];
            for(int i=0; i<24*16; i++)
                b[i] = a[96*((i>>2)&3) + 24*(i&3) + 4*(i>>6) + ((i+(i>>4))&3)];
            memcpy(a, b, sizeof(b));
        }
    }
}

static void upscale_v(uint8_t *dst, uint8_t *src, int width, int height, int dstride, int sstride, int pad_top, int pad_bottom)
{
    int twidth = width+11;
    int tstride = FFALIGN(twidth, 16);
    uint8_t *tested = memalign(16, 3*tstride+16); // FIXME only needs stride=align(tstride/2+2)
    uint8_t *tested2 = memalign(16, tstride+16);
    uint16_t *retest = memalign(16, (tstride+32)*sizeof(uint16_t));
    int (*test_dotp)[4] = memalign(16, (width+17)*2*sizeof(int));
    memset(tested, 0, 3*tstride+16);
    tested += 16;

    for(int y=-2; y<3; y++)
        pad_row(src, width, height, sstride, y, pad_top, pad_bottom);
    for(int y=0, testy=pad_top-1; y<height; y++) {
        pad_row(src, width, height, sstride, y+3, pad_top, pad_bottom);
        for(; testy<=y+1 && (testy<height || !pad_bottom); testy++) {
            uint8_t *pix = src+(testy-1)*sstride-5+!(testy&1);
            int end = (width+(testy&1))>>1;
            dsp.test_dotproducts(dsp.test_weights_i_transpose, test_dotp, pix, sstride, end+5);
            uint8_t *pt = tested+((testy+3)%3)*tstride;
            for(int x=0; x<end; x+=4)
                *(uint32_t*)(pt+x) = dsp.test_net_x4(dsp.test_weights_f, test_dotp+x+5);
            pt[end] = pt[end+1] = pt[end+2] = 0;
        }
        if(y==height-1 && pad_bottom) memset(tested+(y+1)%3*tstride, 0, tstride);
        int nretest = dsp.merge_test_neighbors(tested2, retest, tested+(y+2)%3*tstride, tested+y%3*tstride, tested+(y+1)%3*tstride, width, y&1);
        uint8_t *pix = src+(y-1)*sstride-5;
        for(int i=0; i<nretest; i++)
            dsp.test_dotproduct(dsp.test_weights_i, test_dotp[i], pix+retest[i], sstride);
        retest[nretest] = retest[nretest+1] = retest[nretest+2] = width+1;
        for(int i=0; i<nretest; i+=4) {
            uint32_t v = dsp.test_net_x4(dsp.test_weights_f, test_dotp+i);
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
        nretest = dsp.merge_test_runlength(retest, tested2, width);
        dsp.scale_nets(dsp.scale_weights, pix, sstride, dpix, retest, nretest);
    }
    free(tested-16);
    free(tested2);
    free(retest);
    free(test_dotp);
}

static struct {
    pthread_mutex_t mutex;
    int njobs;
    int progress;
    intptr_t (*args)[8];
} jobs;

static void upscale_worker(void)
{
    for(;;) {
        int i;
        pthread_mutex_lock(&jobs.mutex);
        i = jobs.progress++;
        pthread_mutex_unlock(&jobs.mutex);
        if(i >= jobs.njobs)
            return;
        intptr_t *a = jobs.args[i];
        upscale_v((uint8_t*)a[0], (uint8_t*)a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
    }
}

static void upscale_threaded(uint8_t *dst, uint8_t *src, int width, int height, int dstride, int sstride)
{
    int njobs = FFMIN(8, height/4);
    int nthreads = FFMIN(2, njobs);
    if(nthreads == 1) {
        upscale_v(dst, src, width, height, dstride, sstride, 1, 1);
        return;
    }

    intptr_t args[njobs][8];
    pthread_mutex_init(&jobs.mutex, NULL);
    jobs.njobs = njobs;
    jobs.progress = 0;
    jobs.args = args;
    for(int i=0; i<njobs; i++) {
        int y0 = FFALIGN(height*i/njobs, 2);
        int y1 = FFALIGN(height*(i+1)/njobs, 2);
        memcpy(args[i], (intptr_t[]){(intptr_t)(dst+y0*2*dstride), (intptr_t)(src+y0*sstride), width, y1-y0, dstride, sstride, i==0, i==njobs-1}, sizeof(args[i]));
    }

    pthread_t handle[nthreads];
    for(int i=0; i<nthreads; i++)
        if(pthread_create(handle+i, NULL, (void*)upscale_worker, NULL)) {
            fprintf(stderr, "[nnedi] pthread_create failed\n");
            return;
        }
    for(int i=0; i<nthreads; i++)
        if(pthread_join(handle[i], NULL))
            fprintf(stderr, "[nnedi] pthread_join failed\n");
    pthread_mutex_destroy(&jobs.mutex);
}

void nnedi_upscale_2x(uint8_t *dst, uint8_t *src, int width, int height, int dstride, int sstride)
{
    int h1 = width;
    int w1 = height;
    int s1 = FFALIGN(w1+11, 16)*2;
    uint8_t *b1 = memalign(16, (h1+5)*s1+16);
    uint8_t *p1 = b1+s1*2+16;
    dsp.transpose(p1, src, width, height, s1, sstride);
    upscale_threaded(p1, p1, w1, h1, s1/2, s1);
    int h2 = w1;
    int w2 = h1*2;
    int s2 = FFALIGN(w2+11, 16);
    uint8_t *b2 = memalign(16, (h2+5)*s2+16);
    uint8_t *p2 = b2+s2*2+16;
    dsp.transpose(p2, p1, h2, w2, s2, s1/2);
    upscale_threaded(dst, p2, w2, h2, dstride, s2);
    free(b1);
    free(b2);
}
