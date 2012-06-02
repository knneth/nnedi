#include <inttypes.h>

#define MAX_NNS 256

#define ALIGNED_16(x) __attribute__((aligned(16))) x

struct nnedi_t {
    int cpu;
    int threads;
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
};

extern const float nnedi_test_weights[];
extern const float *const nnedi_scale_weights_8x6xN[];
