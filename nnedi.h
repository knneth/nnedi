#include <inttypes.h>

#define ALIGN(x) (((x)+15)&~15)
#define ALIGNED_16(x) __attribute__((aligned(16))) x

extern const float nnedi_test_weights[];
extern const float *const nnedi_scale_weights_8x6xN[];

void nnedi_upscale_2x(uint8_t *dst, uint8_t *src, int width, int height, int dstride, int sstride);
void nnedi_config(int nns);
