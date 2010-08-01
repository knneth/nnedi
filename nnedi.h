#include <inttypes.h>

#define ALIGNED_16(x) __attribute__((aligned(16))) x

extern const float test_weights[];
extern const float scale_weights_8x6x16[];
extern const float scale_weights_8x6x32[];
extern const float scale_weights_8x6x64[];

void upscale_v(uint8_t *dst, uint8_t *src, int width, int height, int dstride, int sstride);
