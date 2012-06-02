#include <inttypes.h>

// NNEDI is reentrant, and you can even run multiple threads simultaneously on the same nnedi_t.
typedef struct nnedi_t nnedi_t;
nnedi_t *nnedi_config(int nns, int threads);
void nnedi_upscale_2x(nnedi_t *ctx, uint8_t *dst, uint8_t *src, int width, int height, int dstride, int sstride);
