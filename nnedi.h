#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

// NNEDI is reentrant, and you can even run multiple threads simultaneously on the same nnedi_t.
  typedef struct nnedi_t nnedi_t;
  nnedi_t *nnedi_config(int nns, int threads);
  void nnedi_upscale_2x(nnedi_t *ctx, uint8_t *dst, uint8_t *src, int width, int height, int dstride, int sstride);

#ifdef __cplusplus
}
namespace cimg_library
{
template<typename T> struct CImg;
}
cimg_library::CImg<uint8_t> nnedi_upscale_2x_cimg(cimg_library::CImg<uint8_t> src, int nns = 4, int threads = 0);
#endif
