#define ALIGNED_16(x) __attribute__((aligned(16))) x
#define ALIGNED_32(x) __attribute__((aligned(32))) x
extern const float nnedi_test_weights[];
extern const float *const nnedi_scale_weights_8x6xN[];
