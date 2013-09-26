#define NNEDI_CPU_CACHELINE_32    0x0000001  /* avoid memory loads that span the border between two cachelines */
#define NNEDI_CPU_CACHELINE_64    0x0000002  /* 32/64 is the size of a cacheline in bytes */
#define NNEDI_CPU_ALTIVEC         0x0000004
#define NNEDI_CPU_MMX             0x0000008
#define NNEDI_CPU_MMX2            0x0000010  /* MMX2 aka MMXEXT aka ISSE */
#define NNEDI_CPU_SSE             0x0000020
#define NNEDI_CPU_SSE2            0x0000040
#define NNEDI_CPU_SSE2_IS_SLOW    0x0000080  /* avoid most SSE2 functions on Athlon64 */
#define NNEDI_CPU_SSE2_IS_FAST    0x0000100  /* a few functions are only faster on Core2 and Phenom */
#define NNEDI_CPU_SSE3            0x0000200
#define NNEDI_CPU_SSSE3           0x0000400
#define NNEDI_CPU_SHUFFLE_IS_FAST 0x0000800  /* Penryn, Nehalem, and Phenom have fast shuffle units */
#define NNEDI_CPU_STACK_MOD4      0x0001000  /* if stack is only mod4 and not mod16 */
#define NNEDI_CPU_SSE4            0x0002000  /* SSE4.1 */
#define NNEDI_CPU_SSE42           0x0004000  /* SSE4.2 */
#define NNEDI_CPU_SSE_MISALIGN    0x0008000  /* Phenom support for misaligned SSE instruction arguments */
#define NNEDI_CPU_LZCNT           0x0010000  /* Phenom support for "leading zero count" instruction. */
#define NNEDI_CPU_ARMV6           0x0020000
#define NNEDI_CPU_NEON            0x0040000  /* ARM NEON */
#define NNEDI_CPU_FAST_NEON_MRC   0x0080000  /* Transfer from NEON to ARM register is fast (Cortex-A9) */
#define NNEDI_CPU_SLOW_CTZ        0x0100000  /* BSR/BSF x86 instructions are really slow on some CPUs */
#define NNEDI_CPU_SLOW_ATOM       0x0200000  /* The Atom just sucks */
#define NNEDI_CPU_AVX             0x0400000  /* AVX support: requires OS support even if YMM registers
* aren't used. */
#define NNEDI_CPU_XOP             0x0800000  /* AMD XOP */
#define NNEDI_CPU_FMA4            0x1000000  /* AMD FMA4 */
#define NNEDI_CPU_AVX2            0x2000000  /* AVX2 */
#define NNEDI_CPU_FMA3            0x4000000  /* Intel FMA3 */
#define NNEDI_CPU_BMI1            0x8000000  /* BMI1 */
#define NNEDI_CPU_BMI2           0x10000000  /* BMI2 */
#define NNEDI_CPU_TBM            0x20000000  /* AMD TBM */

uint32_t nnedi_cpu_detect(void);
int nnedi_cpu_num_processors(void);
