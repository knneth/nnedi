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

static inline int merge_test_neighbors_asm(uint8_t *dst, uint16_t *retest, uint8_t *row0, uint8_t *row1, uint8_t *row2, int n, int parity, int ssse3)
{
  uint16_t *pretest = retest;
  int n2 = (n + 1) >> 1;
  int n32 = (n + 31) >> 5;
  int n64 = (n + 63) >> 6;
  uint16_t *masks = retest + n2 - n32;
#define MERGE(load_row1offset) {\
    intptr_t x = -n2;\
    asm volatile(\
                 "1: \n"\
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
                 :"eax", "memory"\
                );\
    masks -= n32;\
  }

  if (ssse3) {
    if (parity)
      MERGE("movdqa 16(%3,%1), %%xmm1 \n"
            "palignr $1, %%xmm0, %%xmm1 \n")
      else
        MERGE("movdqa %%xmm0, %%xmm1 \n"
              "palignr $15, -16(%3,%1), %%xmm1 \n")
      } else {
    if (parity)
      MERGE("movdqu 1(%3,%1), %%xmm1 \n")
      else
        MERGE("movdqu -1(%3,%1), %%xmm1 \n")
      }

#undef MERGE
  masks[n32] = 0xffff;
  uint32_t *masks2 = (uint32_t *)masks;

  for (int x = 0; x < n64; x++) {
    int x2 = x * 64 - 2 + parity;
    uint32_t mask = ~masks2[x];

    while (mask) {
      int tz = __builtin_ctz(mask);
      x2 += (tz + 1) * 2;
      *pretest++ = x2;
      mask >>= tz;
      mask >>= 1;
    }
  }

  while (pretest > retest && pretest[-1] >= n) {
    pretest--;
  }

  return pretest - retest;
}

static int merge_test_neighbors_sse2(uint8_t *dst, uint16_t *retest, uint8_t *row0, uint8_t *row1, uint8_t *row2, int n, int parity)
{
  return merge_test_neighbors_asm(dst, retest, row0, row1, row2, n, parity, 0);
}

static int merge_test_neighbors_ssse3(uint8_t *dst, uint16_t *retest, uint8_t *row0, uint8_t *row1, uint8_t *row2, int n, int parity)
{
  return merge_test_neighbors_asm(dst, retest, row0, row1, row2, n, parity, 1);
}

static int merge_test_runlength_sse2(uint16_t *retest, uint8_t *src, int n)
{
  uint16_t *pretest = retest;
  memset(src + n, 1, (-n) & 31);

  for (int x = 0; x < n; x += 32) {
    uint32_t mask, m2;
    asm volatile(
      "movdqa %2, %%xmm0 \n"
      "movdqa %3, %%xmm1 \n"
      "psllw  $7, %%xmm0 \n"
      "psllw  $7, %%xmm1 \n"
      "pmovmskb %%xmm0, %0 \n"
      "pmovmskb %%xmm1, %1 \n"
      :"=r"(mask), "=r"(m2)
      :"m"(src[x]), "m"(src[x+16])
    );
    mask += m2 << 16;
    mask = ~mask;
    int x2 = x - 1;

    while (mask) {
      int tz = __builtin_ctz(mask);
      x2 += tz + 1;
      *pretest++ = x2;
      mask >>= tz;
      mask >>= 1;
    }
  }

  return pretest - retest;
}

static void transpose8x8_sse2(uint8_t *dst, uint8_t *src, intptr_t dstride, intptr_t sstride)
{
  asm volatile(
    "movq   (%1),      %%xmm0 \n"
    "movq   (%1,%4),   %%xmm1 \n"
    "movq   (%1,%4,2), %%xmm2 \n"
    "movq   (%1,%5),   %%xmm3 \n"
    "lea    (%1,%4,4), %1     \n"
    "movhps (%1),      %%xmm0 \n"
    "movhps (%1,%4),   %%xmm1 \n"
    "movhps (%1,%4,2), %%xmm2 \n"
    "movhps (%1,%5),   %%xmm3 \n"

    "movdqa    %%xmm0, %%xmm4 \n"
    "movdqa    %%xmm2, %%xmm5 \n"
    "punpcklbw %%xmm1, %%xmm0 \n"
    "punpckhbw %%xmm1, %%xmm4 \n"
    "punpcklbw %%xmm3, %%xmm2 \n"
    "punpckhbw %%xmm3, %%xmm5 \n"

    "movdqa    %%xmm0, %%xmm1 \n"
    "movdqa    %%xmm4, %%xmm3 \n"
    "punpcklwd %%xmm2, %%xmm0 \n"
    "punpckhwd %%xmm2, %%xmm1 \n"
    "punpcklwd %%xmm5, %%xmm4 \n"
    "punpckhwd %%xmm5, %%xmm3 \n"

    "movdqa    %%xmm0, %%xmm2 \n"
    "movdqa    %%xmm1, %%xmm5 \n"
    "punpckldq %%xmm4, %%xmm0 \n"
    "punpckhdq %%xmm4, %%xmm2 \n"
    "punpckldq %%xmm3, %%xmm1 \n"
    "punpckhdq %%xmm3, %%xmm5 \n"

    "lea    (%0,%2,4), %1     \n"
    "movq   %%xmm0, (%0)      \n"
    "movhps %%xmm0, (%0,%2)   \n"
    "movq   %%xmm2, (%0,%2,2) \n"
    "movhps %%xmm2, (%0,%3)   \n"
    "movq   %%xmm1, (%1)      \n"
    "movhps %%xmm1, (%1,%2)   \n"
    "movq   %%xmm5, (%1,%2,2) \n"
    "movhps %%xmm5, (%1,%3)   \n"

    :"+&r"(dst), "+&r"(src)
    :"r"(dstride), "r"(dstride*3),
    "r"(sstride), "r"(sstride*3)
  );
}

static void transpose_sse2(uint8_t *dst, uint8_t *src, int width, int height, int dstride, int sstride)
{
  for (int y = 0; y < height - 31; y += 32)
    for (int x = 0; x < width - 7; x += 8) {
      transpose8x8_sse2(dst + x * dstride + y + 0, src + (y + 0)*sstride + x, dstride, sstride);
      transpose8x8_sse2(dst + x * dstride + y + 8, src + (y + 8)*sstride + x, dstride, sstride);
      transpose8x8_sse2(dst + x * dstride + y + 16, src + (y + 16)*sstride + x, dstride, sstride);
      transpose8x8_sse2(dst + x * dstride + y + 24, src + (y + 24)*sstride + x, dstride, sstride);
    }

  for (int y = height&~31; y < height - 7; y += 8)
    for (int x = 0; x < width - 7; x += 8) {
      transpose8x8_sse2(dst + x * dstride + y, src + y * sstride + x, dstride, sstride);
    }

  if (width & 7)
    for (int y = 0; y < (height&~7); y++)
      for (int x = width&~7; x < width; x++) {
        dst[x * dstride + y] = src[y * sstride + x];
      }

  if (height & 7)
    for (int x = 0; x < width; x++)
      for (int y = height&~7; y < height; y++) {
        dst[x * dstride + y] = src[y * sstride + x];
      }
}
