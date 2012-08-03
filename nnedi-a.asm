; NNEDI - neural net edge directed interpolation
;
; Copyright (C) 2010 Loren Merritt
; Algorithm designed by Tritical
;
; This library is free software; you can redistribute it and/or
; modify it under the terms of the GNU Lesser General Public
; License as published by the Free Software Foundation; either
; version 2.1 of the License, or (at your option) any later version.
;
; This library is distributed in the hope that it will be useful,
; but WITHOUT ANY WARRANTY; without even the implied warranty of
; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
; Lesser General Public License for more details.
;
; You should have received a copy of the GNU Lesser General Public
; License along with this library; if not, write to the Free Software
; Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA

%include "x86inc.asm"

SECTION_RODATA 32
ps_exp_bias0: times 8 dd 12582912.0 ; 3<<22
ps_exp_bias1: times 8 dd 12583040.0 ; (3<<22)+128
ps_exp_c0:    times 8 dd 1.00035
ps_exp_c1:    times 8 dd 0.701277797 ; 1.01173*M_LN2
ps_exp_c2:    times 8 dd 0.237348593 ; 0.49401*M_LN2*M_LN2
ps_1:         times 8 dd 1.0
ps_abs:       times 8 dd 0x7fffffff
ps_5:         times 4 dd 5.0
pb_38_m6:     times 8 db 38,-6
pw_32:        times 8 dw 32
pw_38:        times 8 dw 38
pw_m6:        times 8 dw -6
shuf_packdb   db 0,4,8,12,0,0,0,0,0,0,0,0,0,0,0,0
shuf_packswiz db 0,8,4,12,0,0,0,0,0,0,0,0,0,0,0,0
ss_48         dd 48.0
ss_1_3:       dd 0.3333333333
ss_1_16:      dd 0.0625

SECTION .text


%macro HADDPS 2 ; dst, src
    movhlps    %1, %2
    addps      %1, %2
    pshuflw    %2, %1, q0032
    addss      %1, %2
%endmacro

%macro HADDPS_X4 5 ; dst, s0, s1, s2, s3
    movaps     %1, %2
    unpcklpd   %2, %3
    unpckhpd   %1, %3
    addps      %1, %2
    movaps     %2, %4
    unpcklpd   %4, %5
    unpckhpd   %2, %5
    addps      %2, %4
    movaps     %4, %1
    shufps     %1, %2, q2020
    shufps     %4, %2, q3131
    addps      %1, %4
%endmacro

%macro HADDPI_X4 5 ; dst, s0, s1, s2, s3
    ; optimized for conroe.
    ; nehalem probably prefers 6x punpck.
    mova       %1, %2
    punpcklqdq %2, %3
    punpckhqdq %1, %3
    paddd      %1, %2
    mova       %2, %4
    punpcklqdq %4, %5
    punpckhqdq %2, %5
    paddd      %2, %4
    mova       %4, %1
    shufps     %1, %2, q2020
    shufps     %4, %2, q3131
    paddd      %1, %4
%endmacro

%macro DOTP_LOAD 1
    %assign %%n %1 ; turn arg into a literal number so that it can be used in names
    %if %%n < 4
        %assign %%i %%n
    %elifndef used4
        %assign %%i 4
    %elifndef used5
        %assign %%i 5
    %elifndef used6
        %assign %%i 6
    %elifndef used7
        %assign %%i 7
    %elifndef used8
        %assign %%i 8
    %elifndef used9
        %assign %%i 9
    %else
        %error dotproduct register allocation failed
    %endif
    %if notcpuflag(avx)
        mova     m %+ %%i, [r0+%%n*16-offset]
        %if %%n*16-offset >= 112 ; keep opcodes small
            add  r0, 256
            %assign offset offset+256
        %endif
    %endif
    CAT_XDEFINE tmp, %%n, %%i
    CAT_XDEFINE used, %%i, 1
%endmacro

%macro DOTP_MUL 1
    %assign %%n %1
    %if ARCH_X86_64
        %assign %%j 10 + (%%n / 16)
    %else
        %assign %%j 7
        %if (%%n & 15) == 0
            mova m7, [rsp+gprsize+(%%n&~15)]
        %endif
    %endif
    %assign %%i tmp %+ %%n
    %if cpuflag(avx)
        %if cpuflag(xop) && %%n >= 4
            %assign %%k %%n&3
            pmadcswd m %+ %%k, m %+ %%j, [r0+%%n*16-offset], m %+ %%k
            CAT_UNDEF used, %%i
        %else
            pmaddwd m %+ %%i, m %+ %%j, [r0+%%n*16-offset]
        %endif
        %if %%n*16-offset >= 112 ; keep opcodes small
            add  r0, 256
            %assign offset offset+256
        %endif
    %else
        pmaddwd m %+ %%i, m %+ %%j
    %endif
    %if (%%n & 3) == 3
        pshufd  m %+ %%j, m %+ %%j, q0321
    %endif
%endmacro

%macro DOTP_ACC 1
    %assign %%n %1
    %assign %%i tmp %+ %%n
    %if notcpuflag(xop) && %%n >= 4
        %assign %%j %%n&3
        paddd m %+ %%j, m %+ %%i
        CAT_UNDEF used, %%i
    %endif
    CAT_UNDEF tmp, %%n
%endmacro

%macro SCALE_DOTP 0
%assign offset 0
%if ARCH_X86_64
cglobal scale_dotproduct
    DOTP_LOAD 0
    DOTP_LOAD 1
    DOTP_LOAD 2
    DOTP_MUL  0
    DOTP_LOAD 3
    DOTP_MUL  1
%assign i 0
%rep 92
    DOTP_LOAD i+4
    DOTP_MUL  i+2
    DOTP_ACC  i+0
%assign i i+1
%endrep
    DOTP_MUL  94
    DOTP_ACC  92
    DOTP_MUL  95
    mova  [r1], m0
    DOTP_ACC  93
    DOTP_ACC  94
    mova  [r1+16], m1
    add        r0, 48*2*16-offset
    DOTP_ACC  95
    mova  [r1+32], m2
    mova  [r1+48], m3
    add        r1, 64
    ret

%else ; X86_32
cglobal scale_dotproduct
%assign i 0
%rep 95
    DOTP_LOAD i
    DOTP_MUL  i
    DOTP_ACC  i
%assign i i+1
%endrep
    DOTP_LOAD 95
    mova  [r1], m0
    DOTP_MUL  95
    mova  [r1+16], m1
    add        r0, 48*2*16-offset
    DOTP_ACC  95
    mova  [r1+32], m2
    mova  [r1+48], m3
    add        r1, 64
    ret
%endif
%endmacro ; SCALE_DOTP


%macro DOTP_MUL2 1
    %assign %%n %1
    %if ARCH_X86_64
        %assign %%j 10 + (%%n / 4)
    %else
        %assign %%j 7
    %endif
    %assign %%i tmp %+ %%n
    %if cpuflag(avx)
        %if cpuflag(xop) && %%n >= 4
            %assign %%k %%n&3
            pmadcswd m %+ %%k, m %+ %%j, [r0+%%n*16-offset], m %+ %%k
            CAT_UNDEF used, %%i
        %else
            pmaddwd m %+ %%i, m %+ %%j, [r0+%%n*16-offset]
        %endif
        %if %%n*16-offset >= 112 ; keep opcodes small
            add  r0, 256
            %assign offset offset+256
        %endif
    %else
        pmaddwd m %+ %%i, m %+ %%j
    %endif
%endmacro

%macro LOAD12x4 8
    lea       r4, [r3*3]
    movq      %1, [r2]
    movd      %2, [r2+8]
    movd      %7, [r2+r3]
    movq      %3, [r2+r3+4]
    movq      %4, [r2+r3*2]
    movd      %5, [r2+r3*2+8]
    movd      %8, [r2+r4]
    movq      %6, [r2+r4+4]
    punpckldq %2, %7
    pxor      %7, %7
    punpckldq %5, %8
    punpcklbw %1, %7
    punpcklbw %3, %7
    punpcklbw %4, %7
    punpcklbw %2, %7
    punpcklbw %5, %7
    punpcklbw %6, %7
%endmacro

%macro TEST_DOTP 0
; void test_dotproduct(const int16_t *weightsi, int *dst, const uint8_t *pix, intptr_t stride)
%assign offset 0
%if ARCH_X86_64
cglobal test_dotproduct, 4,5,16
    LOAD12x4 m10, m11, m12, m13, m14, m15, m0, m1
    DOTP_LOAD 0
    DOTP_LOAD 1
    DOTP_LOAD 2
    DOTP_MUL2 0
    DOTP_LOAD 3
    DOTP_MUL2 1
%assign i 0
%rep 20
    DOTP_LOAD i+4
    DOTP_MUL2 i+2
    DOTP_ACC  i+0
%assign i i+1
%endrep
    DOTP_MUL2 22
    DOTP_ACC  20
    DOTP_MUL2 23
    DOTP_ACC  21
    DOTP_ACC  22
    DOTP_ACC  23
    HADDPI_X4 m4, m0, m1, m2, m3
    mova [r1], m4
    RET

%else ; X86_32
cglobal test_dotproduct, 4,5,8
%assign stack_pad 0x50+((-stack_offset-gprsize)&15)
    SUB rsp, stack_pad
    LOAD12x4 m7, m0, m1, m2, m3, m4, m5, m6
    mova [rsp+0x00], m0
    mova [rsp+0x10], m1
    mova [rsp+0x20], m2
    mova [rsp+0x30], m3
    mova [rsp+0x40], m4
    DOTP_LOAD 0
    DOTP_LOAD 1
    DOTP_MUL2 0
%assign i 0
%rep 22
    DOTP_LOAD i+2
    DOTP_MUL2 i+1
%if ((i+2)&3)==0
    mova m7, [rsp+((i+2)/4-1)*16]
%endif
    DOTP_ACC  i+0
%assign i i+1
%endrep
    DOTP_MUL2 23
    DOTP_ACC  22
    DOTP_ACC  23
    HADDPI_X4 m4, m0, m1, m2, m3
    mova [r1], m4
    ADD rsp, stack_pad
    RET
%endif
%endmacro ; TEST_DOTP


%macro DOTP_LOAD3 1
    %assign %%n %1
    %if %%n == 0
        %assign %%i 0
    %elifndef used6
        %assign %%i 6
    %elifndef used7
        %assign %%i 7
    %elifndef used8
        %assign %%i 8
    %elifndef used9
        %assign %%i 9
    %elifndef used10
        %assign %%i 10
    %else
        %error dotproduct register allocation failed
    %endif
    %if %%n*16-offset0 < 0x80
        CAT_XDEFINE adr, %%i, [r0+%%n*16-offset0]
    %else
        CAT_XDEFINE adr, %%i, [r6+%%n*16-offset1]
    %endif
    CAT_XDEFINE tmp, %%n, %%i
    CAT_XDEFINE used, %%i, 1
    %if notcpuflag(avx)
        mova  m %+ %%i, adr %+ %%i
    %endif
%endmacro

%macro DOTP_MUL3 1
    %assign %%n %1
    %assign %%i tmp %+ %%n
    %if cpuflag(xop) && %%n >= 1
        %assign %%j %%n % 6
        pmadcswd m %+ %%j, m_pix, adr %+ %%i, m %+ %%j
        CAT_UNDEF used, %%i
    %elif cpuflag(avx)
        pmaddwd m %+ %%i, m_pix, adr %+ %%i
    %else
        pmaddwd m %+ %%i, m_pix
    %endif
    %if %%n % 6 == 5 && %%n < 18
        pshufd  m_pix, m_pix, q0321
    %endif
%endmacro

%macro DOTP_ACC3 1
    %assign %%n %1
    %assign %%i tmp %+ %%n
    %if notcpuflag(xop) && %%n >= 1
        %assign %%j %%n % 6
        paddd m %+ %%j, m %+ %%i
        CAT_UNDEF used, %%i
    %endif
    CAT_UNDEF tmp, %%n
%endmacro

%macro LOAD8x4_TRANPOSE 5
    movq      %1, [r2]
    movq      %2, [r2+r3]
    movq      %3, [r2+r3*2]
    movq      %4, [r2+r5]
    punpcklbw %1, %3
    punpcklbw %2, %4
    mova      %3, %1
    punpcklbw %1, %2
    punpckhbw %3, %2
    pxor      %5, %5
    mova      %2, %1
    mova      %4, %3
    punpcklbw %1, %5
    punpckhbw %2, %5
    punpcklbw %3, %5
    punpckhbw %4, %5
%endmacro

%macro TEST_DOTPS 0
; void test_dotproducts(const int16_t *weightsi, int (*dst)[4], const uint8_t *pix, intptr_t stride, int width)
%assign offset0 128
%assign offset1 384
%if ARCH_X86_64
cglobal test_dotproducts, 5,7,16
%define m_pix m12
    lea      r5, [r3*3]
    lea      r6, [r0+offset1]
    add      r0, offset0
.loop:
    LOAD8x4_TRANPOSE m12, m13, m14, m15, m11
    mova     m6, m1
    mova     m1, m3
    mova     m2, m4
    mova     m3, m5
    mova     m4, m0
    mova     m5, m6
%assign j 0
%rep 4
    pxor     m0, m0
    DOTP_LOAD3 0
    DOTP_LOAD3 1
    DOTP_LOAD3 2
    DOTP_MUL3  0
    DOTP_LOAD3 3
    DOTP_MUL3  1
%assign i 0
%rep 20
    DOTP_LOAD3 i+4
    DOTP_MUL3  i+2
    DOTP_ACC3  i+0
%assign i i+1
%endrep
    DOTP_MUL3  22
    DOTP_ACC3  20
    DOTP_MUL3  23
    DOTP_ACC3  21
    DOTP_ACC3  22
    DOTP_ACC3  23
    mova   [r1+j*16], m5
    SWAP 5, 4, 3, 2, 1, 0
    SWAP 12, 13, 14, 15
%if j<3
    dec      r4
    jle .ret
%endif
%assign j j+1
%endrep
    add      r2, 8
    add      r1, 64
    dec      r4
    jg .loop
.ret:
    REP_RET

%else ; X86_32
cglobal test_dotproducts, 5,7,8
%define m_pix m7
%assign stack_pad 0x50+((-stack_offset-gprsize)&15)
    SUB     rsp, stack_pad
    lea      r5, [r3*3]
    lea      r6, [r0+offset1]
    add      r0, offset0
.loop:
    mova     m1, m3
    mova     m2, m4
    mova     m3, m5
    LOAD8x4_TRANPOSE m7, m6, m5, m4, m0
    mova [rsp+0x00], m6
    mova [rsp+0x10], m5
    mova [rsp+0x20], m4
    mova     m4, [rsp+0x30]
    mova     m5, [rsp+0x40]
%assign j 0
%rep 4
    pxor     m0, m0
%assign i 0
%rep 24
    DOTP_LOAD3 i
    DOTP_MUL3  i
    DOTP_ACC3  i
%assign i i+1
%endrep
    mova   [r1+j*16], m5
    SWAP 5, 4, 3, 2, 1, 0
%if j<3
    mova  m_pix, [rsp+j*16]
    dec      r4
    jle .ret
%endif
    mova [rsp+0x30], m4
    mova [rsp+0x40], m5
%assign j j+1
%endrep
    add      r2, 8
    add      r1, 64
    dec      r4
    jg .loop
.ret:
    ADD     rsp, stack_pad
    RET
%endif
%endmacro ; TEST_DOTPS

INIT_XMM sse2
SCALE_DOTP
TEST_DOTP
TEST_DOTPS
INIT_XMM avx
SCALE_DOTP
TEST_DOTP
TEST_DOTPS
INIT_XMM xop
SCALE_DOTP
TEST_DOTP
TEST_DOTPS



%macro SIGMOID 2 ; dst, tmp
    andps    %2, %1, m_abs
    addps    %2, m_1
    rcpps    %2, %2
    mulps    %1, %2
%endmacro

%macro EXP2 3-4 ; dst, tmp, tmp, tmp (avx only)
    addps    %2, %1, m_exp_bias
%if mmsize==32
    vextractf128 %4, %2, 1
    vpslld   xmm %+ n%3, xmm %+ n%2, 23
    vpslld   %4, %4, 23
    vinsertf128 %3, %3, %4, 1
%else
    pslld    %3, %2, 23
%endif
    subps    %2, m_exp_bias
    subps    %1, %2
    mulps    %2, %1, m_exp_c1
    mulps    %1, %1
    mulps    %1, m_exp_c2
    addps    %2, m_exp_c0
    addps    %1, %2
    ; FIXME do I need both versions? xmm could do mulps too, it would just have more latency and constraints on execution units
%if mmsize==32
    mulps    %1, %3
%else
    paddd    %1, %3
%endif
%endmacro

%macro LOAD_SUM_SQUARE 7 ; dst0, dst1, sum0, sum1, t2, src0, src1
    movq       %3, %6
    movq       %5, %7
    punpcklqdq %4, %3, %5
    punpcklbw  %3, m0
    punpcklbw  %5, m0
    mova       %1, %3
    mova       %2, %5
    pmaddwd    %3, %3
    pmaddwd    %5, %5
    psadbw     %4, m0
    paddd      %3, %5
%endmacro

%macro SCALE_NET_TAIL 1
    HADDPS   m4, m0
    HADDPS   m5, m1
    mulss    m2, [ps_5]
    rcpss    m0, m4
    mulss    m5, m2
    mulss    m0, m5
    addss    m0, m3
    cvtss2si %1, m0
    test     %1, ~255
    jz %%.noclip
    not      %1
    sar      %1, 31
    shr      %1, 24
%%.noclip:
%endmacro


%macro SCALE_NET 1 ; nns
; int scale_net(struct { int16_t i[48*2*NNS]; float f[4*NNS]; } *weights, const uint8_t *pix, intptr_t stride)
%assign NNS 16<<%1
cglobal scale_net%1
    %assign stack_pad NNS*8+16+((-gprsize)&15)
%if ARCH_X86_64
    %define buf rsp
%else
    %assign stack_pad stack_pad+0x60
    %define spill rsp
    %define buf rsp+0x60
%endif
    %define mean buf+NNS*8
    %define stddev mean+4
    %define invstddev stddev+4
    SUB       rsp, stack_pad

    ; compute sum and sum squared
    lea        r3, [r2*3]
    add        r3, r1
    pxor       m0, m0
%if ARCH_X86_64
    LOAD_SUM_SQUARE m10, m11, m1, m2, m3, [r1], [r1+r2]
    LOAD_SUM_SQUARE m12, m13, m3, m4, m5, [r1+r2*2], [r3]
    paddd      m2, m4
    paddd      m1, m3
    LOAD_SUM_SQUARE m14, m15, m3, m4, m5, [r3+r2], [r3+r2*2]
%else
    LOAD_SUM_SQUARE [spill+0x00], [spill+0x10], m1, m2, m3, [r1], [r1+r2]
    LOAD_SUM_SQUARE [spill+0x20], [spill+0x30], m3, m4, m5, [r1+r2*2], [r3]
    paddd      m2, m4
    paddd      m1, m3
    LOAD_SUM_SQUARE [spill+0x40], [spill+0x50], m3, m4, m5, [r3+r2], [r3+r2*2]
%endif
    paddd      m2, m4
    paddd      m1, m3
    movhlps    m4, m2
    movhlps    m3, m1
    paddd      m2, m4
    paddd      m1, m3
    pshuflw    m3, m1, q0032
    paddd      m1, m3
    movd      r3d, m2
    movd      r1d, m1

    ; compute means and stddev
    xorps      m2, m2
    cvtsi2ss   m2, r3d
    imul      r1d, 48
    mulss      m2, [ss_1_3]
    cvtps2dq   m3, m2
    imul      r3d, r3d
    mulss      m2, [ss_1_16]
    sub       r1d, r3d
    jle .zero
    cvtsi2ss   m1, r1d
    movss  [mean], m2
    rsqrtss    m1, m1
    mulss      m1, [ss_48]
    movss  [invstddev], m1
    rcpss      m1, m1
    movss  [stddev], m1
.unzero:

    ; neural net
    lea      r1, [buf]
%rep NNS/8
    call scale_dotproduct
%endrep

%if cpuflag(avx) && ARCH_X86_64 ; FIXME register allocation for x86_32
    INIT_YMM cpuname
%else
    %define  mova movaps
%endif
%if ARCH_X86_64
    %define  m_invstddev  m9
    %define  m_exp_bias   m10
    %define  m_exp_c0     m11
    %define  m_exp_c1     m12
    %define  m_exp_c2     m13
    %define  m_1          m14
    %define  m_abs        m15
    mova     m_exp_c0,    [ps_exp_c0]
    mova     m_exp_c1,    [ps_exp_c1]
    mova     m_exp_c2,    [ps_exp_c2]
    mova     m_1,         [ps_1]
    mova     m_abs,       [ps_abs]
%else
    %define  m_invstddev  m6
    %define  m_exp_bias   m7
    %define  m_exp_c0     [ps_exp_c0]
    %define  m_exp_c1     [ps_exp_c1]
    %define  m_exp_c2     [ps_exp_c2]
    %define  m_1          [ps_1]
    %define  m_abs        [ps_abs]
%endif
%if mmsize==32
    mova     m_exp_bias,  [ps_exp_bias1]
    vbroadcastss m_invstddev, [invstddev]
%else
    mova     m_exp_bias,  [ps_exp_bias0]
    movss    m_invstddev, [invstddev]
    shufps   m_invstddev, m_invstddev, 0
%endif

    xorps    m0, m0
    xorps    m1, m1
%assign i 0
%rep NNS*4/mmsize
    mulps    m4, m_invstddev, [r0+i]
    mulps    m5, m_invstddev, [r0+i+NNS*4]
    cvtdq2ps m2, [buf+i]
    cvtdq2ps m3, [buf+i+NNS*4]
    mulps    m2, m4
    mulps    m3, m5 ; could go into the "+1.0" in the sigmoid, for reduced dependency chain
    addps    m2, [r0+i+NNS*8]
    addps    m3, [r0+i+NNS*12]
    SIGMOID  m3, m4
    EXP2     m2, m4, m5, xmm6
    mulps    m3, m2
    addps    m0, m2
    addps    m1, m3
%assign i i+mmsize
%endrep
%if cpuflag(avx)
    vextractf128 xmm2, ymm0, 1
    vextractf128 xmm3, ymm1, 1
    vzeroupper
    INIT_XMM cpuname
    addps        m0, m2
    addps        m1, m3
%endif
    sub      r0, 48*4*NNS
    movss    m2, [stddev]
    movss    m3, [mean]
    ADD rsp, stack_pad
    ret

.zero:
    movaps   m3, m2
    movaps   m0, [ps_1]
    xorps    m1, m1
    xorps    m2, m2
    ADD rsp, stack_pad
    ret



; void scale_nets(const int16_t *weights, const uint8_t *pix, intptr_t stride, uint8_t *dst, const uint16_t *offsets, int n)
cglobal scale_nets%1, 6,12,16
%if ARCH_X86_64
    mov      r6, r1
    mov      r7, r3
%endif
    %assign stack_pad 0x80+((-stack_offset-gprsize)&15)
    SUB     rsp, stack_pad
    sub      r5, 3
    jle .skip4

.loop4:
%if ARCH_X86_64
    movzx   r11, word [r4+6]
    movzx   r10, word [r4+4]
    movzx    r9, word [r4+2]
    movzx    r8, word [r4]
    lea      r1, [r6+r11]
%else
    mov      r6, r1m
    movzx    r1, word [r4+6]
    add      r1, r6
%endif
    call scale_net%1
    movaps [rsp+0x20], m0
    movaps [rsp+0x50], m1
    movss  [rsp+0x6c], m2
    movss  [rsp+0x7c], m3
%if ARCH_X86_64
    lea      r1, [r6+r10]
%else
    movzx    r1, word [r4+4]
    add      r1, r6
%endif
    call scale_net%1
    movaps [rsp+0x10], m0
    movaps [rsp+0x40], m1
    movss  [rsp+0x68], m2
    movss  [rsp+0x78], m3
%if ARCH_X86_64
    lea      r1, [r6+r9]
%else
    movzx    r1, word [r4+2]
    add      r1, r6
%endif
    call scale_net%1
    movaps [rsp+0x00], m0
    movaps [rsp+0x30], m1
    movss  [rsp+0x64], m2
    movss  [rsp+0x74], m3
%if ARCH_X86_64
    lea      r1, [r6+r8]
%else
    movzx    r1, word [r4]
    add      r1, r6
%endif
    call scale_net%1
    SWAP 4, 0
    movaps   m5, [rsp+0x00]
    movaps   m6, [rsp+0x10]
    movaps   m7, [rsp+0x20]
    HADDPS_X4 m0, m4, m5, m6, m7
    SWAP 4, 1
    movaps   m5, [rsp+0x30]
    movaps   m6, [rsp+0x40]
    movaps   m7, [rsp+0x50]
    HADDPS_X4 m1, m4, m5, m6, m7
    movaps   m4, [rsp+0x60]
    movaps   m5, [rsp+0x70]
    movss    m4, m2
    movss    m5, m3
    mulps    m4, [ps_5]
    rcpps    m0, m0
    mulps    m1, m4
    mulps    m0, m1
    addps    m0, m5
    cvtps2dq m0, m0
%if ARCH_X86_64
    packuswb m0, m0
    movq     r1, m0
    mov      [r7+r8], r1b
    shr      r1, 16
    mov      [r7+r9], r1b
    shr      r1, 16
    mov      [r7+r10], r1b
    shr      r1, 16
    mov      [r7+r11], r1b
%else
    packssdw m0, m0
    packuswb m0, m0
    movd     r1, m0
    mov      r3, r3m
    movzx    r6, word [r4]
    mov      [r3+r6], r1b
    shr      r1, 8
    movzx    r6, word [r4+2]
    mov      [r3+r6], r1b
    shr      r1, 8
    movzx    r6, word [r4+4]
    mov      [r3+r6], r1b
    shr      r1, 8
    movzx    r6, word [r4+6]
    mov      [r3+r6], r1b
%endif
    add      r4, 8
    sub      r5, 4
    jg .loop4

    RESET_MM_PERMUTATION
.skip4:
    add      r5, 3
    jle .ret
.loop1:
%if ARCH_X86_64
    movzx   r8, word [r4]
    lea      r1, [r6+r8]
    call scale_net%1
    SCALE_NET_TAIL r1d
    mov      [r7+r8], r1b
%else
    movzx    r1, word [r4]
    add      r1, r1m
    call scale_net%1
    SCALE_NET_TAIL r1d
    movzx    r3, word [r4]
    add      r3, r3m
    mov      [r3], r1b
%endif
    add      r4, 2
    dec      r5
    jg .loop1
.ret:
    ADD     rsp, stack_pad
    RET
%endmacro ; SCALE_NET

%if ARCH_X86_64
    %define pointer dq
%else
    %define pointer dd
%endif

%macro SCALE_NETS 0
SCALE_NET 0
SCALE_NET 1
SCALE_NET 2
SCALE_NET 3
SCALE_NET 4

cglobal scale_nets_tab
    pointer scale_nets0 %+ SUFFIX
    pointer scale_nets1 %+ SUFFIX
    pointer scale_nets2 %+ SUFFIX
    pointer scale_nets3 %+ SUFFIX
    pointer scale_nets4 %+ SUFFIX
%endmacro

INIT_XMM sse2
SCALE_NETS
INIT_XMM avx
SCALE_NETS
INIT_XMM xop
SCALE_NETS



%if 0
; int test_net(const float *weightsf, const int *dotp)
cglobal test_net, 2,2,10
%define m_1   m8
%define m_abs m9
    add      r0, 0x80
    cvtdq2ps m0, [r1]
    mulps    m0, [r0-0x80]
    addps    m0, [r0-0x70]
    movaps   m_1,   [ps_1]
    movaps   m_abs, [ps_abs]
    movaps   m1, [r0-0x60]
    SIGMOID  m0, m4
    pshufd   m5, m0, q0321
    movaps   m2, [r0-0x50]
    pshufd   m6, m0, q1032
    mulps    m1, m0
    movaps   m3, [r0-0x40]
    pshufd   m7, m0, q2103
    mulps    m2, m5
    movaps   m4, [r0-0x30]
    mulps    m3, m6
    addps    m1, [r0-0x20]
    mulps    m4, m7
    addps    m2, m3
    mulps    m0, [r0-0x10]
    addps    m1, m4
    mulps    m5, [r0+0x00]
    addps    m1, m2
    mulps    m6, [r0+0x10]
    SIGMOID  m1, m8
    mulps    m7, [r0+0x20]
    pshufd   m2, m1, q0321
    addps    m0, [r0+0x70]
    addps    m0, m5
    pshufd   m3, m1, q1032
    addps    m6, m7
    pshufd   m4, m1, q2103
    mulps    m1, [r0+0x30]
    addps    m0, m6
    mulps    m2, [r0+0x40]
    addps    m0, m1
    mulps    m3, [r0+0x50]
    mulps    m4, [r0+0x60]
    addps    m2, m3
    addps    m0, m4
    addps    m0, m2
    movhlps  m1, m0
    maxps    m0, m1
    pshuflw  m1, m0, q0032
    xor     eax, eax
    comiss   m0, m1
    seta     al
    RET
%endif



%macro TEST_NET_HEAD 0
    add      r0, 0x80
    movaps   m4, [r0-0x80]
    movaps   m5, [r0-0x70]
    cvtdq2ps m0, [r1+0x00]
    cvtdq2ps m1, [r1+0x10]
    cvtdq2ps m2, [r1+0x20]
    cvtdq2ps m3, [r1+0x30]
    mulps    m0, m4
    mulps    m1, m4
    mulps    m2, m4
    mulps    m3, m4
    movaps   m_1, [ps_1]
    movaps   m_abs, [ps_abs]
    addps    m0, m5
    addps    m1, m5
    addps    m2, m5
    addps    m3, m5
    SIGMOID  m0, m4
    SIGMOID  m1, m4
    SIGMOID  m2, m4
    SIGMOID  m3, m4
%endmacro

%macro TEST_NET_TAIL 0
    movaps   m4, m0
    unpcklpd m0, m1 ; 0,1,4,5
    unpckhpd m4, m1 ; 2,3,6,7
    maxps    m4, m0
    movaps   m0, m2
    unpcklpd m2, m3 ; 8,9,12,13
    unpckhpd m0, m3 ; 10,11,14,15
    maxps    m2, m0
    andps    m0, m_abs
    andps    m2, m_abs
    movaps   m0, m4
    shufps   m4, m2, q2020 ; 0,4,8,12
    shufps   m0, m2, q3131 ; 1,5,9,13
    psubd    m0, m4
    psrld    m0, 31
%if cpuflag(ssse3)
    pshufb   m0, [shuf_packdb]
%else
    packssdw m0, m0
    packuswb m0, m0
%endif
    movd    eax, m0
%endmacro

%macro TEST_NET_HEAD_YMM 0
    add      r0, 0x80
    vbroadcastf128 m2, [r0-0x80]
    vbroadcastf128 m3, [r0-0x70]
    cvtdq2ps m0, [r1+0x00]
    cvtdq2ps m1, [r1+0x20]
    mulps    m0, m2
    mulps    m1, m2
    mova     m_1, [ps_1]
    mova     m_abs, [ps_abs]
    addps    m0, m3
    addps    m1, m3
    SIGMOID  m0, m2
    SIGMOID  m1, m2
%endmacro

%macro TEST_NET_TAIL_YMM 0
    unpckhpd m2, m0, m1 ; 2,3,10,11, 6,7,14,15
    unpcklpd m0, m0, m1 ; 0,1,8,9, 4,5,12,13
    maxps    m0, m2
    andps    m0, m_abs
    vextractf128 xmm2, ymm0, 1
    vzeroupper
    INIT_XMM cpuname
    shufps   m4, m0, m2, q2020 ; 0,8,4,12
    shufps   m0, m0, m2, q3131 ; 1,9,5,13
    psubd    m0, m4
    psrld    m0, 31
    pshufb   m0, [shuf_packswiz]
    movd    eax, m0
%endmacro

%macro YMUL 4
    %if mmsize==32
        ; FIXME broadcast is called twice per address; could keep the register around, or duplicate the table entries
        vbroadcastf128 %4, %3
        mulps          %1, %2, %4
    %else
        mulps          %1, %2, %3
    %endif
%endmacro

%macro YADD 4
    %if mmsize==32
        vbroadcastf128 %4, %3
        addps          %1, %2, %4
    %else
        addps          %1, %2, %3
    %endif
%endmacro

%macro YSHUF 3
    %if mmsize==32
        vpermilps %1, %2, %3
    %else
        pshufd    %1, %2, %3
    %endif
%endmacro

%if ARCH_X86_64

%macro DOTP0 2
    YSHUF   m11,  %1, q0321
    YSHUF   m12,  %1, q1032
    YSHUF   m13,  %1, q2103
    YMUL     m8, m11, [r0-0x50], m5
    YMUL     m9, m12, [r0-0x40], m5
    YMUL    m10, m13, [r0-0x30], m5
    YMUL     %1,  %1, [r0-0x10], m5
    YMUL    m11, m11, [r0+0x00], m5
    YMUL    m12, m12, [r0+0x10], m5
    YMUL    m13, m13, [r0+0x20], m5
    YADD     %1,  %1, [r0+0x70], m5
    addps    %2,  m8
    addps    %1, m11
    addps    %2,  m9
    addps    %1, m12
    addps    %2, m10
    addps    %1, m13
%endmacro

%macro DOTP1 2
    YSHUF    m8,  %2, q0321
    YSHUF    m9,  %2, q1032
    YSHUF   m10,  %2, q2103
    YMUL     %2,  %2, [r0+0x30], m5
    YMUL     m8,  m8, [r0+0x40], m5
    YMUL     m9,  m9, [r0+0x50], m5
    YMUL    m10, m10, [r0+0x60], m5
    addps    %1,  %2
    addps    %1,  m8
    addps    %1,  m9
    addps    %1, m10
%endmacro

INIT_YMM avx
cglobal test_net_x4, 2,2,16
    TEST_NET_HEAD_YMM
    vbroadcastf128 m3, [r0-0x60]
    vbroadcastf128 m4, [r0-0x20]
    mulps    m2, m3, m0
    addps    m2, m4
    DOTP0    m0, m2
    SIGMOID  m2, m5
    mulps    m3, m1
    addps    m3, m4
    DOTP0    m1, m3
    SIGMOID  m3, m5
    DOTP1    m0, m2
    DOTP1    m1, m3
    TEST_NET_TAIL_YMM
    RET

%macro TEST_NET 0
; int test_net_x4(const float *weightsf, const int (*dotp)[4])
cglobal test_net_x4, 2,2,16
%define m_1   m14
%define m_abs m15
    TEST_NET_HEAD
    mulps    m4, m0, [r0-0x60]
    mulps    m5, m1, [r0-0x60]
    addps    m4, [r0-0x20]
    addps    m5, [r0-0x20]
    DOTP0    m0, m4
    DOTP0    m1, m5
    SIGMOID  m4, m8
    SIGMOID  m5, m8
    mulps    m6, m2, [r0-0x60]
    mulps    m7, m3, [r0-0x60]
    addps    m6, [r0-0x20]
    addps    m7, [r0-0x20]
    DOTP0    m2, m6
    DOTP0    m3, m7
    SIGMOID  m6, m8
    SIGMOID  m7, m8
    DOTP1    m0, m4
    DOTP1    m1, m5
    DOTP1    m2, m6
    DOTP1    m3, m7
    TEST_NET_TAIL
    RET
%endmacro ; TEST_NET

%else ; X86_32

%macro DOTP0 6 ; sum0, sum1, tmps
    mulps    %2, %1, [r0-0x60]
    mulps    %3, %1, [r0-0x10]
    pshufd   %4, %1, q0321
    addps    %2, [r0-0x20]
    addps    %3, [r0+0x70]
    mulps    %5, %4, [r0+0x00]
    mulps    %4, [r0-0x50]
    pshufd   %6, %1, q1032
    addps    %2, %4
    addps    %3, %5
    mulps    %4, %6, [r0+0x10]
    mulps    %6, [r0-0x40]
    pshufd   %1, %1, q2103
    addps    %3, %4
    addps    %2, %6
    mulps    %5, %1, [r0-0x30]
    mulps    %1, [r0+0x20]
    addps    %2, %5
    addps    %1, %3
%endmacro

%macro DOTP1 3 ; sum, in, tmp
    pshufd   %3, %2, q0321
    mulps    %3, [r0+0x40]
    addps    %1, %3
    pshufd   %3, %2, q1032
    mulps    %3, [r0+0x50]
    addps    %1, %3
    pshufd   %3, %2, q2103
    mulps    %3, [r0+0x60]
    addps    %1, %3
    mulps    %2, [r0+0x30]
    addps    %1, %2
%endmacro

%macro TEST_NET 0
; int test_net_x4(const float *weightsf, const int (*dotp)[4])
cglobal test_net_x4, 2,2,8
%assign stack_pad 0x60+((-stack_offset-gprsize)&15)
%define m_1   m6
%define m_abs m7
    SUB     rsp, stack_pad
    TEST_NET_HEAD
    movaps   [rsp+0x20], m2
    movaps   [rsp+0x30], m3
    DOTP0    m0, m4, m2, m3, m6, m7
    DOTP0    m1, m5, m2, m3, m6, m7
    movaps   m2, [rsp+0x20]
    movaps   m3, [rsp+0x30]
    movaps   [rsp+0x00], m0
    movaps   [rsp+0x10], m1
    movaps   [rsp+0x40], m4
    movaps   [rsp+0x50], m5
    DOTP0    m2, m6, m0, m1, m4, m5
    DOTP0    m3, m7, m0, m1, m4, m5
    movaps   m4, [rsp+0x40]
    movaps   m5, [rsp+0x50]
%define m_1   m1
%define m_abs [ps_abs]
    movaps   m_1, [ps_1]
    SIGMOID  m4, m0
    SIGMOID  m5, m0
    SIGMOID  m6, m0
    SIGMOID  m7, m0
    movaps   m0, [rsp+0x00]
    DOTP1    m0, m4, m1
    movaps   m1, [rsp+0x10]
    DOTP1    m1, m5, m4
    DOTP1    m2, m6, m4
    DOTP1    m3, m7, m4
    TEST_NET_TAIL
    ADD     rsp, stack_pad
    RET
%endmacro ; TEST_NET
%endif ; X86_32

INIT_XMM sse2
TEST_NET
INIT_XMM ssse3
TEST_NET



%macro BICUBIC_LOOP_SSE2 0
    mova      m4, [pw_m6]
    mova      m5, [pw_38]
    mova      m6, [pw_32]
    pxor      m7, m7
align 16
%%.loop:
    movq      m0, [r1+r3]
    movq      m1, [r4+r3]
    movq      m3, [r6+r3]
    movq      m2, [r5+r3]
    punpcklbw m0, m7
    punpcklbw m1, m7
    punpcklbw m3, m7
    punpcklbw m2, m7
    paddw     m0, m3
    paddw     m1, m2
    pmullw    m0, m4
    pmullw    m1, m5
    add       r3, 8
    paddw     m0, m6
    paddw     m0, m1
    psraw     m0, 6
    packuswb  m0, m0
    movq [r0+r3], m0
    jl %%.loop
%endmacro

%macro BICUBIC_LOOP_SSSE3 1
    mova      m4, [pb_38_m6]
    mova      m5, [pw_32]
align 16
%%.loop:
    mova      m0, [r4+r3]
    mova      m1, [r5+r3]
    mova      m2, m0
    mova      m3, m1
    punpcklbw m0, [r1+r3]
    punpcklbw m1, [r6+r3]
    punpckhbw m2, [r1+r3]
    punpckhbw m3, [r6+r3]
    pmaddubsw m0, m4
    pmaddubsw m1, m4
    pmaddubsw m2, m4
    pmaddubsw m3, m4
    add       r3, 16
    paddw     m0, m5
    paddw     m2, m5
    paddw     m0, m1
    paddw     m2, m3
    psraw     m0, 6
    psraw     m2, 6
    packuswb  m0, m2
    %1   [r0+r3], m0
    jl %%.loop
%endmacro

; void bicubic(uint8_t *dst, uint8_t *src, intptr_t stride, int width)
INIT_XMM sse2
cglobal bicubic, 4,7,8
    add       r1, r3
    lea       r4, [r1+r2]
    lea       r5, [r1+r2*2]
    lea       r6, [r4+r2*2]
    lea       r0, [r0+r3-8]
    neg       r3
    BICUBIC_LOOP_SSE2
    REP_RET

INIT_XMM ssse3
cglobal bicubic, 4,7,6
    add       r1, r3
    lea       r4, [r1+r2]
    lea       r5, [r1+r2*2]
    lea       r6, [r4+r2*2]
    test      r0, 15
    jnz .unaligned
    lea       r0, [r0+r3-16]
    neg       r3
    BICUBIC_LOOP_SSSE3 mova
    REP_RET
.unaligned:
    lea       r0, [r0+r3-16]
    neg       r3
    BICUBIC_LOOP_SSSE3 movu
    REP_RET

