%include "x86inc.asm"
%include "nnedi-a.h"

SECTION_RODATA
ps_exp_bias: times 4 dd 12582912.0 ; 3<<22
ps_exp_c0:   times 4 dd 1.00035
ps_exp_c1:   times 4 dd 0.701277797 ; 1.01173*M_LN2
ps_exp_c2:   times 4 dd 0.237348593 ; 0.49401*M_LN2*M_LN2
ps_1:        times 4 dd 1.0
ps_abs:      times 4 dd 0x7fffffff
pb_38_m6:    times 8 db 38,-6
pw_32:       times 8 dw 32
shuf_packdb  db 0,4,8,12,0,0,0,0,0,0,0,0,0,0,0,0
ss_5:        dd 5.0
ss_48        dd 48.0
ss_1_3:      dd 0.3333333333
ss_1_16:     dd 0.0625

SECTION .text
INIT_XMM


%macro HADDPS 2 ; dst, src
    movhlps    %1, %2
    addps      %1, %2
    pshuflw    %2, %1, 0xe
    addss      %1, %2
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
    shufps     %1, %2, 0x88
    shufps     %4, %2, 0xdd
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
    mova     m %+ %%i, [r0+%%n*16-offset]
    %if %%n*16-offset >= 112 ; keep opcodes small
        add  r0, 256
        %assign offset offset+256
    %endif
    CAT_XDEFINE tmp, %%n, %%i
    CAT_XDEFINE used, %%i, 1
%endmacro

%macro DOTP_MUL 1
    %assign %%n %1
    %ifdef ARCH_X86_64
        %assign %%j 10 + (%%n / 16)
    %else
        %assign %%j 7
        %if (%%n & 15) == 0
            mova m7, [rsp+gprsize+(%%n&~15)]
        %endif
    %endif
    %assign %%i tmp %+ %%n
    pmaddwd m %+ %%i, m %+ %%j
    %if (%%n & 3) == 3
        pshufd  m %+ %%j, m %+ %%j, 0x39
;       palignr m %+ %%j, m %+ %%j, 4
    %endif
%endmacro

%macro DOTP_ACC 1
    %assign %%n %1
    %assign %%i tmp %+ %%n
    %if %%n >= 4
        %assign %%j %%n&3
        paddd m %+ %%j, m %+ %%i
        CAT_UNDEF used, %%i
    %endif
    CAT_UNDEF tmp, %%n
%endmacro

%assign offset 128 ; FIXME could be 0 (possibly without any loss)
%ifdef ARCH_X86_64
cglobal scale_dotproduct_sse2
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
    mova  [r2], m0
    DOTP_ACC  93
    DOTP_ACC  94
    mova  [r2+16], m1
    add        r0, 48*2*16+128-offset
    DOTP_ACC  95
    mova  [r2+32], m2
    mova  [r2+48], m3
    add        r2, 64
    ret

%else ; X86_32
cglobal scale_dotproduct_sse2
%assign i 0
%rep 95
    DOTP_LOAD i
    DOTP_MUL  i
    DOTP_ACC  i
%assign i i+1
%endrep
    DOTP_LOAD 95
    mova  [r2], m0
    DOTP_MUL  95
    mova  [r2+16], m1
    add        r0, 48*2*16+128-offset
    DOTP_ACC  95
    mova  [r2+32], m2
    mova  [r2+48], m3
    add        r2, 64
    ret
%endif


%macro DOTP_MUL2 1
    %assign %%n %1
    %ifdef ARCH_X86_64
        %assign %%j 10 + (%%n / 4)
    %else
        %assign %%j 7
    %endif
    %assign %%i tmp %+ %%n
    pmaddwd m %+ %%i, m %+ %%j
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

; void test_dotproduct(const int16_t *weightsi, int *dst, const uint8_t *pix, int stride)
%assign offset 0
%ifdef ARCH_X86_64
cglobal test_dotproduct_sse2, 4,5,16
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
    HADDPI_X4 m4, m0, m1, m2, m3 ; FIXME partly interleave with the above
    mova [r1], m4
    RET

%else ; X86_32
cglobal test_dotproduct_sse2, 4,5,8
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
    HADDPI_X4 m4, m0, m1, m2, m3 ; FIXME partly interleave with the above
    mova [r1], m4
    ADD rsp, stack_pad
    RET
%endif


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
        mova  m %+ %%i, [r0+%%n*16-offset0]
    %else
        mova  m %+ %%i, [r6+%%n*16-offset1]
    %endif
    CAT_XDEFINE tmp, %%n, %%i
    CAT_XDEFINE used, %%i, 1
%endmacro

%macro DOTP_MUL3 1
    %assign %%n %1
    %assign %%i tmp %+ %%n
    pmaddwd m %+ %%i, m_pix
    %if %%n % 6 == 5 && %%n < 18
        pshufd  m_pix, m_pix, 0x39
    %endif
%endmacro

%macro DOTP_ACC3 1
    %assign %%n %1
    %assign %%i tmp %+ %%n
    %if %%n >= 1
        %assign %%j %%n % 6
        paddd m %+ %%j, m %+ %%i
        CAT_UNDEF used, %%i
    %endif
    CAT_UNDEF tmp, %%n
%endmacro

%macro LOAD8x4_TRANPOSE 5
    movq      %1, [r2] ; FIXME palignr?
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

; void test_dotproducts(const int16_t *weightsi, int (*dst)[4], const uint8_t *pix, int stride, int width)
%assign offset0 128
%assign offset1 384
%ifdef ARCH_X86_64
cglobal test_dotproducts_sse2, 5,7,16
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
cglobal test_dotproducts_sse2, 5,7,8
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



%macro SIGMOID 2 ; dst, tmp
    movaps   %2, %1
    andps    %1, m_abs
    addps    %1, m_1
    rcpps    %1, %1
    mulps    %1, %2
%endmacro

%macro EXP2 3 ; dst, tmp, tmp
    movaps   %2, %1
    addps    %1, m_exp_bias
    movaps   %3, %1
    subps    %1, m_exp_bias
    pslld    %3, 23
    subps    %2, %1
    movaps   %1, %2
    mulps    %2, %2
    mulps    %1, m_exp_c1
    mulps    %2, m_exp_c2
    addps    %1, m_exp_c0
    addps    %1, %2
    paddd    %1, %3
%endmacro

%macro LOAD_SUM_SQUARE 7 ; dst0, dst1, sum0, sum1, t2, src0, src1
    movq       %3, %6
    movq       %5, %7
    mova       %4, %3
    punpcklqdq %4, %5
    punpcklbw  %3, m0
    punpcklbw  %5, m0
    mova       %1, %3
    mova       %2, %5
    pmaddwd    %3, %3
    pmaddwd    %5, %5
    psadbw     %4, m0
    paddd      %3, %5
%endmacro


; int scale_net(const int16_t *weightsi, const float *weightsf, const uint8_t *pix, int stride)
cglobal scale_net_sse2, 4,5,16
    %assign stack_pad NNS*8+16+((-stack_offset-gprsize)&15)
%ifdef ARCH_X86_64
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
    lea        r4, [r3*3]
    add        r4, r2
    pxor       m0, m0
%ifdef ARCH_X86_64
    LOAD_SUM_SQUARE m10, m11, m1, m2, m3, [r2], [r2+r3]
    LOAD_SUM_SQUARE m12, m13, m3, m4, m5, [r2+r3*2], [r4]
    paddd      m2, m4
    paddd      m1, m3
    LOAD_SUM_SQUARE m14, m15, m3, m4, m5, [r4+r3], [r4+r3*2]
%else
    LOAD_SUM_SQUARE [spill+0x00], [spill+0x10], m1, m2, m3, [r2], [r2+r3]
    LOAD_SUM_SQUARE [spill+0x20], [spill+0x30], m3, m4, m5, [r2+r3*2], [r4]
    paddd      m2, m4
    paddd      m1, m3
    LOAD_SUM_SQUARE [spill+0x40], [spill+0x50], m3, m4, m5, [r4+r3], [r4+r3*2]
%endif
    paddd      m2, m4
    paddd      m1, m3
    movhlps    m4, m2
    movhlps    m3, m1
    paddd      m2, m4
    paddd      m1, m3
    pshuflw    m3, m1, 14
    paddd      m1, m3
    movd      r4d, m2
    movd      r3d, m1

    ; compute means and stddev
    xorps      m2, m2
    cvtsi2ss   m2, r4d
    imul      r3d, 48
    mulss      m2, [ss_1_3]
    cvtps2dq   m3, m2
    imul      r4d, r4d
    mulss      m2, [ss_1_16]
    sub       r3d, r4d
    jle .zero
    cvtsi2ss   m1, r3d
    movss  [mean], m2
    rsqrtss    m1, m1
    mulss      m1, [ss_48]
    movss  [invstddev], m1
    rcpss      m1, m1
    movss  [stddev], m1

    ; remove mean
    pshuflw    m3, m3, 0
    punpcklqdq m3, m3
%assign i 10
%rep 6
%ifdef ARCH_X86_64
    psllw    m %+ i, 4
    psubw    m %+ i, m3
%else
    mova     m0, [spill+(i-10)*16]
    psllw    m0, 4
    psubw    m0, m3
    mova     [spill+(i-10)*16], m0
%endif
%assign i i+1
%endrep

    ; neural net
    add      r0, 128
    lea      r2, [buf]
%rep NNS/8
    call scale_dotproduct_sse2
%endrep

%ifdef ARCH_X86_64
    %define  m_invstddev  m9
    %define  m_exp_bias   m10
    %define  m_exp_c0     m11
    %define  m_exp_c1     m12
    %define  m_exp_c2     m13
    %define  m_1          m14
    %define  m_abs        m15
    movss    m_invstddev, [invstddev]
    movaps   m_exp_bias,  [ps_exp_bias]
    movaps   m_exp_c0,    [ps_exp_c0]
    shufps   m_invstddev, m_invstddev, 0
    movaps   m_exp_c1,    [ps_exp_c1]
    movaps   m_exp_c2,    [ps_exp_c2]
    movaps   m_1,         [ps_1]
    movaps   m_abs,       [ps_abs]
%else
    %define  m_invstddev  m6
    %define  m_exp_bias   m7
    %define  m_exp_c0     [ps_exp_c0]
    %define  m_exp_c1     [ps_exp_c1]
    %define  m_exp_c2     [ps_exp_c2]
    %define  m_1          [ps_1]
    %define  m_abs        [ps_abs]
    movss    m_invstddev, [invstddev]
    movaps   m_exp_bias,  [ps_exp_bias]
    shufps   m_invstddev, m_invstddev, 0
%endif

    add      r1, NNS*8
    xorps    m4, m4
    xorps    m5, m5
%assign i 0
%rep NNS/4
    movaps   m2, [r1+i*8-NNS*8]
    movaps   m3, [r1+i*8]
    cvtdq2ps m0, [buf+i*4]
    cvtdq2ps m1, [buf+i*4+NNS*4]
    mulps    m2, m_invstddev
    mulps    m3, m_invstddev
    mulps    m0, m2
    mulps    m1, m3 ; could go into the "+1.0" in the sigmoid, for reduced dependency chain
    addps    m0, [r1+i*8+16-NNS*8]
    addps    m1, [r1+i*8+16]
    SIGMOID  m1, m2
    EXP2     m0, m2, m3
    mulps    m1, m0
    addps    m4, m0
    addps    m5, m1
%assign i i+4
%endrep

    ; FIXME merge several instances? or is OOE enough?
    HADDPS   m0, m4
    movss    m2, [ss_5]
    HADDPS   m1, m5
    mulss    m2, [stddev]
    rcpss    m0, m0
    mulss    m1, m2
    mulss    m0, m1
    addss    m0, [mean]
    cvtss2si eax, m0
    ADD rsp, stack_pad
    RET

.zero:
    cvtss2si eax, m2
    ADD rsp, stack_pad
    RET



%if 0
; int test_net(const float *weightsf, const int *dotp, float dc)
cglobal test_net_sse2, 2,2,10
%define m_1   m8
%define m_abs m9
    add      r0, 0x80
    pshufd   m1, m0, 0 ; dc
    mulps    m1, [r0-0x70]
    cvtdq2ps m0, [r1] ; dotp
    subps    m1, [r0-0x60]
    mulps    m0, [r0-0x80]
    subps    m0, m1
    movaps   m_1,   [ps_1]
    movaps   m_abs, [ps_abs]
    movaps   m1, [r0-0x50]
    SIGMOID  m0, m4
    pshufd   m5, m0, 0x39
    movaps   m2, [r0-0x40]
    pshufd   m6, m0, 0x4e
    mulps    m1, m0
    movaps   m3, [r0-0x30]
    pshufd   m7, m0, 0x93
    mulps    m2, m5
    movaps   m4, [r0-0x20]
    mulps    m3, m6
    addps    m1, [r0-0x10]
    mulps    m4, m7
    addps    m2, m3
    mulps    m0, [r0+0x00]
    addps    m1, m4
    mulps    m5, [r0+0x10]
    addps    m1, m2
    mulps    m6, [r0+0x20]
    SIGMOID  m1, m8
    mulps    m7, [r0+0x30]
    pshufd   m2, m1, 0x39
    addps    m0, [r0+0x80]
    addps    m0, m5
    pshufd   m3, m1, 0x4e
    addps    m6, m7
    pshufd   m4, m1, 0x93
    mulps    m1, [r0+0x40]
    addps    m0, m6
    mulps    m2, [r0+0x50]
    addps    m0, m1
    mulps    m3, [r0+0x60]
    mulps    m4, [r0+0x70]
    addps    m2, m3
    addps    m0, m4
    addps    m0, m2
    movhlps  m1, m0
    maxps    m0, m1
    pshuflw  m1, m0, 0xe
    xor     eax, eax
    comiss   m0, m1
    seta     al
    RET
%endif



%ifdef ARCH_X86_64

%macro DOTP0 2
    pshufd   m8, %1, 0x39
    pshufd   m9, %1, 0x4e
    pshufd  m10, %1, 0x93
    movaps  m11, m8
    movaps  m12, m9
    movaps  m13, m10
    mulps    m8, [r0-0x40]
    mulps    m9, [r0-0x30]
    mulps   m10, [r0-0x20]
    mulps    %1, [r0+0x00]
    mulps   m11, [r0+0x10]
    mulps   m12, [r0+0x20]
    mulps   m13, [r0+0x30]
    addps    %2, m8
    addps    %2, m9
    addps    %2, m10
    addps    %1, [r0+0x80]
    addps    %1, m11
    addps    %1, m12
    addps    %1, m13
%endmacro

%macro DOTP1 2
    pshufd   m8, %2, 0x39
    pshufd   m9, %2, 0x4e
    pshufd   m10, %2, 0x93
    mulps    %2, [r0+0x40]
    mulps    m8, [r0+0x50]
    mulps    m9, [r0+0x60]
    mulps    m10, [r0+0x70]
    addps    %1, %2
    addps    %1, m8
    addps    %1, m9
    addps    %1, m10
%endmacro

; int test_net_x4(const float *weightsf, const int (*dotp)[4], float dc0, float dc1, float dc2, float dc3)
cglobal test_net_x4_ssse3, 2,2,16
%define m_1   m14
%define m_abs m15
    add      r0, 0x80
    pshufd   m4, m0, 0
    pshufd   m5, m1, 0
    pshufd   m6, m2, 0
    pshufd   m7, m3, 0
    movaps   m8, [r0-0x60]
    movaps   m9, [r0-0x70]
    movaps  m10, [r0-0x80]
    cvtdq2ps m0, [r1+0x00]
    cvtdq2ps m1, [r1+0x10]
    cvtdq2ps m2, [r1+0x20]
    cvtdq2ps m3, [r1+0x30]
    mulps    m4, m9
    mulps    m5, m9
    mulps    m6, m9
    mulps    m7, m9
    subps    m4, m8
    subps    m5, m8
    subps    m6, m8
    subps    m7, m8
    mulps    m0, m10
    mulps    m1, m10
    mulps    m2, m10
    mulps    m3, m10
    movaps   m_1,   [ps_1]
    movaps   m_abs, [ps_abs]
    subps    m0, m4
    subps    m1, m5
    subps    m2, m6
    subps    m3, m7
    SIGMOID  m0, m4
    SIGMOID  m1, m4
    SIGMOID  m2, m4
    SIGMOID  m3, m4
    movaps   m4, m0
    movaps   m5, m1
    mulps    m4, [r0-0x50]
    mulps    m5, [r0-0x50]
    addps    m4, [r0-0x10]
    addps    m5, [r0-0x10]
    DOTP0    m0, m4
    DOTP0    m1, m5
    SIGMOID  m4, m8
    SIGMOID  m5, m8
    movaps   m6, m2
    movaps   m7, m3
    mulps    m6, [r0-0x50]
    mulps    m7, [r0-0x50]
    addps    m6, [r0-0x10]
    addps    m7, [r0-0x10]
    DOTP0    m2, m6
    DOTP0    m3, m7
    SIGMOID  m6, m8
    SIGMOID  m7, m8
    DOTP1    m0, m4
    DOTP1    m1, m5
    DOTP1    m2, m6
    DOTP1    m3, m7
    movaps     m4, m0
    punpcklqdq m0, m1 ; unpcklpd?
    punpckhqdq m4, m1
    maxps      m4, m0
    movaps     m0, m2
    punpcklqdq m2, m3
    punpckhqdq m0, m3
    maxps      m2, m0
    movaps   m0, m4
    shufps   m4, m2, 0x88
    shufps   m0, m2, 0xdd
    andps    m0, m_abs
    andps    m4, m_abs
    movaps   m2, [shuf_packdb]
    psubd    m0, m4
    psrld    m0, 31
    pshufb   m0, m2 ; the only non-sse2 instruction
    movd    eax, m0
    RET

%else ; X86_32

%macro DOTP0 6 ; sum0, sum1, tmps
    movaps   %2, %1
    movaps   %3, %1
    mulps    %2, [r0-0x50]
    mulps    %3, [r0+0x00]
    pshufd   %4, %1, 0x39
    addps    %2, [r0-0x10]
    addps    %3, [r0+0x80]
    movaps   %5, [r0+0x10]
    mulps    %5, %4
    mulps    %4, [r0-0x40]
    pshufd   %6, %1, 0x4e
    addps    %2, %4
    addps    %3, %5
    movaps   %4, [r0+0x20]
    mulps    %4, %6
    mulps    %6, [r0-0x30]
    pshufd   %1, %1, 0x93
    addps    %3, %4
    addps    %2, %6
    movaps   %5, [r0-0x20]
    mulps    %5, %1
    mulps    %1, [r0+0x30]
    addps    %2, %5
    addps    %1, %3
%endmacro

%macro DOTP1 3 ; sum, in, tmp
    pshufd   %3, %2, 0x39
    mulps    %3, [r0+0x50]
    addps    %1, %3
    pshufd   %3, %2, 0x4e
    mulps    %3, [r0+0x60]
    addps    %1, %3
    pshufd   %3, %2, 0x93
    mulps    %3, [r0+0x70]
    addps    %1, %3
    mulps    %2, [r0+0x40]
    addps    %1, %2
%endmacro

; int test_net_x4(const float *weightsf, const int (*dotp)[4], float dc0, float dc1, float dc2, float dc3)
cglobal test_net_x4_ssse3, 2,2,8
    movd     m4, r2m
    movd     m5, r3m
    movd     m6, r4m
    movd     m7, r5m
%assign stack_pad 0x60+((-stack_offset-gprsize)&15)
    SUB     rsp, stack_pad
    add      r0, 0x80
    pshufd   m4, m4, 0
    pshufd   m5, m5, 0
    pshufd   m6, m6, 0
    pshufd   m7, m7, 0
    movaps   m0, [r0-0x70]
    movaps   m1, [r0-0x60]
    mulps    m4, m0
    mulps    m5, m0
    mulps    m6, m0
    mulps    m7, m0
    subps    m4, m1
    subps    m5, m1
    subps    m6, m1
    subps    m7, m1
    movaps   m3, [r0-0x80]
    cvtdq2ps m0, [r1+0x00]
    cvtdq2ps m1, [r1+0x10]
    cvtdq2ps m2, [r1+0x20]
    mulps    m0, m3
    mulps    m1, m3
    mulps    m2, m3
    cvtdq2ps m3, [r1+0x30]
    mulps    m3, [r0-0x80]
    subps    m0, m4
    subps    m1, m5
    subps    m2, m6
    subps    m3, m7
%define m_1   m6
%define m_abs m7
    movaps   m_1,   [ps_1]
    movaps   m_abs, [ps_abs]
    SIGMOID  m0, m4
    SIGMOID  m1, m4
    SIGMOID  m2, m4
    SIGMOID  m3, m4
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
    ; FIXME duplicate code
    movaps     m4, m0
    punpcklqdq m0, m1 ; unpcklpd?
    punpckhqdq m4, m1
    maxps      m4, m0
    movaps     m0, m2
    punpcklqdq m2, m3
    punpckhqdq m0, m3
    maxps      m2, m0
    movaps   m0, m4
    shufps   m4, m2, 0x88
    shufps   m0, m2, 0xdd
    andps    m0, m_abs
    andps    m4, m_abs
    mova     m2, [shuf_packdb]
    psubd    m0, m4
    psrld    m0, 31
    pshufb   m0, m2 ; the only non-sse2 instruction
    movd    eax, m0
    ADD     rsp, stack_pad
    RET
%endif



%macro BICUBIC_LOOP 1
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

; void bicubic(uint8_t *dst, uint8_t *src, int stride, int width)
cglobal bicubic_ssse3, 4,7,6
    add       r1, r3
    lea       r4, [r1+r2]
    lea       r5, [r1+r2*2]
    lea       r6, [r4+r2*2]
    mova      m4, [pb_38_m6]
    mova      m5, [pw_32]
    test      r0, 15
    jnz .unaligned
    lea       r0, [r0+r3-16]
    neg       r3
    BICUBIC_LOOP mova
    REP_RET
.unaligned:
    lea       r0, [r0+r3-16]
    neg       r3
    BICUBIC_LOOP movu
    REP_RET



; void block_sums_core(float *dst, uint16_t *src, int stride, int width)
cglobal block_sums_core_sse2, 4,7,3
    pxor      m2, m2
    shl       r2, 1
    lea       r1, [r1+r3*2]
    lea       r0, [r0+r3*4-32]
    lea       r4, [r1+r2]
    lea       r5, [r1+r2*2]
    lea       r6, [r4+r2*2]
    neg       r3
align 16
.loop:
    mova      m0, [r1+r3*2]
    paddw     m0, [r4+r3*2]
    paddw     m0, [r5+r3*2]
    paddw     m0, [r6+r3*2]
    add       r3, 8
    mova      m1, m0
    punpcklwd m0, m2
    punpckhwd m1, m2
    cvtdq2ps  m0, m0
    cvtdq2ps  m1, m1
    movaps    [r0+r3*4], m0
    movaps    [r0+r3*4+16], m1
    jl .loop
    REP_RET

