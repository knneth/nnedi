%include "x86inc.asm"
%include "nnedi-a.h"

SECTION_RODATA
ps_exp_bias: times 4 dd 12582912.0 ; 3<<22
ps_exp_c0:   times 4 dd 1.00035
ps_exp_c1:   times 4 dd 0.701277797 ; 1.01173*M_LN2
ps_exp_c2:   times 4 dd 0.237348593 ; 0.49401*M_LN2*M_LN2
ps_1:        times 4 dd 1.0
ps_abs:      times 4 dd 0x7fffffff
ss_5:        dd 5.0
ss_48        dd 48.0
ss_1_3:      dd 0.3333333333
ss_1_16:     dd 0.0625

%define m_invstddev m9
%define m_exp_bias m10
%define m_exp_c0   m11
%define m_exp_c1   m12
%define m_exp_c2   m13
%define m_1        m14
%define m_abs      m15

SECTION .text
INIT_XMM


; void cast_pixels_scale(int16_t *dst, const uint8_t *src, intptr_t stride, float *mean_stddev_inv)
cglobal cast_pixels_scale_sse2, 4,7,16
    lea        r4, [r2*3]
    add        r4, r1
    pxor       m0, m0
    movq       m1, [r1]
    movq       m3, [r1+r2]
    movdqa     m2, m1
    punpcklqdq m2, m3
    punpcklbw  m1, m0
    punpcklbw  m3, m0
    movdqa    m10, m1
    movdqa    m11, m3
    pmaddwd    m1, m1
    pmaddwd    m3, m3
    psadbw     m2, m0
    paddd      m1, m3

    movq       m3, [r1+r2*2]
    movq       m5, [r4]
    movdqa     m4, m3
    punpcklqdq m4, m5
    punpcklbw  m3, m0
    punpcklbw  m5, m0
    movdqa    m12, m3
    movdqa    m13, m5
    pmaddwd    m3, m3
    pmaddwd    m5, m5
    psadbw     m4, m0
    paddd      m3, m5
    paddd      m2, m4
    paddd      m1, m3

    movq       m3, [r4+r2]
    movq       m5, [r4+r2*2]
    movdqa     m4, m3
    punpcklqdq m4, m5
    punpcklbw  m3, m0
    punpcklbw  m5, m0
    movdqa    m14, m3
    movdqa    m15, m5
    pmaddwd    m3, m3
    pmaddwd    m5, m5
    psadbw     m4, m0
    paddd      m3, m5
    paddd      m2, m4
    paddd      m1, m3

    movhlps    m4, m2
    movhlps    m3, m1
    paddd      m2, m4
    paddd      m1, m3
    pshuflw    m3, m1, 14
    paddd      m1, m3
    movd      r5d, m2
    movd      r6d, m1

    xorps      m2, m2
    cvtsi2ss   m2, r5d
    imul      r6d, 48
    mulss      m2, [ss_1_3]
    cvtps2dq   m3, m2
    imul      r5d, r5d
    mulss      m2, [ss_1_16]
    sub       r6d, r5d
    jle .zero
    cvtsi2ss   m1, r6d
    movss    [r3], m2
    rsqrtss    m1, m1
    mulss      m1, [ss_48]
    movss  [r3+8], m1
    rcpss      m1, m1
    movss  [r3+4], m1

    pshuflw    m3, m3, 0
    punpcklqdq m3, m3
    psllw     m10, 4
    psllw     m11, 4
    psllw     m12, 4
    psllw     m13, 4
    psllw     m14, 4
    psllw     m15, 4
    psubw     m10, m3
    psubw     m11, m3
    psubw     m12, m3
    psubw     m13, m3
    psubw     m14, m3
    psubw     m15, m3
    mova [r0+0x00], m10
    mova [r0+0x10], m11
    mova [r0+0x20], m12
    mova [r0+0x30], m13
    mova [r0+0x40], m14
    mova [r0+0x50], m15
    RET

.zero:
    movss     [r3], m2
    mov dword [r3+4], 0
    mov dword [r3+8], 0
    RET



%macro HADDPS 2 ; dst, src
    movhlps    %1, %2
    addps      %1, %2
    pshuflw    %2, %1, 0xe
    addss      %1, %2
%endmacro

%macro HADDPI_X4 5 ; dst, s0, s1, s2, s3
    ; optimized for conroe.
    ; nehalem probably prefers 6x punpck.
    movdqa     %1, %2
    punpcklqdq %2, %3
    punpckhqdq %1, %3
    paddd      %1, %2
    movdqa     %2, %4
    punpcklqdq %4, %5
    punpckhqdq %2, %5
    paddd      %2, %4
    movdqa     %4, %1
    shufps     %1, %2, 0x88
    shufps     %4, %2, 0xdd
    paddd      %1, %4
%endmacro


%macro NOP_PAD 0
;   times (($-$$)&15)/14 nop
;   times (($-$$)&15)/15 nop
%endmacro

%macro DOTP_LOAD 1
    NOP_PAD
    %assign %%n %1 ; turn arg into a literal number so that it can be used in names
    %if (%%n % 6) == 0
        %assign %%i %%n/6
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

%macro DOTP_MUL  1
    NOP_PAD
    %assign %%n %1
    %assign %%j 10 + (%%n % 6)
    %assign %%i tmp %+ %%n
    pmaddwd m %+ %%i, m %+ %%j
%endmacro

%macro DOTP_ACC 1
    NOP_PAD
    %assign %%n %1
    %assign %%j %%n/6
    %assign %%i tmp %+ %%n
    %if %%n % 6
        paddd m %+ %%j, m %+ %%i
        CAT_UNDEF used, %%i
    %endif
    CAT_UNDEF tmp, %%n
%endmacro

cglobal dotproducts
%define stride 48*2
%assign offset 128
.loop:
    movdqa     m1, m2
    punpcklqdq m2, m3
    DOTP_LOAD 0
    punpckhqdq m1, m3
    DOTP_LOAD 1
    paddd      m2, m1
    movdqa     m3, m9
    DOTP_LOAD 2
    DOTP_MUL  0
    shufps     m9, m2, 0x88
    DOTP_LOAD 3
    DOTP_MUL  1
    shufps     m3, m2, 0xdd
    DOTP_LOAD 4
    DOTP_ACC  0
    paddd      m9, m3
    DOTP_MUL  2
    mova [r2+r4-16], m9
.skip: ; FIXME skip the hadd on the first iteration
%assign i 1
%rep 19
    DOTP_LOAD i+4
    DOTP_ACC  i+0
    DOTP_MUL  i+2
%assign i i+1
%endrep
    DOTP_ACC  20
    mova       m9, m0
    DOTP_MUL  22
    punpcklqdq m0, m1
    DOTP_ACC  21
    punpckhqdq m9, m1
    DOTP_MUL  23
    paddd      m9, m0
    DOTP_ACC  22
    add        r0, stride*4+128-offset
    add        r4, 16
    DOTP_ACC  23
    jl .loop
    movdqa     m1, m2
    punpcklqdq m2, m3
    punpckhqdq m1, m3
    paddd      m2, m1
    movdqa     m3, m9
    shufps     m9, m2, 0x88
    shufps     m3, m2, 0xdd
    paddd      m9, m3
    mova [r2+r4-16], m9
    ret


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

cglobal exp2_and_sigmoid
    SIGMOID m1, m2
    EXP2 m0, m2, m3
    ret


; int cale_net(const int16_t *weightsi, const float *weightsf, const int16_t *pix, float *mean_stddev_inv)
cglobal scale_net_sse2, 3,5,16
    sub      rsp, NNS*8+24
%define buf rsp+16

    ; load all the pixels into regs, where they will stay throughout the dotproduct pass
    mova     m10, [r2+0x00]
    mova     m11, [r2+0x10]
    mova     m12, [r2+0x20]
    mova     m13, [r2+0x30]
    mova     m14, [r2+0x40]
    mova     m15, [r2+0x50]

    add      r0, 128
    lea      r2, [buf+NNS*8]
    mov      r4, -NNS*8
    call dotproducts

    movss    m_invstddev, [r3+8]
    mova     m_exp_bias, [ps_exp_bias]
    mova     m_exp_c0,   [ps_exp_c0]
    shufps   m_invstddev, m_invstddev, 0
    mova     m_exp_c1,   [ps_exp_c1]
    mova     m_exp_c2,   [ps_exp_c2]
    mova     m_1,        [ps_1]
    mova     m_abs,      [ps_abs]

    xorps    m5, m5
    xorps    m6, m6
%assign i 0
%rep NNS/4
    mova     m2, [r1]
    mova     m3, [r1+NNS*8]
    cvtdq2ps m0, [buf+i*4]
    cvtdq2ps m1, [buf+i*4+NNS*4]
    mulps    m2, m_invstddev
    mulps    m3, m_invstddev
    mulps    m0, m2
    mulps    m1, m3 ; could go into the "+1.0" in the sigmoid, for reduced dependency chain
    addps    m0, [r1+16]
    addps    m1, [r1+16+NNS*8]
    add      r1, 32

    call exp2_and_sigmoid
    mulps    m1, m0
    addps    m5, m0
    addps    m6, m1
%assign i i+4
%endrep

    HADDPS   m0, m5
    movss    m2, [ss_5]
    HADDPS   m1, m6
    mulss    m2, [r3+4]
    rcpss    m0, m0
    mulss    m1, m2
    mulss    m0, m1
    addss    m0, [r3]
    cvtss2si eax, m0
    add rsp, NNS*8+24
    RET
