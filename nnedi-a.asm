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

%macro HADDPS_X4 5 ; dst, s0, s1, s2, s3
    movaps     %1, %2
    punpcklqdq %2, %3
    punpckhqdq %1, %3
    addps      %1, %2
    movaps     %2, %4
    punpcklqdq %4, %5
    punpckhqdq %2, %5
    addps      %2, %4
    movaps     %4, %1
    shufps     %1, %2, 0x88
    shufps     %4, %2, 0xdd
    addps      %1, %4
%endmacro


%macro NOP_PAD 0 ; FIXME remove
;   times (($-$$)&15)/14 nop
;   times (($-$$)&15)/15 nop
%endmacro

%macro DOTP_LOAD 1
    NOP_PAD
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
    NOP_PAD
    %assign %%n %1
    %assign %%j 10 + (%%n / 16)
    %assign %%i tmp %+ %%n
    pmaddwd m %+ %%i, m %+ %%j
    %if (%%n & 3) == 3
        pshufd  m %+ %%j, m %+ %%j, 0x39
;       palignr m %+ %%j, m %+ %%j, 4
    %endif
%endmacro

%macro DOTP_ACC 1
    NOP_PAD
    %assign %%n %1
    %assign %%i tmp %+ %%n
    %if %%n >= 4
        %assign %%j %%n&3
        paddd m %+ %%j, m %+ %%i
        CAT_UNDEF used, %%i
    %endif
    CAT_UNDEF tmp, %%n
%endmacro

cglobal dotproducts
%define stride 48*2
%assign offset 128 ; FIXME could be 0 (possibly without any loss)
    DOTP_LOAD 0
    DOTP_LOAD 1
    DOTP_LOAD 2
    DOTP_MUL  0
    DOTP_LOAD 3
    DOTP_MUL  1
%assign i 0
%rep 92
    DOTP_LOAD i+4
    DOTP_ACC  i+0
    DOTP_MUL  i+2
%assign i i+1
%endrep
    DOTP_ACC  92
    DOTP_MUL  94
    DOTP_ACC  93
    mova  [r2], m0
    DOTP_MUL  95
    DOTP_ACC  94
    mova  [r2+16], m1
    add        r0, stride*16+128-offset
    DOTP_ACC  95
    mova  [r2+32], m2
    mova  [r2+48], m3
    add        r2, 64
    ret


%macro DOTP_MUL2 1
    NOP_PAD
    %assign %%n %1
    %assign %%j 10 + (%%n / 4)
    %assign %%i tmp %+ %%n
    pmaddwd m %+ %%i, m %+ %%j
%endmacro

; v4si test_dotproduct(const int16_t *weightsi)
cglobal test_dotproduct_sse2, 1,1
%assign offset 0
    SWAP 0, 4
    ; FIXME this ordering isn't doing as much good as I might expect
    DOTP_LOAD 0
    DOTP_LOAD 1
    DOTP_LOAD 2
    DOTP_MUL2 0
    DOTP_LOAD 3
    DOTP_MUL2 1
%assign i 0
%rep 20
    DOTP_LOAD i+4
    DOTP_ACC  i+0
    DOTP_MUL2 i+2
%assign i i+1
%endrep
    DOTP_ACC  20
    DOTP_MUL2 22
    DOTP_ACC  21
    DOTP_MUL2 23
    DOTP_ACC  22
    DOTP_ACC  23
    HADDPI_X4 m4, m0, m1, m2, m3 ; FIXME partly interleave with the above; or do multiple test_nets at once.
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


; int scale_one(const int16_t *weightsi, const float *weightsf, const uint8_t *pix, int stride)
cglobal scale_one_sse2, 4,6,16
    sub      rsp, NNS*8+24
%define buf rsp
%define mean buf+NNS*8
%define stddev mean+4
%define invstddev stddev+4

    ; compute sum and sum squared
    lea        r4, [r3*3]
    add        r4, r2
    pxor       m0, m0
    LOAD_SUM_SQUARE m10, m11, m1, m2, m3, [r2], [r2+r3]
    LOAD_SUM_SQUARE m12, m13, m3, m4, m5, [r2+r3*2], [r4]
    paddd      m2, m4
    paddd      m1, m3
    LOAD_SUM_SQUARE m14, m15, m3, m4, m5, [r4+r3], [r4+r3*2]
    paddd      m2, m4
    paddd      m1, m3
    movhlps    m4, m2
    movhlps    m3, m1
    paddd      m2, m4
    paddd      m1, m3
    pshuflw    m3, m1, 14
    paddd      m1, m3
    movd      r4d, m2
    movd      r5d, m1

    ; compute means and stddev
    xorps      m2, m2
    cvtsi2ss   m2, r4d
    imul      r5d, 48
    mulss      m2, [ss_1_3]
    cvtps2dq   m3, m2
    imul      r4d, r4d
    mulss      m2, [ss_1_16]
    sub       r5d, r4d
    jle .zero
    cvtsi2ss   m1, r5d
    movss  [mean], m2
    rsqrtss    m1, m1
    mulss      m1, [ss_48]
    movss  [invstddev], m1
    rcpss      m1, m1
    movss  [stddev], m1

    ; remove mean
    pshuflw    m3, m3, 0
    punpcklqdq m3, m3
    psllw     m10, 4
    psubw     m10, m3
    psllw     m11, 4
    psubw     m11, m3
    psllw     m12, 4
    psubw     m12, m3
    psllw     m13, 4
    psubw     m13, m3
    psllw     m14, 4
    psubw     m14, m3
    psllw     m15, 4
    psubw     m15, m3

    ; neural net
    add      r0, 128
    mov      r2, buf
%rep NNS/8
    call dotproducts
%endrep

    movss    m_invstddev, [invstddev]
    movaps   m_exp_bias, [ps_exp_bias]
    movaps   m_exp_c0,   [ps_exp_c0]
    shufps   m_invstddev, m_invstddev, 0
    movaps   m_exp_c1,   [ps_exp_c1]
    movaps   m_exp_c2,   [ps_exp_c2]
    movaps   m_1,        [ps_1]
    movaps   m_abs,      [ps_abs]

    add      r1, NNS*8
    xorps    m5, m5
    xorps    m6, m6
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
    addps    m5, m0
    addps    m6, m1
%assign i i+4
%endrep

    HADDPS   m0, m5
    movss    m2, [ss_5]
    HADDPS   m1, m6
    mulss    m2, [stddev]
    rcpss    m0, m0
    mulss    m1, m2
    mulss    m0, m1
    addss    m0, [mean]
    cvtss2si eax, m0
    add rsp, NNS*8+24
    RET

.zero:
    cvtss2si eax, m2
    add rsp, NNS*8+24
    RET



; FIXME keeps data in xmmregs across function calls, which isn't win64 compatible
; void cast_testblock(uint8_t *pix, int stride)
cglobal cast_testblock_sse2, 2,4
    lea        r2, [r1*3]
    movq      m10, [r0]
    movd      m11, [r0+8]
    movd       m1, [r0+r1]
    movq      m12, [r0+r1+4]
    movq      m13, [r0+r1*2]
    movd      m14, [r0+r1*2+8]
    movd       m2, [r0+r2]
    movq      m15, [r0+r2+4]
    pxor       m0, m0
    punpckldq m11, m1
    punpckldq m14, m2
    punpcklbw m10, m0
    punpcklbw m12, m0
    punpcklbw m13, m0
    punpcklbw m11, m0
    punpcklbw m14, m0
    punpcklbw m15, m0
    RET

; void shift_testblock(uint8_t *pix, int stride)
cglobal shift_testblock_sse2, 2,4
%define buf rsp-24
    lea      r2, [r1*3]
    movzx   r3d, byte [r0]
    mov     [buf+0], r3w
    movzx   r3d, byte [r0+r1]
    mov     [buf+2], r3w
    movzx   r3d, byte [r0+r1*2]
    mov     [buf+4], r3w
    movzx   r3d, byte [r0+r2]
    mov     [buf+6], r3w
    movzx   r3d, byte [r0+1]
    mov     [buf+8], r3w
    movzx   r3d, byte [r0+r1+1]
    mov     [buf+10], r3w
    movzx   r3d, byte [r0+r1*2+1]
    mov     [buf+12], r3w
    movzx   r3d, byte [r0+r2+1]
    mov     [buf+14], r3w
    mova    m10, m11
    mova    m11, m12
    mova    m12, m13
    mova    m13, m14
    mova    m14, m15
    mova    m15, [buf]
    RET


; int test_net(const float *weightsf, v4si dotp, float mean)
cglobal test_net_sse2, 1,1
    add      r0, 0x80
    pshufd   m9, m1, 0 ; mean
    mulps    m9, [r0-0x70]
    cvtdq2ps m0, m0 ; dotp
    subps    m9, [r0-0x60]
    mulps    m0, [r0-0x80]
    subps    m0, m9
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

; int test_net_x2(const float *weightsf, v4si dotp0, v4si dotp1, float mean0, float mean1)
cglobal test_net_x2_sse2, 1,1
    add      r0, 0x80

    movaps   m5, [r0-0x70]
    pshufd   m2, m2, 0 ; mean0
    pshufd   m3, m3, 0 ; mean1
    movaps   m4, [r0-0x60]
    movaps   m6, [r0-0x80]
    mulps    m2, m5
    mulps    m3, m5
    cvtdq2ps m0, m0 ; dotp0
    cvtdq2ps m1, m1 ; dotp1
    subps    m2, m4
    subps    m3, m4
    mulps    m0, m6
    mulps    m1, m6
    movaps   m_1,   [ps_1]
    movaps   m_abs, [ps_abs]
    subps    m0, m2
    subps    m1, m3
    SIGMOID  m0, m5
    SIGMOID  m1, m7
    movaps   m2, [r0-0x50]
    movaps   m4, [r0-0x40]
    movaps   m6, [r0-0x30]
    movaps   m8, [r0-0x20]
    movaps   m3, m2
    movaps   m5, m4
    movaps   m7, m6
    movaps   m9, m8
    pshufd   m10, m0, 0x39
    pshufd   m11, m1, 0x39
    pshufd   m12, m0, 0x4e
    pshufd   m13, m1, 0x4e
    pshufd   m14, m0, 0x93
    pshufd   m15, m1, 0x93
    mulps    m2, m0
    mulps    m3, m1
    mulps    m4, m10
    mulps    m5, m11
    mulps    m6, m12
    mulps    m7, m13
    mulps    m8, m14
    mulps    m9, m15
    addps    m2, [r0-0x10]
    addps    m3, [r0-0x10]
    addps    m4, m6
    addps    m5, m7
    addps    m2, m8
    addps    m3, m9
    addps    m2, m4
    addps    m3, m5
    movaps   m6, [r0+0x00]
    movaps   m7, [r0+0x10]
    movaps   m8, [r0+0x20]
    movaps   m9, [r0+0x30]
    movaps   m5, [r0+0x80]
    mulps    m0, m6
    mulps    m1, m6
    mulps    m10, m7
    mulps    m11, m7
    mulps    m12, m8
    mulps    m13, m8
    mulps    m14, m9
    mulps    m15, m9
    addps    m0, m5
    addps    m1, m5
    addps    m0, m10
    addps    m1, m11
    addps    m12, m14
    addps    m13, m15
    addps    m0, m12
    addps    m1, m13
    movaps   m_1,   [ps_1]
    movaps   m_abs, [ps_abs]
    SIGMOID  m2, m4
    SIGMOID  m3, m5
    pshufd   m4, m2, 0x39
    pshufd   m5, m3, 0x39
    pshufd   m6, m2, 0x4e
    pshufd   m7, m3, 0x4e
    pshufd   m8, m2, 0x93
    pshufd   m9, m3, 0x93
    mulps    m2, [r0+0x40]
    mulps    m3, [r0+0x40]
    mulps    m4, [r0+0x50]
    mulps    m5, [r0+0x50]
    mulps    m6, [r0+0x60]
    mulps    m7, [r0+0x60]
    mulps    m8, [r0+0x70]
    mulps    m9, [r0+0x70]
    addps    m0, m2
    addps    m1, m3
    addps    m4, m6
    addps    m5, m7
    addps    m0, m8
    addps    m1, m9
    addps    m0, m4
    addps    m1, m5
    movhlps  m2, m0
    movhlps  m3, m1
    maxps    m0, m2
    maxps    m1, m3
    pshuflw  m2, m0, 0xe
    pshuflw  m3, m1, 0xe
    xor     r1d, r1d
    xor     eax, eax
    comiss   m0, m2
    seta    r1b
    comiss   m1, m3
    seta     ah
    or      eax, r1d
    RET
