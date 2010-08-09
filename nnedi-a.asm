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

SECTION .text
INIT_XMM

%macro HADDPS 2 ; dst, src
    movhlps    %1, %2
    addps      %1, %2
    pshuflw    %2, %1, 0xe
    addss      %1, %2
%endmacro

%macro HADDPI_X4 5 ; dst, s0, s1, s2, s3
    movdqa     %1, %2
    punpckldq  %2, %3
    punpckhdq  %1, %3
    paddd      %1, %2
    movdqa     %2, %4
    punpckldq  %4, %5
    punpckhdq  %2, %5
    paddd      %2, %4
    movdqa     %4, %1
    punpcklqdq %1, %2
    punpckhqdq %4, %2
    paddd      %1, %4
%endmacro

%macro SIGMOID 2 ; dst, tmp
    movaps   %2, %1
    andps    %1, [ps_abs]
    addps    %1, [ps_1]
    rcpps    %1, %1
    mulps    %1, %2
%endmacro


%macro NOP_PAD 0
    times (($-$$)&15)/14 nop
    times (($-$$)&15)/15 nop
%endmacro

%macro DOTP_LOAD 1
    NOP_PAD
    %assign %%n %1 ; turn arg into a literal number so that it can be used in names
    %ifndef used1
        %xdefine %%i 1
    %elifndef used2
        %xdefine %%i 2
    %elifndef used3
        %xdefine %%i 3
    %elifndef used8
        %xdefine %%i 8
    %elifndef used4
        %xdefine %%i 4
    %elifndef used5
        %xdefine %%i 5
    %elifndef used6
        %xdefine %%i 6
    %elifndef used0
        %xdefine %%i 0
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
    %assign  %%n %1
    %assign  %%j 10 + (%%n % 6)
    %xdefine %%i tmp %+ %%n
    pmaddwd  m %+ %%i, m %+ %%j
%endmacro

%macro DOTP_ACC 1
    NOP_PAD
    %assign  %%n %1
    %assign  %%j %%n/6
    %xdefine %%i tmp %+ %%n
    %xdefine %%k acc %+ %%j
    %if %%n % 6
        paddd m %+ %%k, m %+ %%i
        CAT_UNDEF used, %%i
    %else
        CAT_XDEFINE acc, %%j, %%i
    %endif
    CAT_UNDEF tmp, %%n
%endmacro

cglobal dotproduct_x4
%define stride 48*2
%assign offset 128
    DOTP_LOAD 0
    DOTP_LOAD 1
    DOTP_MUL  0
%assign i 0
%rep 22
    DOTP_LOAD i+2
    DOTP_ACC  i+0
    DOTP_MUL  i+1
%assign i i+1
%endrep
    DOTP_MUL  23
    DOTP_ACC  22
    DOTP_ACC  23
    add      r0, stride*4+128-offset
    HADDPI_X4 m0, m %+ acc0, m %+ acc1, m %+ acc2, m %+ acc3
    ret


cglobal exp2_x4
    movaps   m1, m0
    addps    m0, [ps_exp_bias]
    movaps   m2, m0
    subps    m0, [ps_exp_bias]
    pslld    m2, 23
    subps    m1, m0
    movaps   m0, m1
    mulps    m1, m1
    mulps    m0, [ps_exp_c1]
    mulps    m1, [ps_exp_c2]
    addps    m0, [ps_exp_c0]
    addps    m0, m1
    paddd    m0, m2
    ret


; float scale_net(const int16_t *weightsi, const float *weightsf, const int16_t *pix, float invstddev)
cglobal scale_net_sse2, 3,4,8
    sub      rsp, NNS*8+24
%define invstddev [rsp+NNS*8]
    shufps   m0, m0, 0
    mova     invstddev, m0

    ; load all the pixels into regs, where they will stay throughout the dotproduct pass
    mova     m10, [r2+0x00]
    mova     m11, [r2+0x10]
    mova     m12, [r2+0x20]
    mova     m13, [r2+0x30]
    mova     m14, [r2+0x40]
    mova     m15, [r2+0x50]

    add      r0, 128
    add      r2, 128
%assign i 0
%rep NNS/2
    call dotproduct_x4
    mova     m1, [r1]
    cvtdq2ps m0, m0
    mulps    m1, invstddev
    mulps    m0, m1 ; could go into the "+1.0" in the sigmoid, for reduced dependency chain
    addps    m0, [r1+16]
    mova     [rsp+i*4], m0
%if i==0
    mova     m9, m0
%elif i<NNS
    maxps    m9, m0
%endif
    add      r1, 8*4
%assign i i+4
%endrep

    movhlps  m7, m9
    maxps    m7, m9
    pshuflw  m9, m7, 0xe
    maxss    m7, m9
    shufps   m7, m7, 0

    xorps    m6, m6 ; FIXME
    xorps    m5, m5 ; FIXME
%assign i 0
%rep NNS/4
    mova     m0, [rsp+i*4]
    subps    m0, m7
    call exp2_x4
    mova     m1, [rsp+i*4+NNS*4]
    SIGMOID  m1, m2
    mulps    m1, m0
    addps    m6, m0
    addps    m5, m1
%assign i i+4
%endrep

    HADDPS   m1, m6
    HADDPS   m0, m5
    rcpss    m1, m1
    mulss    m0, [ss_5]
    mulss    m0, m1
    add rsp, NNS*8+24
    RET
