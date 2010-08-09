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


%define stride 48*2
dotproduct_x4:
%assign i -128
    mova     m7, [r2+i]
    mova     m4, [r0+i]
    mova     m5, [r0+i+stride]
    mova     m6, [r0+i+stride*2]
    pmaddwd  m4, m7
    pmaddwd  m5, m7
    pmaddwd  m6, m7
    pmaddwd  m7, [r0+i+stride*3]
%rep 5
%assign i i+16
    mova     m3, [r2+i]
    mova     m0, [r0+i]
    mova     m1, [r0+i+stride]
    mova     m2, [r0+i+stride*2]
    pmaddwd  m0, m3
    pmaddwd  m1, m3
    paddd    m4, m0
    pmaddwd  m2, m3
    paddd    m5, m1
    pmaddwd  m3, [r0+i+stride*3]
    paddd    m6, m2
    paddd    m7, m3
%endrep
    add      r0, stride*4
    HADDPI_X4 m0, m4, m5, m6, m7
    ret


exp2_x4:
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
    mova     m8, m0
%elif i<NNS
    maxps    m8, m0
%endif
    add      r1, 8*4
%assign i i+4
%endrep

    movhlps  m7, m8
    maxps    m7, m8
    pshuflw  m8, m7, 0xe
    maxss    m7, m8
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
