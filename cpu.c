/*****************************************************************************
 * cpu.c: cpu detection
 *****************************************************************************
 * Copyright (C) 2003-2012 x264 project
 *
 * Authors: Loren Merritt <lorenm@u.washington.edu>
 *          Laurent Aimar <fenrir@via.ecp.fr>
 *          Jason Garrett-Glaser <darkshikari@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
 *
 * This program is also available under a commercial proprietary license.
 * For more information, contact us at licensing@x264.com.
 *****************************************************************************/

#define _GNU_SOURCE // for sched_getaffinity

#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "cpu.h"

#if SYS_LINUX
#include <sched.h>
#endif
#if SYS_BEOS
#include <kernel/OS.h>
#endif
#if SYS_MACOSX || SYS_FREEBSD
#include <sys/types.h>
#include <sys/sysctl.h>
#endif
#if SYS_OPENBSD
#include <sys/param.h>
#include <sys/sysctl.h>
#include <machine/cpu.h>
#endif

#if (ARCH_PPC && SYS_LINUX) || (ARCH_ARM && !HAVE_NEON)
#include <signal.h>
#include <setjmp.h>
static sigjmp_buf jmpbuf;
static volatile sig_atomic_t canjump = 0;

static void sigill_handler( int sig )
{
    if( !canjump )
    {
        signal( sig, SIG_DFL );
        raise( sig );
    }

    canjump = 0;
    siglongjmp( jmpbuf, 1 );
}
#endif

#if ARCH_X86
int nnedi_cpu_cpuid_test( void );
void nnedi_cpu_cpuid( uint32_t op, uint32_t *eax, uint32_t *ebx, uint32_t *ecx, uint32_t *edx );
void nnedi_cpu_xgetbv( uint32_t op, uint32_t *eax, uint32_t *edx );
void nnedi_cpu_mask_misalign_sse( void );

uint32_t nnedi_cpu_detect( void )
{
    uint32_t cpu = 0;
    uint32_t eax, ebx, ecx, edx;
    uint32_t vendor[4] = {0};
    uint32_t max_extended_cap;
    int cache;

#if !ARCH_X86_64
    if( !nnedi_cpu_cpuid_test() )
        return 0;
#endif

    nnedi_cpu_cpuid( 0, &eax, vendor+0, vendor+2, vendor+1 );
    if( eax == 0 )
        return 0;

    nnedi_cpu_cpuid( 1, &eax, &ebx, &ecx, &edx );
    if( edx&0x00800000 )
        cpu |= NNEDI_CPU_MMX;
    else
        return 0;
    if( edx&0x02000000 )
        cpu |= NNEDI_CPU_MMX2|NNEDI_CPU_SSE;
    if( edx&0x04000000 )
        cpu |= NNEDI_CPU_SSE2;
    if( ecx&0x00000001 )
        cpu |= NNEDI_CPU_SSE3;
    if( ecx&0x00000200 )
        cpu |= NNEDI_CPU_SSSE3;
    if( ecx&0x00080000 )
        cpu |= NNEDI_CPU_SSE4;
    if( ecx&0x00100000 )
        cpu |= NNEDI_CPU_SSE42;
    /* Check OXSAVE and AVX bits */
    if( (ecx&0x18000000) == 0x18000000 )
    {
        /* Check for OS support */
        nnedi_cpu_xgetbv( 0, &eax, &edx );
        if( (eax&0x6) == 0x6 )
        {
            cpu |= NNEDI_CPU_AVX;
            if( ecx&0x00001000 )
                cpu |= NNEDI_CPU_FMA3;
        }
    }

    nnedi_cpu_cpuid( 7, &eax, &ebx, &ecx, &edx );
    /* AVX2 requires OS support, but BMI1/2 don't. */
    if( (cpu&NNEDI_CPU_AVX) && (ebx&0x00000020) )
        cpu |= NNEDI_CPU_AVX2;
    if( ebx&0x00000008 )
    {
        cpu |= NNEDI_CPU_BMI1;
        if( ebx&0x00000100 )
            cpu |= NNEDI_CPU_BMI2;
    }

    if( cpu & NNEDI_CPU_SSSE3 )
        cpu |= NNEDI_CPU_SSE2_IS_FAST;
    if( cpu & NNEDI_CPU_SSE4 )
        cpu |= NNEDI_CPU_SHUFFLE_IS_FAST;

    nnedi_cpu_cpuid( 0x80000000, &eax, &ebx, &ecx, &edx );
    max_extended_cap = eax;

    if( !strcmp((char*)vendor, "AuthenticAMD") && max_extended_cap >= 0x80000001 )
    {
        cpu |= NNEDI_CPU_SLOW_CTZ;
        nnedi_cpu_cpuid( 0x80000001, &eax, &ebx, &ecx, &edx );
        if( edx&0x00400000 )
            cpu |= NNEDI_CPU_MMX2;
        if( cpu & NNEDI_CPU_SSE2 )
        {
            if( ecx&0x00000040 ) /* SSE4a */
            {
                cpu |= NNEDI_CPU_SSE2_IS_FAST;
                cpu |= NNEDI_CPU_LZCNT;
                cpu |= NNEDI_CPU_SHUFFLE_IS_FAST;
                cpu &= ~NNEDI_CPU_SLOW_CTZ;
            }
            else
                cpu |= NNEDI_CPU_SSE2_IS_SLOW;

            if( ecx&0x00000080 ) /* Misalign SSE */
            {
                cpu |= NNEDI_CPU_SSE_MISALIGN;
                nnedi_cpu_mask_misalign_sse();
            }

            if( cpu & NNEDI_CPU_AVX )
            {
                if( ecx&0x00000800 ) /* XOP */
                    cpu |= NNEDI_CPU_XOP;
                if( ecx&0x00010000 ) /* FMA4 */
                    cpu |= NNEDI_CPU_FMA4;
            }

            if( ecx&0x00200000 )
                cpu |= NNEDI_CPU_TBM;
        }
    }

    if( !strcmp((char*)vendor, "GenuineIntel") )
    {
        nnedi_cpu_cpuid( 1, &eax, &ebx, &ecx, &edx );
        int family = ((eax>>8)&0xf) + ((eax>>20)&0xff);
        int model  = ((eax>>4)&0xf) + ((eax>>12)&0xf0);
        if( family == 6 )
        {
            /* 6/9 (pentium-m "banias"), 6/13 (pentium-m "dothan"), and 6/14 (core1 "yonah")
             * theoretically support sse2, but it's significantly slower than mmx for
             * almost all of x264's functions, so let's just pretend they don't. */
            if( model == 9 || model == 13 || model == 14 )
            {
                cpu &= ~(NNEDI_CPU_SSE2|NNEDI_CPU_SSE3);
                assert(!(cpu&(NNEDI_CPU_SSSE3|NNEDI_CPU_SSE4)));
            }
            /* Detect Atom CPU */
            else if( model == 28 )
            {
                cpu |= NNEDI_CPU_SLOW_ATOM;
                cpu |= NNEDI_CPU_SLOW_CTZ;
            }
            /* Some Penryns and Nehalems are pointlessly crippled (SSE4 disabled), so
             * detect them here. */
            else if( model >= 23 )
                cpu |= NNEDI_CPU_SHUFFLE_IS_FAST;
        }
    }

    if( (!strcmp((char*)vendor, "GenuineIntel") || !strcmp((char*)vendor, "CyrixInstead")) && !(cpu&NNEDI_CPU_SSE42))
    {
        /* cacheline size is specified in 3 places, any of which may be missing */
        nnedi_cpu_cpuid( 1, &eax, &ebx, &ecx, &edx );
        cache = (ebx&0xff00)>>5; // cflush size
        if( !cache && max_extended_cap >= 0x80000006 )
        {
            nnedi_cpu_cpuid( 0x80000006, &eax, &ebx, &ecx, &edx );
            cache = ecx&0xff; // cacheline size
        }
        if( !cache )
        {
            // Cache and TLB Information
            static const char cache32_ids[] = { 0x0a, 0x0c, 0x41, 0x42, 0x43, 0x44, 0x45, 0x82, 0x83, 0x84, 0x85, 0 };
            static const char cache64_ids[] = { 0x22, 0x23, 0x25, 0x29, 0x2c, 0x46, 0x47, 0x49, 0x60, 0x66, 0x67,
                                                0x68, 0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7c, 0x7f, 0x86, 0x87, 0 };
            uint32_t buf[4];
            int max, i = 0;
            do {
                nnedi_cpu_cpuid( 2, buf+0, buf+1, buf+2, buf+3 );
                max = buf[0]&0xff;
                buf[0] &= ~0xff;
                for( int j = 0; j < 4; j++ )
                    if( !(buf[j]>>31) )
                        while( buf[j] )
                        {
                            if( strchr( cache32_ids, buf[j]&0xff ) )
                                cache = 32;
                            if( strchr( cache64_ids, buf[j]&0xff ) )
                                cache = 64;
                            buf[j] >>= 8;
                        }
            } while( ++i < max );
        }

        if( cache == 32 )
            cpu |= NNEDI_CPU_CACHELINE_32;
        else if( cache == 64 )
            cpu |= NNEDI_CPU_CACHELINE_64;
        else
            fprintf( stderr, "nnedi: unable to determine cacheline size\n" );
    }

#if BROKEN_STACK_ALIGNMENT
    cpu |= NNEDI_CPU_STACK_MOD4;
#endif

    return cpu;
}

#elif ARCH_PPC

#if SYS_MACOSX || SYS_OPENBSD
#include <sys/sysctl.h>
uint32_t nnedi_cpu_detect( void )
{
    /* Thank you VLC */
    uint32_t cpu = 0;
#if SYS_OPENBSD
    int      selectors[2] = { CTL_MACHDEP, CPU_ALTIVEC };
#else
    int      selectors[2] = { CTL_HW, HW_VECTORUNIT };
#endif
    int      has_altivec = 0;
    size_t   length = sizeof( has_altivec );
    int      error = sysctl( selectors, 2, &has_altivec, &length, NULL, 0 );

    if( error == 0 && has_altivec != 0 )
        cpu |= NNEDI_CPU_ALTIVEC;

    return cpu;
}

#elif SYS_LINUX

uint32_t nnedi_cpu_detect( void )
{
    static void (*oldsig)( int );

    oldsig = signal( SIGILL, sigill_handler );
    if( sigsetjmp( jmpbuf, 1 ) )
    {
        signal( SIGILL, oldsig );
        return 0;
    }

    canjump = 1;
    asm volatile( "mtspr 256, %0\n\t"
                  "vand 0, 0, 0\n\t"
                  :
                  : "r"(-1) );
    canjump = 0;

    signal( SIGILL, oldsig );

    return NNEDI_CPU_ALTIVEC;
}
#endif

#elif ARCH_ARM

void nnedi_cpu_neon_test( void );
int nnedi_cpu_fast_neon_mrc_test( void );

uint32_t nnedi_cpu_detect( void )
{
    int flags = 0;
#if HAVE_ARMV6
    flags |= NNEDI_CPU_ARMV6;

    // don't do this hack if compiled with -mfpu=neon
#if !HAVE_NEON
    static void (* oldsig)( int );
    oldsig = signal( SIGILL, sigill_handler );
    if( sigsetjmp( jmpbuf, 1 ) )
    {
        signal( SIGILL, oldsig );
        return flags;
    }

    canjump = 1;
    nnedi_cpu_neon_test();
    canjump = 0;
    signal( SIGILL, oldsig );
#endif

    flags |= NNEDI_CPU_NEON;

    // fast neon -> arm (Cortex-A9) detection relies on user access to the
    // cycle counter; this assumes ARMv7 performance counters.
    // NEON requires at least ARMv7, ARMv8 may require changes here, but
    // hopefully this hacky detection method will have been replaced by then.
    // Note that there is potential for a race condition if another program or
    // x264 instance disables or reinits the counters while x264 is using them,
    // which may result in incorrect detection and the counters stuck enabled.
    flags |= nnedi_cpu_fast_neon_mrc_test() ? NNEDI_CPU_FAST_NEON_MRC : 0;
    // TODO: write dual issue test? currently it's A8 (dual issue) vs. A9 (fast mrc)
#endif
    return flags;
}

#else

uint32_t nnedi_cpu_detect( void )
{
    return 0;
}

#endif

int nnedi_cpu_num_processors( void )
{
#if SYS_WINDOWS
    return nnedi_pthread_num_processors_np();

#elif SYS_CYGWIN
    return sysconf( _SC_NPROCESSORS_ONLN );

#elif SYS_LINUX
    cpu_set_t p_aff;
    memset( &p_aff, 0, sizeof(p_aff) );
    if( sched_getaffinity( 0, sizeof(p_aff), &p_aff ) )
        return 1;
#if HAVE_CPU_COUNT
    return CPU_COUNT(&p_aff);
#else
    int np = 0;
    for( unsigned int bit = 0; bit < 8 * sizeof(p_aff); bit++ )
        np += (((uint8_t *)&p_aff)[bit / 8] >> (bit % 8)) & 1;
    return np;
#endif

#elif SYS_BEOS
    system_info info;
    get_system_info( &info );
    return info.cpu_count;

#elif SYS_MACOSX || SYS_FREEBSD || SYS_OPENBSD
    int ncpu;
    size_t length = sizeof( ncpu );
#if SYS_OPENBSD
    int mib[2] = { CTL_HW, HW_NCPU };
    if( sysctl(mib, 2, &ncpu, &length, NULL, 0) )
#else
    if( sysctlbyname("hw.ncpu", &ncpu, &length, NULL, 0) )
#endif
    {
        ncpu = 1;
    }
    return ncpu;

#else
    return 1;
#endif
}
