/* -*- c++ -*- */
/*
 * Copyright 2023 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_reciprocal_32f
 *
 * \b Overview
 *
 * Computes the reciprocal of the input vector and stores the results
 * in the output vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_reciprocal_32f(float* out, const float* in, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li in: A pointer to the input vector of floats.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li bVector: A pointer to the output vector of floats.
 *
 * \b Example
 * \code
    int N = 10;
    unsigned int alignment = volk_get_alignment();
    float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
    float* out = (float*)volk_malloc(sizeof(float)*N, alignment);

    for(unsigned int ii = 1; ii < N; ++ii){
        in[ii] = (float)(ii*ii);
    }

    volk_32f_reciprocal_32f(out, in, N);

    for(unsigned int ii = 0; ii < N; ++ii){
        printf("out(%i) = %f\n", ii, out[ii]);
    }

    volk_free(in);
    volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_reciprocal_32f_a_H
#define INCLUDED_volk_32f_reciprocal_32f_a_H

#ifdef LV_HAVE_GENERIC
static inline void
volk_32f_reciprocal_32f_generic(float* out, const float* in, unsigned int num_points)
{
    for (unsigned int i = 0; i < num_points; i++) {
        out[i] = 1.f / in[i];
    }
}
#endif /* LV_HAVE_GENERIC */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>
static inline void
volk_32f_reciprocal_32f_a_avx2_fma(float* out, const float* in, unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    for (unsigned int number = 0; number < eighth_points; number++) {
        __m256 x = _mm256_load_ps(in);
        in += 8;

        __m256 r = _mm256_reciprocal_1_avx2_fma_ps(x);

        _mm256_store_ps(out, r);
        out += 8;
    }

    const unsigned int done = eighth_points * 8;
    volk_32f_reciprocal_32f_generic(out, in, num_points - done);
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */


#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>
static inline void
volk_32f_reciprocal_32f_a_avx2_fma_chk(float* out, const float* in, unsigned int num_points)
{
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256i zero = _mm256_set1_epi32(0x00000000);
    const __m256 infinity = _mm256_set1_ps(INFINITY);

    const unsigned int eighth_points = num_points / 8;
    for (unsigned int number = 0; number < eighth_points; number++) {
        __m256 x = _mm256_load_ps(in);
        in += 8;

        const __m256 x_abs = _mm256_and_ps(x, abs_mask);
        const __m256 is_zero = _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_castps_si256(x_abs), zero));
        const __m256 is_inf = _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_castps_si256(x_abs), _mm256_castps_si256(infinity)));        
        const __m256 x_sign = _mm256_and_ps(x, sign_mask);
        __m256 r = _mm256_reciprocal_1_avx2_fma_ps(x);     
        r = _mm256_blendv_ps(r, _mm256_or_ps(x_sign, infinity), is_zero);
        r = _mm256_blendv_ps(r, _mm256_or_ps(x_sign, _mm256_castsi256_ps(zero)), is_inf);

        _mm256_store_ps(out, r);
        out += 8;
    }

    const unsigned int done = eighth_points * 8;
    volk_32f_reciprocal_32f_generic(out, in, num_points - done);
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */


#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>
static inline void
volk_32f_reciprocal_32f_a_avx(float* out, const float* in, unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    for (unsigned int number = 0; number < eighth_points; number++) {
        __m256 x = _mm256_load_ps(in);
        in += 8;

        __m256 r = _mm256_reciprocal_1_avx_ps(x);

        _mm256_store_ps(out, r);
        out += 8;
    }

    const unsigned int done = eighth_points * 8;
    volk_32f_reciprocal_32f_generic(out, in, num_points - done);
}
#endif /* LV_HAVE_AVX */

#endif /* INCLUDED_volk_32f_reciprocal_32f_a_H */

#ifndef INCLUDED_volk_32f_reciprocal_32f_u_H
#define INCLUDED_volk_32f_reciprocal_32f_u_H

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>
static inline void
volk_32f_reciprocal_32f_u_rcp(float* out, const float* in, unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    for (unsigned int number = 0; number < eighth_points; number++) {
        __m256 x = _mm256_loadu_ps(in);
        in += 8;

        __m256 r = _mm256_rcp_ps(x);

        _mm256_storeu_ps(out, r);
        out += 8;
    }

    const unsigned int done = eighth_points * 8;
    volk_32f_reciprocal_32f_generic(out, in, num_points - done);
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>
static inline void
volk_32f_reciprocal_32f_u_div(float* out, const float* in, unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    for (unsigned int number = 0; number < eighth_points; number++) {
        __m256 x = _mm256_loadu_ps(in);
        in += 8;

        __m256 r = _mm256_div_ps(_mm256_set1_ps(1.f), x);

        _mm256_storeu_ps(out, r);
        out += 8;
    }

    const unsigned int done = eighth_points * 8;
    volk_32f_reciprocal_32f_generic(out, in, num_points - done);
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

/*
        __m256 swap_mask = _mm256_cmp_ps(_mm256_and_ps(x, abs_mask), one, _CMP_GT_OS);
        __m256 x_star = _mm256_div_ps(_mm256_blendv_ps(x, one, swap_mask),
                                      _mm256_blendv_ps(one, x, swap_mask));
        __m256 result = _m256_arctan_poly_avx(x_star);
        __m256 term = _mm256_and_ps(x_star, sign_mask);
        term = _mm256_or_ps(pi_over_2, term);
        term = _mm256_sub_ps(term, result);
        result = _mm256_blendv_ps(result, term, swap_mask);
*/

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>
#include <stdio.h>
static inline void
volk_32f_reciprocal_32f_u_avx2_fma_chk(float* out, const float* in, unsigned int num_points)
{

    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    //const __m256 infi_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256i ZERO = (_mm256_set1_epi32(0x00000000));
    const __m256 INFI = _mm256_set1_ps(INFINITY);


    const unsigned int eighth_points = num_points / 8;
    for (unsigned int number = 0; number < eighth_points; number++) {
        __m256 x = _mm256_loadu_ps(in);
        in += 8;

        __m256 x_abs = _mm256_and_ps(x, abs_mask);

        /*
        float f[8];
        int32_t u[8];
        _mm256_storeu_ps(f, x_abs);
        printf("x_abs: %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f \n", f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
*/

        __m256 is_zero = _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_castps_si256(x_abs), ZERO));
        __m256 is_inf = _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_castps_si256(x_abs), _mm256_castps_si256(INFI)));
        __m256 x_sign = _mm256_and_ps(x, sign_mask);
  //      _mm256_storeu_si256((__m256i*)u, _mm256_castps_si256(is_zero));
    //    printf("is_zero: %d %d %d %d %d %d %d %d \n", u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7]);        

        //__m256 is_negative = _mm256_and_ps(x, sign_mask);




        __m256 r = _mm256_reciprocal_1_avx2_fma_ps(x);
      //  _mm256_storeu_ps(f, r);
       // printf("r   : %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f \n", f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);        
        r = _mm256_blendv_ps(r, _mm256_or_ps(x_sign, INFI), is_zero);
        r = _mm256_blendv_ps(r, _mm256_or_ps(x_sign, _mm256_castsi256_ps(ZERO)), is_inf);


        _mm256_storeu_ps(out, r);
        out += 8;
    }

    const unsigned int done = eighth_points * 8;
    volk_32f_reciprocal_32f_generic(out, in, num_points - done);
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>
static inline void
volk_32f_reciprocal_32f_u_avx2_fma(float* out, const float* in, unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    for (unsigned int number = 0; number < eighth_points; number++) {
        __m256 x = _mm256_loadu_ps(in);
        in += 8;

        __m256 r = _mm256_reciprocal_1_avx2_fma_ps(x);

        _mm256_storeu_ps(out, r);
        out += 8;
    }

    const unsigned int done = eighth_points * 8;
    volk_32f_reciprocal_32f_generic(out, in, num_points - done);
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>
static inline void
volk_32f_reciprocal_32f_u_avx(float* out, const float* in, unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    for (unsigned int number = 0; number < eighth_points; number++) {
        __m256 x = _mm256_loadu_ps(in);
        in += 8;

        __m256 r = _mm256_reciprocal_1_avx_ps(x);

        _mm256_storeu_ps(out, r);
        out += 8;
    }

    const unsigned int done = eighth_points * 8;
    volk_32f_reciprocal_32f_generic(out, in, num_points - done);
}
#endif /* LV_HAVE_AVX */

#endif /* INCLUDED_volk_32f_reciprocal_32f_u_H */
