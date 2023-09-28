/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 * Copyright 2023 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_s32f_atan2_32f
 *
 * \b Overview
 *
 * Computes the arctan for each value in a complex vector and applies
 * a normalization factor.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32f_atan2_32f(float* outputVector, const lv_32fc_t* complexVector,
 * const float normalizeFactor, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li inputVector: The byte-aligned input vector containing interleaved IQ data (I = cos,
 * Q = sin). \li normalizeFactor: The atan results are divided by this normalization
 * factor. \li num_points: The number of complex values in \p inputVector.
 *
 * \b Outputs
 * \li outputVector: The vector where the results will be stored.
 *
 * \b Example
 * Calculate the arctangent of points around the unit circle.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float scale = 1.f; // we want unit circle
 *
 *   for(unsigned int ii = 0; ii < N/2; ++ii){
 *       // Generate points around the unit circle
 *       float real = -4.f * ((float)ii / (float)N) + 1.f;
 *       float imag = std::sqrt(1.f - real * real);
 *       in[ii] = lv_cmake(real, imag);
 *       in[ii+N/2] = lv_cmake(-real, -imag);
 *   }
 *
 *   volk_32fc_s32f_atan2_32f(out, in, scale, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("atan2(%1.2f, %1.2f) = %1.2f\n",
 *           lv_cimag(in[ii]), lv_creal(in[ii]), out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */


#ifndef INCLUDED_volk_32fc_s32f_atan2_32f_a_H
#define INCLUDED_volk_32fc_s32f_atan2_32f_a_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>
static inline void volk_32fc_s32f_atan2_32f_a_avx2_fma(float* outputVector,
                                                     const lv_32fc_t* complexVector,
                                                     const float normalizeFactor,
                                                     unsigned int num_points) {
    const float* in = (float*) complexVector;
    float* out = (float*) outputVector;
    
    const float invNormalizeFactor = 1.f / normalizeFactor;     
    const __m256 vinvNormalizeFactor = _mm256_set1_ps(invNormalizeFactor);
    const __m256 pi = _mm256_set1_ps(0x1.921fb6p1f);    // pi
    const __m256 pi_2 = _mm256_set1_ps(0x1.921fb6p0f);  // pi / 2
    const __m256i permute_mask = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));;
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));   

    unsigned int number = 0;
    unsigned int eighth_points = num_points / 8;
    for (; number < eighth_points; number++) {   
        __m256 z1 = _mm256_load_ps(in);
        in += 8;
        __m256 z2 = _mm256_load_ps(in);
        in += 8;

        __m256 x = _mm256_shuffle_ps(z1, z2, _MM_SHUFFLE(2, 0, 2, 0));
        x = _mm256_permutevar8x32_ps(x, permute_mask);
        __m256 y = _mm256_shuffle_ps(z1, z2, _MM_SHUFFLE(3, 1, 3, 1));
        y = _mm256_permutevar8x32_ps(y, permute_mask);

        __m256 swap_mask = _mm256_cmp_ps(_mm256_and_ps(y, abs_mask), _mm256_and_ps(x, abs_mask), _CMP_GT_OS);

        __m256 input = _mm256_div_ps(_mm256_blendv_ps(y, x, swap_mask), _mm256_blendv_ps(x, y, swap_mask) );

        __m256 result = _m256_arctan_approximation_avx2_fma(input);

        input = _mm256_sub_ps( _mm256_or_ps(pi_2, _mm256_and_ps(input, sign_mask)), result );
        result = _mm256_blendv_ps( result, input, swap_mask );

        __m256 x_sign_mask = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(x), 31));
        float a[8];
        printf("xsm: %f %f %f %f %f %f %f %f\n", a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]);

        result = _mm256_add_ps(_mm256_and_ps(_mm256_xor_ps(pi, _mm256_and_ps(sign_mask, y)), x_sign_mask), result);
        result = _mm256_mul_ps(result, vinvNormalizeFactor);

        _mm256_store_ps(out, result);
        out += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        const float x = *in++;
        const float y = *in++;
        *out++ = atan2f(y, x) * invNormalizeFactor;
    }    

}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for aligned */

#if LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>
static inline void volk_32fc_s32f_atan2_32f_a_avx(float* outputVector,
                                                     const lv_32fc_t* complexVector,
                                                     const float normalizeFactor,
                                                     unsigned int num_points) {
    const float* in = (float*) complexVector;
    float* out = (float*) outputVector;
    
    const float invNormalizeFactor = 1.f / normalizeFactor;     
    const __m256 vinvNormalizeFactor = _mm256_set1_ps(invNormalizeFactor);
    //const __m256 zero  = _mm256_set1_ps(0.f);    // pi
    const __m256 pi = _mm256_set1_ps(0x1.921fb6p1f);    // pi
    const __m256 pi_2 = _mm256_set1_ps(0x1.921fb6p0f);  // pi / 2
    const __m256i permute_mask = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));;
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));   

    unsigned int number = 0;
    unsigned int eighth_points = num_points / 8;
    for (; number < eighth_points; number++) {   
        __m256 z1 = _mm256_load_ps(in);
        in += 8;
        __m256 z2 = _mm256_load_ps(in);
        in += 8;

        __m256 x = _mm256_shuffle_ps(z1, z2, _MM_SHUFFLE(2, 0, 2, 0));
        x = _mm256_permutevar_ps(x, permute_mask);
        __m256 y = _mm256_shuffle_ps(z1, z2, _MM_SHUFFLE(3, 1, 3, 1));
        y = _mm256_permutevar_ps(y, permute_mask);

        __m256 swap_mask = _mm256_cmp_ps(_mm256_and_ps(y, abs_mask), _mm256_and_ps(x, abs_mask), _CMP_GT_OS);  
        __m256 input = _mm256_div_ps(_mm256_blendv_ps(y, x, swap_mask), _mm256_blendv_ps(x, y, swap_mask) );

        __m256 result = _m256_arctan_approximation_avx(input);

        input = _mm256_sub_ps( _mm256_or_ps(pi_2, _mm256_and_ps(input, sign_mask)), result );
        result = _mm256_blendv_ps( result, input, swap_mask );

        __m128 x_l = _mm256_extractf128_ps(x, 1);
        __m128 x_h = _mm256_extractf128_ps(x, 0);
        x_l = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(x_l), 31));
        x_h = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(x_h), 31));
        __m256 x_sign_mask = _mm256_set_m128(x_h, x_l);


        //__m256 x_sign_mask = _mm256_srai_epi32(_mm256_castps_si256(x), 31)
        //x_sign_mask = _mm256_castsi256_ps(x_sign_mask);

        result = _mm256_add_ps(_mm256_and_ps(_mm256_xor_ps(pi, _mm256_and_ps(sign_mask, y)), x_sign_mask), result);
        result = _mm256_mul_ps(result, vinvNormalizeFactor);

        _mm256_store_ps(out, result);
        out += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        const float x = *in++;
        const float y = *in++;
        *out++ = atan2f(y, x) * invNormalizeFactor;
    }    

}
#endif /* LV_HAVE_AVX for aligned */

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_s32f_atan2_32f_generic(float* outputVector,
                                                    const lv_32fc_t* inputVector,
                                                    const float normalizeFactor,
                                                    unsigned int num_points)
{
    float* outPtr = outputVector;
    const float* inPtr = (float*)inputVector;
    const float invNormalizeFactor = 1.0 / normalizeFactor;
    unsigned int number;
    for (number = 0; number < num_points; number++) {
        const float real = *inPtr++;
        const float imag = *inPtr++;
        *outPtr++ = atan2f(imag, real) * invNormalizeFactor;
    }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32fc_s32f_atan2_32f_a_H */
