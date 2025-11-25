/* -*- c++ -*- */
/*
 * Copyright 2024 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_average_power_32f
 *
 * \b Overview
 *
 * Calculates the average power of a complex vector: sum(|z|^2)/N = sum(r^2 + i^2)/N
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_average_power_32f(float* result, const lv_32fc_t* input, unsigned int
 * num_points) \endcode
 *
 * \b Inputs
 * \li input: The input vector of complex floats.
 * \li num_points: The number of complex data points.
 *
 * \b Outputs
 * \li result: The average power value.
 *
 * \b Example
 * \code
 *   int N = 10000;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* input = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   float result;
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       float real = 2.0f * ((float)rand() / (float)RAND_MAX - 0.5f);
 *       float imag = 2.0f * ((float)rand() / (float)RAND_MAX - 0.5f);
 *       input[ii] = lv_cmake(real, imag);
 *   }
 *
 *   volk_32fc_average_power_32f(&result, input, N);
 *
 *   printf("Average power: %f\n", result);
 *
 *   volk_free(input);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_average_power_32f_H
#define INCLUDED_volk_32fc_average_power_32f_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_average_power_32f_generic(float* result,
                                                       const lv_32fc_t* input,
                                                       unsigned int num_points)
{
    float sum = 0.0f;
    for (unsigned int i = 0; i < num_points; i++) {
        const float real = lv_creal(input[i]);
        const float imag = lv_cimag(input[i]);
        sum += real * real + imag * imag;
    }
    *result = sum / (float)num_points;
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32fc_average_power_32f_a_sse(float* result,
                                                     const lv_32fc_t* input,
                                                     unsigned int num_points)
{
    const unsigned int half_points = num_points / 2;
    __m128 accumulator = _mm_setzero_ps();
    const float* aPtr = (const float*)input;

    for (unsigned int number = 0; number < half_points; number++) {
        // Load 2 complex values (4 floats): r0,i0,r1,i1
        __m128 aVal = _mm_load_ps(aPtr);
        accumulator = _mm_add_ps(accumulator, _mm_mul_ps(aVal, aVal));
        aPtr += 4;
    }

    // Horizontal sum
    __m128 shuf = _mm_shuffle_ps(accumulator, accumulator, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(accumulator, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float sum;
    _mm_store_ss(&sum, sums);

    // Handle remaining elements
    for (unsigned int number = half_points * 2; number < num_points; number++) {
        const float real = lv_creal(input[number]);
        const float imag = lv_cimag(input[number]);
        sum += real * real + imag * imag;
    }

    *result = sum / (float)num_points;
}

#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32fc_average_power_32f_u_sse(float* result,
                                                     const lv_32fc_t* input,
                                                     unsigned int num_points)
{
    const unsigned int half_points = num_points / 2;
    __m128 accumulator = _mm_setzero_ps();
    const float* aPtr = (const float*)input;

    for (unsigned int number = 0; number < half_points; number++) {
        __m128 aVal = _mm_loadu_ps(aPtr);
        accumulator = _mm_add_ps(accumulator, _mm_mul_ps(aVal, aVal));
        aPtr += 4;
    }

    __m128 shuf = _mm_shuffle_ps(accumulator, accumulator, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(accumulator, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float sum;
    _mm_store_ss(&sum, sums);

    for (unsigned int number = half_points * 2; number < num_points; number++) {
        const float real = lv_creal(input[number]);
        const float imag = lv_cimag(input[number]);
        sum += real * real + imag * imag;
    }

    *result = sum / (float)num_points;
}

#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_average_power_32f_a_avx(float* result,
                                                     const lv_32fc_t* input,
                                                     unsigned int num_points)
{
    const unsigned int quarter_points = num_points / 4;
    __m256 accumulator = _mm256_setzero_ps();
    const float* aPtr = (const float*)input;

    for (unsigned int number = 0; number < quarter_points; number++) {
        // Load 4 complex values (8 floats): r0,i0,r1,i1,r2,i2,r3,i3
        __m256 aVal = _mm256_load_ps(aPtr);
        accumulator = _mm256_add_ps(accumulator, _mm256_mul_ps(aVal, aVal));
        aPtr += 8;
    }

    // Horizontal sum
    __m128 low = _mm256_castps256_ps128(accumulator);
    __m128 high = _mm256_extractf128_ps(accumulator, 1);
    low = _mm_add_ps(low, high);

    __m128 shuf = _mm_shuffle_ps(low, low, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(low, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float sum;
    _mm_store_ss(&sum, sums);

    for (unsigned int number = quarter_points * 4; number < num_points; number++) {
        const float real = lv_creal(input[number]);
        const float imag = lv_cimag(input[number]);
        sum += real * real + imag * imag;
    }

    *result = sum / (float)num_points;
}

#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_average_power_32f_u_avx(float* result,
                                                     const lv_32fc_t* input,
                                                     unsigned int num_points)
{
    const unsigned int quarter_points = num_points / 4;
    __m256 accumulator = _mm256_setzero_ps();
    const float* aPtr = (const float*)input;

    for (unsigned int number = 0; number < quarter_points; number++) {
        __m256 aVal = _mm256_loadu_ps(aPtr);
        accumulator = _mm256_add_ps(accumulator, _mm256_mul_ps(aVal, aVal));
        aPtr += 8;
    }

    __m128 low = _mm256_castps256_ps128(accumulator);
    __m128 high = _mm256_extractf128_ps(accumulator, 1);
    low = _mm_add_ps(low, high);

    __m128 shuf = _mm_shuffle_ps(low, low, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(low, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float sum;
    _mm_store_ss(&sum, sums);

    for (unsigned int number = quarter_points * 4; number < num_points; number++) {
        const float real = lv_creal(input[number]);
        const float imag = lv_cimag(input[number]);
        sum += real * real + imag * imag;
    }

    *result = sum / (float)num_points;
}

#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_average_power_32f_a_avx2(float* result,
                                                      const lv_32fc_t* input,
                                                      unsigned int num_points)
{
    volk_32fc_average_power_32f_a_avx(result, input, num_points);
}

#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_average_power_32f_u_avx2(float* result,
                                                      const lv_32fc_t* input,
                                                      unsigned int num_points)
{
    volk_32fc_average_power_32f_u_avx(result, input, num_points);
}

#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_average_power_32f_a_avx512f(float* result,
                                                         const lv_32fc_t* input,
                                                         unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    __m512 accumulator = _mm512_setzero_ps();
    const float* aPtr = (const float*)input;

    for (unsigned int number = 0; number < eighth_points; number++) {
        // Load 8 complex values (16 floats)
        __m512 aVal = _mm512_load_ps(aPtr);
        accumulator = _mm512_fmadd_ps(aVal, aVal, accumulator);
        aPtr += 16;
    }

    // Horizontal sum using AVX512 reduce
    float sum = _mm512_reduce_add_ps(accumulator);

    for (unsigned int number = eighth_points * 8; number < num_points; number++) {
        const float real = lv_creal(input[number]);
        const float imag = lv_cimag(input[number]);
        sum += real * real + imag * imag;
    }

    *result = sum / (float)num_points;
}

#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_average_power_32f_u_avx512f(float* result,
                                                         const lv_32fc_t* input,
                                                         unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    __m512 accumulator = _mm512_setzero_ps();
    const float* aPtr = (const float*)input;

    for (unsigned int number = 0; number < eighth_points; number++) {
        __m512 aVal = _mm512_loadu_ps(aPtr);
        accumulator = _mm512_fmadd_ps(aVal, aVal, accumulator);
        aPtr += 16;
    }

    float sum = _mm512_reduce_add_ps(accumulator);

    for (unsigned int number = eighth_points * 8; number < num_points; number++) {
        const float real = lv_creal(input[number]);
        const float imag = lv_cimag(input[number]);
        sum += real * real + imag * imag;
    }

    *result = sum / (float)num_points;
}

#endif /* LV_HAVE_AVX512F */

#endif /* INCLUDED_volk_32fc_average_power_32f_H */
