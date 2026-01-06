/* -*- c++ -*- */
/*
 * Copyright 2025 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_log1p_32f
 *
 * \b Overview
 *
 * Computes ln(1 + x) for each element in the input vector.
 *
 * This function provides improved numerical accuracy compared to computing
 * log(1 + x) directly, especially for small values of x where catastrophic
 * cancellation would otherwise occur. For |x| << 1, log1p(x) ≈ x with full
 * precision, whereas log(1 + x) loses significant precision.
 *
 * The implementation uses range reduction combined with polynomial approximation.
 * For small x, it uses the identity log1p(x) = x * g(x) where g(x) = log1p(x)/x.
 * For larger x, it decomposes 1+x = 2^k * m and computes k*ln(2) + log(m).
 *
 * Note: This implementation does not fully conform to IEEE 754 for special values.
 * NaN inputs produce NaN outputs. Inputs <= -1 produce NaN.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_log1p_32f(float* bVector, const float* aVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: The input vector of floats x, where x > -1.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li bVector: The output vector containing ln(1 + x).
 *
 * \b Example
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   // Test with small values where log1p shines
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       in[ii] = 1e-7f * (float)(ii + 1);
 *   }
 *
 *   volk_32f_log1p_32f(out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("log1p(%e) = %e\n", in[ii], out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_log1p_32f_H
#define INCLUDED_volk_32f_log1p_32f_H

#include <inttypes.h>
#include <math.h>

#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_log1p_32f_generic(float* bVector, const float* aVector, unsigned int num_points)
{
    for (unsigned int i = 0; i < num_points; i++) {
        bVector[i] = log1pf(aVector[i]);
    }
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_32f_log1p_32f_u_sse4_1(float* bVector, const float* aVector, unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;

    /* Polynomial coefficients for g(u) = log(1+u)/u, u in [-0.293, 0.414] */
    const __m128 c0 = _mm_set1_ps(+0x1.00001p+0f);
    const __m128 c1 = _mm_set1_ps(-0x1.000176p-1f);
    const __m128 c2 = _mm_set1_ps(+0x1.552444p-2f);
    const __m128 c3 = _mm_set1_ps(-0x1.fe269p-3f);
    const __m128 c4 = _mm_set1_ps(+0x1.a3bcc2p-3f);
    const __m128 c5 = _mm_set1_ps(-0x1.7e5224p-3f);
    const __m128 c6 = _mm_set1_ps(+0x1.e773a6p-4f);

    const __m128 one = _mm_set1_ps(1.0f);
    const __m128 ln2 = _mm_set1_ps(+0x1.62e43p-1f);
    const __m128 neg_one = _mm_set1_ps(-1.0f);
    const __m128 nan_val = _mm_set1_ps(NAN);
    const __m128 sqrt2 = _mm_set1_ps(1.41421356f);
    const __m128 sqrt2_half = _mm_set1_ps(0.70710678f);

    const __m128i exp_mask = _mm_set1_epi32(0x7f800000);
    const __m128i mant_mask = _mm_set1_epi32(0x007fffff);
    const __m128i bias = _mm_set1_epi32(127);
    const __m128i one_bits = _mm_set1_epi32(0x3f800000);

    for (unsigned int i = 0; i < quarterPoints; i++) {
        __m128 x = _mm_loadu_ps(aVector);

        /* Check for invalid inputs: x <= -1 or NaN */
        __m128 invalid_mask = _mm_cmple_ps(x, neg_one);
        invalid_mask = _mm_or_ps(invalid_mask, _mm_cmpunord_ps(x, x));

        /* Compute 1 + x */
        __m128 one_plus_x = _mm_add_ps(one, x);

        /* For tiny x where 1+x rounds to 1, use log1p(x) ≈ x */
        __m128 tiny_mask = _mm_cmpeq_ps(one_plus_x, one);

        /* Extract exponent: k = floor(log2(1+x)) */
        __m128i xi = _mm_castps_si128(one_plus_x);
        __m128i exp_bits = _mm_and_si128(xi, exp_mask);
        __m128i k_i = _mm_sub_epi32(_mm_srli_epi32(exp_bits, 23), bias);
        __m128 k = _mm_cvtepi32_ps(k_i);

        /* Extract mantissa: m = (1+x) / 2^k, m in [1, 2) */
        __m128i mant_bits = _mm_and_si128(xi, mant_mask);
        __m128 m = _mm_castsi128_ps(_mm_or_si128(mant_bits, one_bits));

        /* Normalize to [sqrt(2)/2, sqrt(2)] */
        __m128 m_big = _mm_cmpge_ps(m, sqrt2);
        k = _mm_add_ps(k, _mm_and_ps(m_big, one));
        m = _mm_blendv_ps(m, _mm_mul_ps(m, _mm_set1_ps(0.5f)), m_big);

        /* u = m - 1, u in [-0.293, 0.414] */
        __m128 u = _mm_sub_ps(m, one);

        /* Evaluate polynomial: log(m) = u * (c0 + u*(c1 + u*(c2 + ...))) */
        __m128 p = c6;
        p = _mm_add_ps(_mm_mul_ps(p, u), c5);
        p = _mm_add_ps(_mm_mul_ps(p, u), c4);
        p = _mm_add_ps(_mm_mul_ps(p, u), c3);
        p = _mm_add_ps(_mm_mul_ps(p, u), c2);
        p = _mm_add_ps(_mm_mul_ps(p, u), c1);
        p = _mm_add_ps(_mm_mul_ps(p, u), c0);

        /* log(1+x) = k*ln(2) + u*P(u) */
        __m128 result = _mm_add_ps(_mm_mul_ps(k, ln2), _mm_mul_ps(u, p));

        /* For tiny x, use x directly */
        result = _mm_blendv_ps(result, x, tiny_mask);

        /* Replace invalid results with NaN */
        result = _mm_blendv_ps(result, nan_val, invalid_mask);

        _mm_storeu_ps(bVector, result);

        aVector += 4;
        bVector += 4;
    }

    /* Handle remaining elements */
    for (unsigned int i = 0; i < num_points - quarterPoints * 4; i++) {
        bVector[i] = log1pf(aVector[i]);
    }
}

static inline void
volk_32f_log1p_32f_a_sse4_1(float* bVector, const float* aVector, unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;

    /* Polynomial coefficients for g(u) = log(1+u)/u, u in [-0.293, 0.414] */
    const __m128 c0 = _mm_set1_ps(+0x1.00001p+0f);
    const __m128 c1 = _mm_set1_ps(-0x1.000176p-1f);
    const __m128 c2 = _mm_set1_ps(+0x1.552444p-2f);
    const __m128 c3 = _mm_set1_ps(-0x1.fe269p-3f);
    const __m128 c4 = _mm_set1_ps(+0x1.a3bcc2p-3f);
    const __m128 c5 = _mm_set1_ps(-0x1.7e5224p-3f);
    const __m128 c6 = _mm_set1_ps(+0x1.e773a6p-4f);

    const __m128 one = _mm_set1_ps(1.0f);
    const __m128 ln2 = _mm_set1_ps(+0x1.62e43p-1f);
    const __m128 neg_one = _mm_set1_ps(-1.0f);
    const __m128 nan_val = _mm_set1_ps(NAN);
    const __m128 sqrt2 = _mm_set1_ps(1.41421356f);

    const __m128i exp_mask = _mm_set1_epi32(0x7f800000);
    const __m128i mant_mask = _mm_set1_epi32(0x007fffff);
    const __m128i bias = _mm_set1_epi32(127);
    const __m128i one_bits = _mm_set1_epi32(0x3f800000);

    for (unsigned int i = 0; i < quarterPoints; i++) {
        __m128 x = _mm_load_ps(aVector);

        __m128 invalid_mask = _mm_cmple_ps(x, neg_one);
        invalid_mask = _mm_or_ps(invalid_mask, _mm_cmpunord_ps(x, x));

        __m128 one_plus_x = _mm_add_ps(one, x);

        /* For tiny x where 1+x rounds to 1, use log1p(x) ≈ x */
        __m128 tiny_mask = _mm_cmpeq_ps(one_plus_x, one);

        __m128i xi = _mm_castps_si128(one_plus_x);
        __m128i exp_bits = _mm_and_si128(xi, exp_mask);
        __m128i k_i = _mm_sub_epi32(_mm_srli_epi32(exp_bits, 23), bias);
        __m128 k = _mm_cvtepi32_ps(k_i);

        __m128i mant_bits = _mm_and_si128(xi, mant_mask);
        __m128 m = _mm_castsi128_ps(_mm_or_si128(mant_bits, one_bits));

        __m128 m_big = _mm_cmpge_ps(m, sqrt2);
        k = _mm_add_ps(k, _mm_and_ps(m_big, one));
        m = _mm_blendv_ps(m, _mm_mul_ps(m, _mm_set1_ps(0.5f)), m_big);

        __m128 u = _mm_sub_ps(m, one);

        __m128 p = c6;
        p = _mm_add_ps(_mm_mul_ps(p, u), c5);
        p = _mm_add_ps(_mm_mul_ps(p, u), c4);
        p = _mm_add_ps(_mm_mul_ps(p, u), c3);
        p = _mm_add_ps(_mm_mul_ps(p, u), c2);
        p = _mm_add_ps(_mm_mul_ps(p, u), c1);
        p = _mm_add_ps(_mm_mul_ps(p, u), c0);

        __m128 result = _mm_add_ps(_mm_mul_ps(k, ln2), _mm_mul_ps(u, p));
        result = _mm_blendv_ps(result, x, tiny_mask);
        result = _mm_blendv_ps(result, nan_val, invalid_mask);

        _mm_store_ps(bVector, result);

        aVector += 4;
        bVector += 4;
    }

    for (unsigned int i = 0; i < num_points - quarterPoints * 4; i++) {
        bVector[i] = log1pf(aVector[i]);
    }
}

#endif /* LV_HAVE_SSE4_1 */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_32f_log1p_32f_u_avx2(float* bVector, const float* aVector, unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    /* Polynomial coefficients for g(u) = log(1+u)/u, u in [-0.293, 0.414] */
    const __m256 c0 = _mm256_set1_ps(+0x1.00001p+0f);
    const __m256 c1 = _mm256_set1_ps(-0x1.000176p-1f);
    const __m256 c2 = _mm256_set1_ps(+0x1.552444p-2f);
    const __m256 c3 = _mm256_set1_ps(-0x1.fe269p-3f);
    const __m256 c4 = _mm256_set1_ps(+0x1.a3bcc2p-3f);
    const __m256 c5 = _mm256_set1_ps(-0x1.7e5224p-3f);
    const __m256 c6 = _mm256_set1_ps(+0x1.e773a6p-4f);

    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 ln2 = _mm256_set1_ps(+0x1.62e43p-1f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);
    const __m256 nan_val = _mm256_set1_ps(NAN);
    const __m256 sqrt2 = _mm256_set1_ps(1.41421356f);

    const __m256i exp_mask = _mm256_set1_epi32(0x7f800000);
    const __m256i mant_mask = _mm256_set1_epi32(0x007fffff);
    const __m256i bias = _mm256_set1_epi32(127);
    const __m256i one_bits = _mm256_set1_epi32(0x3f800000);

    for (unsigned int i = 0; i < eighthPoints; i++) {
        __m256 x = _mm256_loadu_ps(aVector);

        __m256 invalid_mask = _mm256_cmp_ps(x, neg_one, _CMP_LE_OQ);
        invalid_mask = _mm256_or_ps(invalid_mask, _mm256_cmp_ps(x, x, _CMP_UNORD_Q));

        __m256 one_plus_x = _mm256_add_ps(one, x);

        /* For tiny x where 1+x rounds to 1, use log1p(x) ≈ x */
        __m256 tiny_mask = _mm256_cmp_ps(one_plus_x, one, _CMP_EQ_OQ);

        __m256i xi = _mm256_castps_si256(one_plus_x);
        __m256i exp_bits = _mm256_and_si256(xi, exp_mask);
        __m256i k_i = _mm256_sub_epi32(_mm256_srli_epi32(exp_bits, 23), bias);
        __m256 k = _mm256_cvtepi32_ps(k_i);

        __m256i mant_bits = _mm256_and_si256(xi, mant_mask);
        __m256 m = _mm256_castsi256_ps(_mm256_or_si256(mant_bits, one_bits));

        __m256 m_big = _mm256_cmp_ps(m, sqrt2, _CMP_GE_OQ);
        k = _mm256_add_ps(k, _mm256_and_ps(m_big, one));
        m = _mm256_blendv_ps(m, _mm256_mul_ps(m, half), m_big);

        __m256 u = _mm256_sub_ps(m, one);

        __m256 p = c6;
        p = _mm256_add_ps(_mm256_mul_ps(p, u), c5);
        p = _mm256_add_ps(_mm256_mul_ps(p, u), c4);
        p = _mm256_add_ps(_mm256_mul_ps(p, u), c3);
        p = _mm256_add_ps(_mm256_mul_ps(p, u), c2);
        p = _mm256_add_ps(_mm256_mul_ps(p, u), c1);
        p = _mm256_add_ps(_mm256_mul_ps(p, u), c0);

        __m256 result = _mm256_add_ps(_mm256_mul_ps(k, ln2), _mm256_mul_ps(u, p));
        result = _mm256_blendv_ps(result, x, tiny_mask);
        result = _mm256_blendv_ps(result, nan_val, invalid_mask);

        _mm256_storeu_ps(bVector, result);

        aVector += 8;
        bVector += 8;
    }

    for (unsigned int i = 0; i < num_points - eighthPoints * 8; i++) {
        bVector[i] = log1pf(aVector[i]);
    }
}

static inline void
volk_32f_log1p_32f_a_avx2(float* bVector, const float* aVector, unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    /* Polynomial coefficients for g(u) = log(1+u)/u, u in [-0.293, 0.414] */
    const __m256 c0 = _mm256_set1_ps(+0x1.00001p+0f);
    const __m256 c1 = _mm256_set1_ps(-0x1.000176p-1f);
    const __m256 c2 = _mm256_set1_ps(+0x1.552444p-2f);
    const __m256 c3 = _mm256_set1_ps(-0x1.fe269p-3f);
    const __m256 c4 = _mm256_set1_ps(+0x1.a3bcc2p-3f);
    const __m256 c5 = _mm256_set1_ps(-0x1.7e5224p-3f);
    const __m256 c6 = _mm256_set1_ps(+0x1.e773a6p-4f);

    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 ln2 = _mm256_set1_ps(+0x1.62e43p-1f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);
    const __m256 nan_val = _mm256_set1_ps(NAN);
    const __m256 sqrt2 = _mm256_set1_ps(1.41421356f);

    const __m256i exp_mask = _mm256_set1_epi32(0x7f800000);
    const __m256i mant_mask = _mm256_set1_epi32(0x007fffff);
    const __m256i bias = _mm256_set1_epi32(127);
    const __m256i one_bits = _mm256_set1_epi32(0x3f800000);

    for (unsigned int i = 0; i < eighthPoints; i++) {
        __m256 x = _mm256_load_ps(aVector);

        __m256 invalid_mask = _mm256_cmp_ps(x, neg_one, _CMP_LE_OQ);
        invalid_mask = _mm256_or_ps(invalid_mask, _mm256_cmp_ps(x, x, _CMP_UNORD_Q));

        __m256 one_plus_x = _mm256_add_ps(one, x);

        /* For tiny x where 1+x rounds to 1, use log1p(x) ≈ x */
        __m256 tiny_mask = _mm256_cmp_ps(one_plus_x, one, _CMP_EQ_OQ);

        __m256i xi = _mm256_castps_si256(one_plus_x);
        __m256i exp_bits = _mm256_and_si256(xi, exp_mask);
        __m256i k_i = _mm256_sub_epi32(_mm256_srli_epi32(exp_bits, 23), bias);
        __m256 k = _mm256_cvtepi32_ps(k_i);

        __m256i mant_bits = _mm256_and_si256(xi, mant_mask);
        __m256 m = _mm256_castsi256_ps(_mm256_or_si256(mant_bits, one_bits));

        __m256 m_big = _mm256_cmp_ps(m, sqrt2, _CMP_GE_OQ);
        k = _mm256_add_ps(k, _mm256_and_ps(m_big, one));
        m = _mm256_blendv_ps(m, _mm256_mul_ps(m, half), m_big);

        __m256 u = _mm256_sub_ps(m, one);

        __m256 p = c6;
        p = _mm256_add_ps(_mm256_mul_ps(p, u), c5);
        p = _mm256_add_ps(_mm256_mul_ps(p, u), c4);
        p = _mm256_add_ps(_mm256_mul_ps(p, u), c3);
        p = _mm256_add_ps(_mm256_mul_ps(p, u), c2);
        p = _mm256_add_ps(_mm256_mul_ps(p, u), c1);
        p = _mm256_add_ps(_mm256_mul_ps(p, u), c0);

        __m256 result = _mm256_add_ps(_mm256_mul_ps(k, ln2), _mm256_mul_ps(u, p));
        result = _mm256_blendv_ps(result, x, tiny_mask);
        result = _mm256_blendv_ps(result, nan_val, invalid_mask);

        _mm256_store_ps(bVector, result);

        aVector += 8;
        bVector += 8;
    }

    for (unsigned int i = 0; i < num_points - eighthPoints * 8; i++) {
        bVector[i] = log1pf(aVector[i]);
    }
}

#endif /* LV_HAVE_AVX2 */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_log1p_32f_u_avx2_fma(float* bVector,
                                                 const float* aVector,
                                                 unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    /* Polynomial coefficients for g(u) = log(1+u)/u, u in [-0.293, 0.414] */
    const __m256 c0 = _mm256_set1_ps(+0x1.00001p+0f);
    const __m256 c1 = _mm256_set1_ps(-0x1.000176p-1f);
    const __m256 c2 = _mm256_set1_ps(+0x1.552444p-2f);
    const __m256 c3 = _mm256_set1_ps(-0x1.fe269p-3f);
    const __m256 c4 = _mm256_set1_ps(+0x1.a3bcc2p-3f);
    const __m256 c5 = _mm256_set1_ps(-0x1.7e5224p-3f);
    const __m256 c6 = _mm256_set1_ps(+0x1.e773a6p-4f);

    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 ln2 = _mm256_set1_ps(+0x1.62e43p-1f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);
    const __m256 nan_val = _mm256_set1_ps(NAN);
    const __m256 sqrt2 = _mm256_set1_ps(1.41421356f);

    const __m256i exp_mask = _mm256_set1_epi32(0x7f800000);
    const __m256i mant_mask = _mm256_set1_epi32(0x007fffff);
    const __m256i bias = _mm256_set1_epi32(127);
    const __m256i one_bits = _mm256_set1_epi32(0x3f800000);

    for (unsigned int i = 0; i < eighthPoints; i++) {
        __m256 x = _mm256_loadu_ps(aVector);

        __m256 invalid_mask = _mm256_cmp_ps(x, neg_one, _CMP_LE_OQ);
        invalid_mask = _mm256_or_ps(invalid_mask, _mm256_cmp_ps(x, x, _CMP_UNORD_Q));

        __m256 one_plus_x = _mm256_add_ps(one, x);

        /* For tiny x where 1+x rounds to 1, use log1p(x) ≈ x */
        __m256 tiny_mask = _mm256_cmp_ps(one_plus_x, one, _CMP_EQ_OQ);

        __m256i xi = _mm256_castps_si256(one_plus_x);
        __m256i exp_bits = _mm256_and_si256(xi, exp_mask);
        __m256i k_i = _mm256_sub_epi32(_mm256_srli_epi32(exp_bits, 23), bias);
        __m256 k = _mm256_cvtepi32_ps(k_i);

        __m256i mant_bits = _mm256_and_si256(xi, mant_mask);
        __m256 m = _mm256_castsi256_ps(_mm256_or_si256(mant_bits, one_bits));

        __m256 m_big = _mm256_cmp_ps(m, sqrt2, _CMP_GE_OQ);
        k = _mm256_add_ps(k, _mm256_and_ps(m_big, one));
        m = _mm256_blendv_ps(m, _mm256_mul_ps(m, half), m_big);

        __m256 u = _mm256_sub_ps(m, one);

        __m256 p = _mm256_fmadd_ps(c6, u, c5);
        p = _mm256_fmadd_ps(p, u, c4);
        p = _mm256_fmadd_ps(p, u, c3);
        p = _mm256_fmadd_ps(p, u, c2);
        p = _mm256_fmadd_ps(p, u, c1);
        p = _mm256_fmadd_ps(p, u, c0);

        __m256 result = _mm256_fmadd_ps(k, ln2, _mm256_mul_ps(u, p));
        result = _mm256_blendv_ps(result, x, tiny_mask);
        result = _mm256_blendv_ps(result, nan_val, invalid_mask);

        _mm256_storeu_ps(bVector, result);

        aVector += 8;
        bVector += 8;
    }

    for (unsigned int i = 0; i < num_points - eighthPoints * 8; i++) {
        bVector[i] = log1pf(aVector[i]);
    }
}

static inline void volk_32f_log1p_32f_a_avx2_fma(float* bVector,
                                                 const float* aVector,
                                                 unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    /* Polynomial coefficients for g(u) = log(1+u)/u, u in [-0.293, 0.414] */
    const __m256 c0 = _mm256_set1_ps(+0x1.00001p+0f);
    const __m256 c1 = _mm256_set1_ps(-0x1.000176p-1f);
    const __m256 c2 = _mm256_set1_ps(+0x1.552444p-2f);
    const __m256 c3 = _mm256_set1_ps(-0x1.fe269p-3f);
    const __m256 c4 = _mm256_set1_ps(+0x1.a3bcc2p-3f);
    const __m256 c5 = _mm256_set1_ps(-0x1.7e5224p-3f);
    const __m256 c6 = _mm256_set1_ps(+0x1.e773a6p-4f);

    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 ln2 = _mm256_set1_ps(+0x1.62e43p-1f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);
    const __m256 nan_val = _mm256_set1_ps(NAN);
    const __m256 sqrt2 = _mm256_set1_ps(1.41421356f);

    const __m256i exp_mask = _mm256_set1_epi32(0x7f800000);
    const __m256i mant_mask = _mm256_set1_epi32(0x007fffff);
    const __m256i bias = _mm256_set1_epi32(127);
    const __m256i one_bits = _mm256_set1_epi32(0x3f800000);

    for (unsigned int i = 0; i < eighthPoints; i++) {
        __m256 x = _mm256_load_ps(aVector);

        __m256 invalid_mask = _mm256_cmp_ps(x, neg_one, _CMP_LE_OQ);
        invalid_mask = _mm256_or_ps(invalid_mask, _mm256_cmp_ps(x, x, _CMP_UNORD_Q));

        __m256 one_plus_x = _mm256_add_ps(one, x);

        /* For tiny x where 1+x rounds to 1, use log1p(x) ≈ x */
        __m256 tiny_mask = _mm256_cmp_ps(one_plus_x, one, _CMP_EQ_OQ);

        __m256i xi = _mm256_castps_si256(one_plus_x);
        __m256i exp_bits = _mm256_and_si256(xi, exp_mask);
        __m256i k_i = _mm256_sub_epi32(_mm256_srli_epi32(exp_bits, 23), bias);
        __m256 k = _mm256_cvtepi32_ps(k_i);

        __m256i mant_bits = _mm256_and_si256(xi, mant_mask);
        __m256 m = _mm256_castsi256_ps(_mm256_or_si256(mant_bits, one_bits));

        __m256 m_big = _mm256_cmp_ps(m, sqrt2, _CMP_GE_OQ);
        k = _mm256_add_ps(k, _mm256_and_ps(m_big, one));
        m = _mm256_blendv_ps(m, _mm256_mul_ps(m, half), m_big);

        __m256 u = _mm256_sub_ps(m, one);

        __m256 p = _mm256_fmadd_ps(c6, u, c5);
        p = _mm256_fmadd_ps(p, u, c4);
        p = _mm256_fmadd_ps(p, u, c3);
        p = _mm256_fmadd_ps(p, u, c2);
        p = _mm256_fmadd_ps(p, u, c1);
        p = _mm256_fmadd_ps(p, u, c0);

        __m256 result = _mm256_fmadd_ps(k, ln2, _mm256_mul_ps(u, p));
        result = _mm256_blendv_ps(result, x, tiny_mask);
        result = _mm256_blendv_ps(result, nan_val, invalid_mask);

        _mm256_store_ps(bVector, result);

        aVector += 8;
        bVector += 8;
    }

    for (unsigned int i = 0; i < num_points - eighthPoints * 8; i++) {
        bVector[i] = log1pf(aVector[i]);
    }
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void
volk_32f_log1p_32f_u_avx512(float* bVector, const float* aVector, unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;

    /* Polynomial coefficients for g(u) = log(1+u)/u, u in [-0.293, 0.414] */
    const __m512 c0 = _mm512_set1_ps(+0x1.00001p+0f);
    const __m512 c1 = _mm512_set1_ps(-0x1.000176p-1f);
    const __m512 c2 = _mm512_set1_ps(+0x1.552444p-2f);
    const __m512 c3 = _mm512_set1_ps(-0x1.fe269p-3f);
    const __m512 c4 = _mm512_set1_ps(+0x1.a3bcc2p-3f);
    const __m512 c5 = _mm512_set1_ps(-0x1.7e5224p-3f);
    const __m512 c6 = _mm512_set1_ps(+0x1.e773a6p-4f);

    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 ln2 = _mm512_set1_ps(+0x1.62e43p-1f);
    const __m512 neg_one = _mm512_set1_ps(-1.0f);
    const __m512 nan_val = _mm512_set1_ps(NAN);
    const __m512 sqrt2 = _mm512_set1_ps(1.41421356f);

    const __m512i exp_mask = _mm512_set1_epi32(0x7f800000);
    const __m512i mant_mask = _mm512_set1_epi32(0x007fffff);
    const __m512i bias = _mm512_set1_epi32(127);
    const __m512i one_bits = _mm512_set1_epi32(0x3f800000);

    for (unsigned int i = 0; i < sixteenthPoints; i++) {
        __m512 x = _mm512_loadu_ps(aVector);

        __mmask16 invalid_mask = _mm512_cmp_ps_mask(x, neg_one, _CMP_LE_OQ);
        __mmask16 nan_mask = _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q);
        invalid_mask = _kor_mask16(invalid_mask, nan_mask);

        __m512 one_plus_x = _mm512_add_ps(one, x);

        /* For tiny x where 1+x rounds to 1, use log1p(x) ≈ x */
        __mmask16 tiny_mask = _mm512_cmp_ps_mask(one_plus_x, one, _CMP_EQ_OQ);

        __m512i xi = _mm512_castps_si512(one_plus_x);
        __m512i exp_bits = _mm512_and_si512(xi, exp_mask);
        __m512i k_i = _mm512_sub_epi32(_mm512_srli_epi32(exp_bits, 23), bias);
        __m512 k = _mm512_cvtepi32_ps(k_i);

        __m512i mant_bits = _mm512_and_si512(xi, mant_mask);
        __m512 m = _mm512_castsi512_ps(_mm512_or_si512(mant_bits, one_bits));

        __mmask16 m_big = _mm512_cmp_ps_mask(m, sqrt2, _CMP_GE_OQ);
        k = _mm512_mask_add_ps(k, m_big, k, one);
        m = _mm512_mask_mul_ps(m, m_big, m, half);

        __m512 u = _mm512_sub_ps(m, one);

        __m512 p = _mm512_fmadd_ps(c6, u, c5);
        p = _mm512_fmadd_ps(p, u, c4);
        p = _mm512_fmadd_ps(p, u, c3);
        p = _mm512_fmadd_ps(p, u, c2);
        p = _mm512_fmadd_ps(p, u, c1);
        p = _mm512_fmadd_ps(p, u, c0);

        __m512 result = _mm512_fmadd_ps(k, ln2, _mm512_mul_ps(u, p));
        result = _mm512_mask_blend_ps(tiny_mask, result, x);
        result = _mm512_mask_blend_ps(invalid_mask, result, nan_val);

        _mm512_storeu_ps(bVector, result);

        aVector += 16;
        bVector += 16;
    }

    for (unsigned int i = 0; i < num_points - sixteenthPoints * 16; i++) {
        bVector[i] = log1pf(aVector[i]);
    }
}

static inline void
volk_32f_log1p_32f_a_avx512(float* bVector, const float* aVector, unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;

    /* Polynomial coefficients for g(u) = log(1+u)/u, u in [-0.293, 0.414] */
    const __m512 c0 = _mm512_set1_ps(+0x1.00001p+0f);
    const __m512 c1 = _mm512_set1_ps(-0x1.000176p-1f);
    const __m512 c2 = _mm512_set1_ps(+0x1.552444p-2f);
    const __m512 c3 = _mm512_set1_ps(-0x1.fe269p-3f);
    const __m512 c4 = _mm512_set1_ps(+0x1.a3bcc2p-3f);
    const __m512 c5 = _mm512_set1_ps(-0x1.7e5224p-3f);
    const __m512 c6 = _mm512_set1_ps(+0x1.e773a6p-4f);

    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 ln2 = _mm512_set1_ps(+0x1.62e43p-1f);
    const __m512 neg_one = _mm512_set1_ps(-1.0f);
    const __m512 nan_val = _mm512_set1_ps(NAN);
    const __m512 sqrt2 = _mm512_set1_ps(1.41421356f);

    const __m512i exp_mask = _mm512_set1_epi32(0x7f800000);
    const __m512i mant_mask = _mm512_set1_epi32(0x007fffff);
    const __m512i bias = _mm512_set1_epi32(127);
    const __m512i one_bits = _mm512_set1_epi32(0x3f800000);

    for (unsigned int i = 0; i < sixteenthPoints; i++) {
        __m512 x = _mm512_load_ps(aVector);

        __mmask16 invalid_mask = _mm512_cmp_ps_mask(x, neg_one, _CMP_LE_OQ);
        __mmask16 nan_mask = _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q);
        invalid_mask = _kor_mask16(invalid_mask, nan_mask);

        __m512 one_plus_x = _mm512_add_ps(one, x);

        /* For tiny x where 1+x rounds to 1, use log1p(x) ≈ x */
        __mmask16 tiny_mask = _mm512_cmp_ps_mask(one_plus_x, one, _CMP_EQ_OQ);

        __m512i xi = _mm512_castps_si512(one_plus_x);
        __m512i exp_bits = _mm512_and_si512(xi, exp_mask);
        __m512i k_i = _mm512_sub_epi32(_mm512_srli_epi32(exp_bits, 23), bias);
        __m512 k = _mm512_cvtepi32_ps(k_i);

        __m512i mant_bits = _mm512_and_si512(xi, mant_mask);
        __m512 m = _mm512_castsi512_ps(_mm512_or_si512(mant_bits, one_bits));

        __mmask16 m_big = _mm512_cmp_ps_mask(m, sqrt2, _CMP_GE_OQ);
        k = _mm512_mask_add_ps(k, m_big, k, one);
        m = _mm512_mask_mul_ps(m, m_big, m, half);

        __m512 u = _mm512_sub_ps(m, one);

        __m512 p = _mm512_fmadd_ps(c6, u, c5);
        p = _mm512_fmadd_ps(p, u, c4);
        p = _mm512_fmadd_ps(p, u, c3);
        p = _mm512_fmadd_ps(p, u, c2);
        p = _mm512_fmadd_ps(p, u, c1);
        p = _mm512_fmadd_ps(p, u, c0);

        __m512 result = _mm512_fmadd_ps(k, ln2, _mm512_mul_ps(u, p));
        result = _mm512_mask_blend_ps(tiny_mask, result, x);
        result = _mm512_mask_blend_ps(invalid_mask, result, nan_val);

        _mm512_store_ps(bVector, result);

        aVector += 16;
        bVector += 16;
    }

    for (unsigned int i = 0; i < num_points - sixteenthPoints * 16; i++) {
        bVector[i] = log1pf(aVector[i]);
    }
}

#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void
volk_32f_log1p_32f_neonv8(float* bVector, const float* aVector, unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;

    /* Polynomial coefficients for g(u) = log(1+u)/u, u in [-0.293, 0.414] */
    const float32x4_t c0 = vdupq_n_f32(+0x1.00001p+0f);
    const float32x4_t c1 = vdupq_n_f32(-0x1.000176p-1f);
    const float32x4_t c2 = vdupq_n_f32(+0x1.552444p-2f);
    const float32x4_t c3 = vdupq_n_f32(-0x1.fe269p-3f);
    const float32x4_t c4 = vdupq_n_f32(+0x1.a3bcc2p-3f);
    const float32x4_t c5 = vdupq_n_f32(-0x1.7e5224p-3f);
    const float32x4_t c6 = vdupq_n_f32(+0x1.e773a6p-4f);

    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t ln2 = vdupq_n_f32(+0x1.62e43p-1f);
    const float32x4_t neg_one = vdupq_n_f32(-1.0f);
    const float32x4_t nan_val = vdupq_n_f32(NAN);
    const float32x4_t sqrt2 = vdupq_n_f32(1.41421356f);

    const int32x4_t exp_mask = vdupq_n_s32(0x7f800000);
    const int32x4_t mant_mask = vdupq_n_s32(0x007fffff);
    const int32x4_t bias = vdupq_n_s32(127);
    const int32x4_t one_bits = vdupq_n_s32(0x3f800000);

    for (unsigned int i = 0; i < quarterPoints; i++) {
        float32x4_t x = vld1q_f32(aVector);

        /* Check for invalid inputs */
        uint32x4_t invalid_mask = vcleq_f32(x, neg_one);
        uint32x4_t nan_mask = vmvnq_u32(vceqq_f32(x, x));
        invalid_mask = vorrq_u32(invalid_mask, nan_mask);

        float32x4_t one_plus_x = vaddq_f32(one, x);

        /* For tiny x where 1+x rounds to 1, use log1p(x) ≈ x */
        uint32x4_t tiny_mask = vceqq_f32(one_plus_x, one);

        int32x4_t xi = vreinterpretq_s32_f32(one_plus_x);
        int32x4_t exp_bits = vandq_s32(xi, exp_mask);
        int32x4_t k_i = vsubq_s32(vshrq_n_s32(exp_bits, 23), bias);
        float32x4_t k = vcvtq_f32_s32(k_i);

        int32x4_t mant_bits = vandq_s32(xi, mant_mask);
        float32x4_t m = vreinterpretq_f32_s32(vorrq_s32(mant_bits, one_bits));

        uint32x4_t m_big = vcgeq_f32(m, sqrt2);
        k = vbslq_f32(m_big, vaddq_f32(k, one), k);
        m = vbslq_f32(m_big, vmulq_f32(m, half), m);

        float32x4_t u = vsubq_f32(m, one);

        float32x4_t p = vfmaq_f32(c5, c6, u);
        p = vfmaq_f32(c4, p, u);
        p = vfmaq_f32(c3, p, u);
        p = vfmaq_f32(c2, p, u);
        p = vfmaq_f32(c1, p, u);
        p = vfmaq_f32(c0, p, u);

        float32x4_t result = vfmaq_f32(vmulq_f32(u, p), k, ln2);
        result = vbslq_f32(tiny_mask, x, result);
        result = vbslq_f32(invalid_mask, nan_val, result);

        vst1q_f32(bVector, result);

        aVector += 4;
        bVector += 4;
    }

    for (unsigned int i = 0; i < num_points - quarterPoints * 4; i++) {
        bVector[i] = log1pf(aVector[i]);
    }
}

#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32f_log1p_32f_neon(float* bVector, const float* aVector, unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;

    /* Polynomial coefficients for g(u) = log(1+u)/u, u in [-0.293, 0.414] */
    const float32x4_t c0 = vdupq_n_f32(+0x1.00001p+0f);
    const float32x4_t c1 = vdupq_n_f32(-0x1.000176p-1f);
    const float32x4_t c2 = vdupq_n_f32(+0x1.552444p-2f);
    const float32x4_t c3 = vdupq_n_f32(-0x1.fe269p-3f);
    const float32x4_t c4 = vdupq_n_f32(+0x1.a3bcc2p-3f);
    const float32x4_t c5 = vdupq_n_f32(-0x1.7e5224p-3f);
    const float32x4_t c6 = vdupq_n_f32(+0x1.e773a6p-4f);

    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t ln2 = vdupq_n_f32(+0x1.62e43p-1f);
    const float32x4_t neg_one = vdupq_n_f32(-1.0f);
    const float32x4_t nan_val = vdupq_n_f32(NAN);
    const float32x4_t sqrt2 = vdupq_n_f32(1.41421356f);

    const int32x4_t exp_mask = vdupq_n_s32(0x7f800000);
    const int32x4_t mant_mask = vdupq_n_s32(0x007fffff);
    const int32x4_t bias = vdupq_n_s32(127);
    const int32x4_t one_bits = vdupq_n_s32(0x3f800000);

    for (unsigned int i = 0; i < quarterPoints; i++) {
        float32x4_t x = vld1q_f32(aVector);

        uint32x4_t invalid_mask = vcleq_f32(x, neg_one);
        uint32x4_t nan_mask = vmvnq_u32(vceqq_f32(x, x));
        invalid_mask = vorrq_u32(invalid_mask, nan_mask);

        float32x4_t one_plus_x = vaddq_f32(one, x);

        /* For tiny x where 1+x rounds to 1, use log1p(x) ≈ x */
        uint32x4_t tiny_mask = vceqq_f32(one_plus_x, one);

        int32x4_t xi = vreinterpretq_s32_f32(one_plus_x);
        int32x4_t exp_bits = vandq_s32(xi, exp_mask);
        int32x4_t k_i = vsubq_s32(vshrq_n_s32(exp_bits, 23), bias);
        float32x4_t k = vcvtq_f32_s32(k_i);

        int32x4_t mant_bits = vandq_s32(xi, mant_mask);
        float32x4_t m = vreinterpretq_f32_s32(vorrq_s32(mant_bits, one_bits));

        uint32x4_t m_big = vcgeq_f32(m, sqrt2);
        k = vbslq_f32(m_big, vaddq_f32(k, one), k);
        m = vbslq_f32(m_big, vmulq_f32(m, half), m);

        float32x4_t u = vsubq_f32(m, one);

        /* Horner's method without FMA */
        float32x4_t p = c6;
        p = vaddq_f32(vmulq_f32(p, u), c5);
        p = vaddq_f32(vmulq_f32(p, u), c4);
        p = vaddq_f32(vmulq_f32(p, u), c3);
        p = vaddq_f32(vmulq_f32(p, u), c2);
        p = vaddq_f32(vmulq_f32(p, u), c1);
        p = vaddq_f32(vmulq_f32(p, u), c0);

        float32x4_t result = vaddq_f32(vmulq_f32(k, ln2), vmulq_f32(u, p));
        result = vbslq_f32(tiny_mask, x, result);
        result = vbslq_f32(invalid_mask, nan_val, result);

        vst1q_f32(bVector, result);

        aVector += 4;
        bVector += 4;
    }

    for (unsigned int i = 0; i < num_points - quarterPoints * 4; i++) {
        bVector[i] = log1pf(aVector[i]);
    }
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void
volk_32f_log1p_32f_rvv(float* bVector, const float* aVector, unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m2();

    /* Polynomial coefficients for g(u) = log(1+u)/u, u in [-0.293, 0.414] */
    const vfloat32m2_t c0 = __riscv_vfmv_v_f_f32m2(+0x1.00001p+0f, vlmax);
    const vfloat32m2_t c1 = __riscv_vfmv_v_f_f32m2(-0x1.000176p-1f, vlmax);
    const vfloat32m2_t c2 = __riscv_vfmv_v_f_f32m2(+0x1.552444p-2f, vlmax);
    const vfloat32m2_t c3 = __riscv_vfmv_v_f_f32m2(-0x1.fe269p-3f, vlmax);
    const vfloat32m2_t c4 = __riscv_vfmv_v_f_f32m2(+0x1.a3bcc2p-3f, vlmax);
    const vfloat32m2_t c5 = __riscv_vfmv_v_f_f32m2(-0x1.7e5224p-3f, vlmax);
    const vfloat32m2_t c6 = __riscv_vfmv_v_f_f32m2(+0x1.e773a6p-4f, vlmax);

    const vfloat32m2_t one = __riscv_vfmv_v_f_f32m2(1.0f, vlmax);
    const vfloat32m2_t half = __riscv_vfmv_v_f_f32m2(0.5f, vlmax);
    const vfloat32m2_t ln2 = __riscv_vfmv_v_f_f32m2(+0x1.62e43p-1f, vlmax);
    const vfloat32m2_t neg_one = __riscv_vfmv_v_f_f32m2(-1.0f, vlmax);
    const vfloat32m2_t nan_val = __riscv_vfmv_v_f_f32m2(NAN, vlmax);
    const vfloat32m2_t sqrt2 = __riscv_vfmv_v_f_f32m2(1.41421356f, vlmax);

    const vint32m2_t exp_mask = __riscv_vmv_v_x_i32m2(0x7f800000, vlmax);
    const vint32m2_t mant_mask = __riscv_vmv_v_x_i32m2(0x007fffff, vlmax);
    const vint32m2_t bias = __riscv_vmv_v_x_i32m2(127, vlmax);
    const vint32m2_t one_bits = __riscv_vmv_v_x_i32m2(0x3f800000, vlmax);

    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl) {
        vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t x = __riscv_vle32_v_f32m2(aVector, vl);

        vbool16_t invalid_mask = __riscv_vmfle(x, neg_one, vl);
        vbool16_t nan_mask = __riscv_vmfne(x, x, vl);
        invalid_mask = __riscv_vmor(invalid_mask, nan_mask, vl);

        vfloat32m2_t one_plus_x = __riscv_vfadd(x, one, vl);

        /* For tiny x where 1+x rounds to 1, use log1p(x) ≈ x */
        vbool16_t tiny_mask = __riscv_vmfeq(one_plus_x, one, vl);

        vint32m2_t xi = __riscv_vreinterpret_i32m2(one_plus_x);
        vint32m2_t exp_bits = __riscv_vand(xi, exp_mask, vl);
        vint32m2_t k_i = __riscv_vsub(__riscv_vsra(exp_bits, 23, vl), bias, vl);
        vfloat32m2_t k = __riscv_vfcvt_f(k_i, vl);

        vint32m2_t mant_bits = __riscv_vand(xi, mant_mask, vl);
        vfloat32m2_t m = __riscv_vreinterpret_f32m2(__riscv_vor(mant_bits, one_bits, vl));

        vbool16_t m_big = __riscv_vmfge(m, sqrt2, vl);
        k = __riscv_vfadd_vf_f32m2_mu(m_big, k, k, 1.0f, vl);
        m = __riscv_vfmul_vf_f32m2_mu(m_big, m, m, 0.5f, vl);

        vfloat32m2_t u = __riscv_vfsub(m, one, vl);

        vfloat32m2_t p = __riscv_vfmadd(c6, u, c5, vl);
        p = __riscv_vfmadd(p, u, c4, vl);
        p = __riscv_vfmadd(p, u, c3, vl);
        p = __riscv_vfmadd(p, u, c2, vl);
        p = __riscv_vfmadd(p, u, c1, vl);
        p = __riscv_vfmadd(p, u, c0, vl);

        vfloat32m2_t result = __riscv_vfmacc(__riscv_vfmul(u, p, vl), k, ln2, vl);
        result = __riscv_vmerge(result, x, tiny_mask, vl);
        result = __riscv_vmerge(result, nan_val, invalid_mask, vl);

        __riscv_vse32(bVector, result, vl);
    }
}

#endif /* LV_HAVE_RVV */

#endif /* INCLUDED_volk_32f_log1p_32f_H */
