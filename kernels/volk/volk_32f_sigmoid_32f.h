/* -*- c++ -*- */
/*
 * Copyright 2025 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_sigmoid_32f
 *
 * \b Overview
 *
 * Computes the sigmoid (logistic) function of each element:
 *
 * out[i] = 1 / (1 + exp(-in[i]))
 *
 * This kernel uses a polynomial approximation with maximum relative error < 1e-6.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_sigmoid_32f(float* outVector, const float* inVector, unsigned int
 * num_points) \endcode
 *
 * \b Inputs
 * \li inVector: The input buffer.
 * \li num_points: The number of values.
 *
 * \b Outputs
 * \li outVector: The output buffer.
 *
 * \b Example
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       in[ii] = (float)ii - 5.0f;  // Range from -5 to 4
 *   }
 *
 *   volk_32f_sigmoid_32f(out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("sigmoid(%+.1f) = %.6f\n", in[ii], out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_sigmoid_32f_a_H
#define INCLUDED_volk_32f_sigmoid_32f_a_H

#include <inttypes.h>
#include <math.h>

/*
 * Polynomial coefficients for sigmoid approximation on [0, 5]
 * Generated using Sollya: fpminimax(1/(1+exp(-x)), 11, [|single...|], [0, 5])
 * Max relative error: 6.3e-7
 *
 * For x < 0: sigmoid(x) = 1 - sigmoid(-x)
 * For x > 5: sigmoid(x) = 1 (clamp)
 */
#define SIGMOID_C0 0x1.ffffecp-2f
#define SIGMOID_C1 0x1.0004dep-2f
#define SIGMOID_C2 -0x1.7a1fdp-13f
#define SIGMOID_C3 -0x1.4a9d98p-6f
#define SIGMOID_C4 -0x1.2c70aap-10f
#define SIGMOID_C5 0x1.9a181ep-9f
#define SIGMOID_C6 -0x1.c963f2p-12f
#define SIGMOID_C7 -0x1.f3fe06p-13f
#define SIGMOID_C8 0x1.ccac2ap-14f
#define SIGMOID_C9 -0x1.5074e6p-16f
#define SIGMOID_C10 0x1.e5c18cp-20f
#define SIGMOID_C11 -0x1.1e5e74p-24f

#define SIGMOID_CLAMP 5.0f


#ifdef LV_HAVE_GENERIC

static inline void volk_32f_sigmoid_32f_generic(float* outVector,
                                                 const float* inVector,
                                                 unsigned int num_points)
{
    for (unsigned int i = 0; i < num_points; i++) {
        float x = inVector[i];
        float ax = fabsf(x);

        float result;
        if (ax > SIGMOID_CLAMP) {
            result = (x > 0.0f) ? 1.0f : 0.0f;
        } else {
            /* Horner's method for polynomial evaluation */
            float y = SIGMOID_C11;
            y = y * ax + SIGMOID_C10;
            y = y * ax + SIGMOID_C9;
            y = y * ax + SIGMOID_C8;
            y = y * ax + SIGMOID_C7;
            y = y * ax + SIGMOID_C6;
            y = y * ax + SIGMOID_C5;
            y = y * ax + SIGMOID_C4;
            y = y * ax + SIGMOID_C3;
            y = y * ax + SIGMOID_C2;
            y = y * ax + SIGMOID_C1;
            y = y * ax + SIGMOID_C0;

            /* Use symmetry: sigmoid(-x) = 1 - sigmoid(x) */
            result = (x >= 0.0f) ? y : (1.0f - y);
        }
        outVector[i] = result;
    }
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_sigmoid_32f_a_avx(float* outVector,
                                               const float* inVector,
                                               unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    float* out_ptr = outVector;
    const float* in_ptr = inVector;

    const __m256 clamp = _mm256_set1_ps(SIGMOID_CLAMP);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    const __m256 c0 = _mm256_set1_ps(SIGMOID_C0);
    const __m256 c1 = _mm256_set1_ps(SIGMOID_C1);
    const __m256 c2 = _mm256_set1_ps(SIGMOID_C2);
    const __m256 c3 = _mm256_set1_ps(SIGMOID_C3);
    const __m256 c4 = _mm256_set1_ps(SIGMOID_C4);
    const __m256 c5 = _mm256_set1_ps(SIGMOID_C5);
    const __m256 c6 = _mm256_set1_ps(SIGMOID_C6);
    const __m256 c7 = _mm256_set1_ps(SIGMOID_C7);
    const __m256 c8 = _mm256_set1_ps(SIGMOID_C8);
    const __m256 c9 = _mm256_set1_ps(SIGMOID_C9);
    const __m256 c10 = _mm256_set1_ps(SIGMOID_C10);
    const __m256 c11 = _mm256_set1_ps(SIGMOID_C11);

    for (unsigned int i = 0; i < eighth_points; i++) {
        __m256 x = _mm256_load_ps(in_ptr);

        __m256 ax = _mm256_andnot_ps(sign_mask, x);
        __m256 neg_mask = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);

        /* Horner's method */
        __m256 y = c11;
        y = _mm256_add_ps(_mm256_mul_ps(y, ax), c10);
        y = _mm256_add_ps(_mm256_mul_ps(y, ax), c9);
        y = _mm256_add_ps(_mm256_mul_ps(y, ax), c8);
        y = _mm256_add_ps(_mm256_mul_ps(y, ax), c7);
        y = _mm256_add_ps(_mm256_mul_ps(y, ax), c6);
        y = _mm256_add_ps(_mm256_mul_ps(y, ax), c5);
        y = _mm256_add_ps(_mm256_mul_ps(y, ax), c4);
        y = _mm256_add_ps(_mm256_mul_ps(y, ax), c3);
        y = _mm256_add_ps(_mm256_mul_ps(y, ax), c2);
        y = _mm256_add_ps(_mm256_mul_ps(y, ax), c1);
        y = _mm256_add_ps(_mm256_mul_ps(y, ax), c0);

        /* Apply symmetry */
        __m256 y_neg = _mm256_sub_ps(one, y);
        y = _mm256_blendv_ps(y, y_neg, neg_mask);

        /* Clamp */
        __m256 clamp_mask = _mm256_cmp_ps(ax, clamp, _CMP_GT_OQ);
        __m256 clamp_val = _mm256_blendv_ps(one, zero, neg_mask);
        y = _mm256_blendv_ps(y, clamp_val, clamp_mask);

        _mm256_store_ps(out_ptr, y);
        in_ptr += 8;
        out_ptr += 8;
    }

    volk_32f_sigmoid_32f_generic(out_ptr, in_ptr, num_points - eighth_points * 8);
}

#endif /* LV_HAVE_AVX */


#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_sigmoid_32f_a_avx_fma(float* outVector,
                                                   const float* inVector,
                                                   unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    float* out_ptr = outVector;
    const float* in_ptr = inVector;

    const __m256 clamp = _mm256_set1_ps(SIGMOID_CLAMP);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    const __m256 c0 = _mm256_set1_ps(SIGMOID_C0);
    const __m256 c1 = _mm256_set1_ps(SIGMOID_C1);
    const __m256 c2 = _mm256_set1_ps(SIGMOID_C2);
    const __m256 c3 = _mm256_set1_ps(SIGMOID_C3);
    const __m256 c4 = _mm256_set1_ps(SIGMOID_C4);
    const __m256 c5 = _mm256_set1_ps(SIGMOID_C5);
    const __m256 c6 = _mm256_set1_ps(SIGMOID_C6);
    const __m256 c7 = _mm256_set1_ps(SIGMOID_C7);
    const __m256 c8 = _mm256_set1_ps(SIGMOID_C8);
    const __m256 c9 = _mm256_set1_ps(SIGMOID_C9);
    const __m256 c10 = _mm256_set1_ps(SIGMOID_C10);
    const __m256 c11 = _mm256_set1_ps(SIGMOID_C11);

    for (unsigned int i = 0; i < eighth_points; i++) {
        __m256 x = _mm256_load_ps(in_ptr);

        __m256 ax = _mm256_andnot_ps(sign_mask, x);
        __m256 neg_mask = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);

        /* Horner's method with FMA */
        __m256 y = _mm256_fmadd_ps(c11, ax, c10);
        y = _mm256_fmadd_ps(y, ax, c9);
        y = _mm256_fmadd_ps(y, ax, c8);
        y = _mm256_fmadd_ps(y, ax, c7);
        y = _mm256_fmadd_ps(y, ax, c6);
        y = _mm256_fmadd_ps(y, ax, c5);
        y = _mm256_fmadd_ps(y, ax, c4);
        y = _mm256_fmadd_ps(y, ax, c3);
        y = _mm256_fmadd_ps(y, ax, c2);
        y = _mm256_fmadd_ps(y, ax, c1);
        y = _mm256_fmadd_ps(y, ax, c0);

        __m256 y_neg = _mm256_sub_ps(one, y);
        y = _mm256_blendv_ps(y, y_neg, neg_mask);

        __m256 clamp_mask = _mm256_cmp_ps(ax, clamp, _CMP_GT_OQ);
        __m256 clamp_val = _mm256_blendv_ps(one, zero, neg_mask);
        y = _mm256_blendv_ps(y, clamp_val, clamp_mask);

        _mm256_store_ps(out_ptr, y);
        in_ptr += 8;
        out_ptr += 8;
    }

    volk_32f_sigmoid_32f_generic(out_ptr, in_ptr, num_points - eighth_points * 8);
}

#endif /* LV_HAVE_AVX && LV_HAVE_FMA */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_sigmoid_32f_a_avx512f(float* outVector,
                                                   const float* inVector,
                                                   unsigned int num_points)
{
    const unsigned int sixteenth_points = num_points / 16;
    float* out_ptr = outVector;
    const float* in_ptr = inVector;

    const __m512 clamp = _mm512_set1_ps(SIGMOID_CLAMP);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 zero = _mm512_setzero_ps();

    const __m512 c0 = _mm512_set1_ps(SIGMOID_C0);
    const __m512 c1 = _mm512_set1_ps(SIGMOID_C1);
    const __m512 c2 = _mm512_set1_ps(SIGMOID_C2);
    const __m512 c3 = _mm512_set1_ps(SIGMOID_C3);
    const __m512 c4 = _mm512_set1_ps(SIGMOID_C4);
    const __m512 c5 = _mm512_set1_ps(SIGMOID_C5);
    const __m512 c6 = _mm512_set1_ps(SIGMOID_C6);
    const __m512 c7 = _mm512_set1_ps(SIGMOID_C7);
    const __m512 c8 = _mm512_set1_ps(SIGMOID_C8);
    const __m512 c9 = _mm512_set1_ps(SIGMOID_C9);
    const __m512 c10 = _mm512_set1_ps(SIGMOID_C10);
    const __m512 c11 = _mm512_set1_ps(SIGMOID_C11);

    for (unsigned int i = 0; i < sixteenth_points; i++) {
        __m512 x = _mm512_load_ps(in_ptr);

        __m512 ax = _mm512_abs_ps(x);
        __mmask16 neg_mask = _mm512_cmp_ps_mask(x, zero, _CMP_LT_OQ);

        /* Horner's method with FMA */
        __m512 y = _mm512_fmadd_ps(c11, ax, c10);
        y = _mm512_fmadd_ps(y, ax, c9);
        y = _mm512_fmadd_ps(y, ax, c8);
        y = _mm512_fmadd_ps(y, ax, c7);
        y = _mm512_fmadd_ps(y, ax, c6);
        y = _mm512_fmadd_ps(y, ax, c5);
        y = _mm512_fmadd_ps(y, ax, c4);
        y = _mm512_fmadd_ps(y, ax, c3);
        y = _mm512_fmadd_ps(y, ax, c2);
        y = _mm512_fmadd_ps(y, ax, c1);
        y = _mm512_fmadd_ps(y, ax, c0);

        /* Apply symmetry */
        __m512 y_neg = _mm512_sub_ps(one, y);
        y = _mm512_mask_blend_ps(neg_mask, y, y_neg);

        /* Clamp */
        __mmask16 clamp_mask = _mm512_cmp_ps_mask(ax, clamp, _CMP_GT_OQ);
        __m512 clamp_val = _mm512_mask_blend_ps(neg_mask, one, zero);
        y = _mm512_mask_blend_ps(clamp_mask, y, clamp_val);

        _mm512_store_ps(out_ptr, y);
        in_ptr += 16;
        out_ptr += 16;
    }

    volk_32f_sigmoid_32f_generic(out_ptr, in_ptr, num_points - sixteenth_points * 16);
}

#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_sigmoid_32f_neon(float* outVector,
                                              const float* inVector,
                                              unsigned int num_points)
{
    const unsigned int quarter_points = num_points / 4;
    float* out_ptr = outVector;
    const float* in_ptr = inVector;

    const float32x4_t clamp = vdupq_n_f32(SIGMOID_CLAMP);
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);

    const float32x4_t c0 = vdupq_n_f32(SIGMOID_C0);
    const float32x4_t c1 = vdupq_n_f32(SIGMOID_C1);
    const float32x4_t c2 = vdupq_n_f32(SIGMOID_C2);
    const float32x4_t c3 = vdupq_n_f32(SIGMOID_C3);
    const float32x4_t c4 = vdupq_n_f32(SIGMOID_C4);
    const float32x4_t c5 = vdupq_n_f32(SIGMOID_C5);
    const float32x4_t c6 = vdupq_n_f32(SIGMOID_C6);
    const float32x4_t c7 = vdupq_n_f32(SIGMOID_C7);
    const float32x4_t c8 = vdupq_n_f32(SIGMOID_C8);
    const float32x4_t c9 = vdupq_n_f32(SIGMOID_C9);
    const float32x4_t c10 = vdupq_n_f32(SIGMOID_C10);
    const float32x4_t c11 = vdupq_n_f32(SIGMOID_C11);

    for (unsigned int i = 0; i < quarter_points; i++) {
        float32x4_t x = vld1q_f32(in_ptr);

        float32x4_t ax = vabsq_f32(x);
        uint32x4_t neg_mask = vcltq_f32(x, zero);

        /* Horner's method with FMA */
        float32x4_t y = vmlaq_f32(c10, c11, ax);
        y = vmlaq_f32(c9, y, ax);
        y = vmlaq_f32(c8, y, ax);
        y = vmlaq_f32(c7, y, ax);
        y = vmlaq_f32(c6, y, ax);
        y = vmlaq_f32(c5, y, ax);
        y = vmlaq_f32(c4, y, ax);
        y = vmlaq_f32(c3, y, ax);
        y = vmlaq_f32(c2, y, ax);
        y = vmlaq_f32(c1, y, ax);
        y = vmlaq_f32(c0, y, ax);

        /* Apply symmetry */
        float32x4_t y_neg = vsubq_f32(one, y);
        y = vbslq_f32(neg_mask, y_neg, y);

        /* Clamp */
        uint32x4_t clamp_mask = vcgtq_f32(ax, clamp);
        float32x4_t clamp_val = vbslq_f32(neg_mask, zero, one);
        y = vbslq_f32(clamp_mask, clamp_val, y);

        vst1q_f32(out_ptr, y);
        in_ptr += 4;
        out_ptr += 4;
    }

    volk_32f_sigmoid_32f_generic(out_ptr, in_ptr, num_points - quarter_points * 4);
}

#endif /* LV_HAVE_NEON */


#endif /* INCLUDED_volk_32f_sigmoid_32f_a_H */


#ifndef INCLUDED_volk_32f_sigmoid_32f_u_H
#define INCLUDED_volk_32f_sigmoid_32f_u_H

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_sigmoid_32f_u_avx(float* outVector,
                                               const float* inVector,
                                               unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    float* out_ptr = outVector;
    const float* in_ptr = inVector;

    const __m256 clamp = _mm256_set1_ps(SIGMOID_CLAMP);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    const __m256 c0 = _mm256_set1_ps(SIGMOID_C0);
    const __m256 c1 = _mm256_set1_ps(SIGMOID_C1);
    const __m256 c2 = _mm256_set1_ps(SIGMOID_C2);
    const __m256 c3 = _mm256_set1_ps(SIGMOID_C3);
    const __m256 c4 = _mm256_set1_ps(SIGMOID_C4);
    const __m256 c5 = _mm256_set1_ps(SIGMOID_C5);
    const __m256 c6 = _mm256_set1_ps(SIGMOID_C6);
    const __m256 c7 = _mm256_set1_ps(SIGMOID_C7);
    const __m256 c8 = _mm256_set1_ps(SIGMOID_C8);
    const __m256 c9 = _mm256_set1_ps(SIGMOID_C9);
    const __m256 c10 = _mm256_set1_ps(SIGMOID_C10);
    const __m256 c11 = _mm256_set1_ps(SIGMOID_C11);

    for (unsigned int i = 0; i < eighth_points; i++) {
        __m256 x = _mm256_loadu_ps(in_ptr);

        __m256 ax = _mm256_andnot_ps(sign_mask, x);
        __m256 neg_mask = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);

        __m256 y = c11;
        y = _mm256_add_ps(_mm256_mul_ps(y, ax), c10);
        y = _mm256_add_ps(_mm256_mul_ps(y, ax), c9);
        y = _mm256_add_ps(_mm256_mul_ps(y, ax), c8);
        y = _mm256_add_ps(_mm256_mul_ps(y, ax), c7);
        y = _mm256_add_ps(_mm256_mul_ps(y, ax), c6);
        y = _mm256_add_ps(_mm256_mul_ps(y, ax), c5);
        y = _mm256_add_ps(_mm256_mul_ps(y, ax), c4);
        y = _mm256_add_ps(_mm256_mul_ps(y, ax), c3);
        y = _mm256_add_ps(_mm256_mul_ps(y, ax), c2);
        y = _mm256_add_ps(_mm256_mul_ps(y, ax), c1);
        y = _mm256_add_ps(_mm256_mul_ps(y, ax), c0);

        __m256 y_neg = _mm256_sub_ps(one, y);
        y = _mm256_blendv_ps(y, y_neg, neg_mask);

        __m256 clamp_mask = _mm256_cmp_ps(ax, clamp, _CMP_GT_OQ);
        __m256 clamp_val = _mm256_blendv_ps(one, zero, neg_mask);
        y = _mm256_blendv_ps(y, clamp_val, clamp_mask);

        _mm256_storeu_ps(out_ptr, y);
        in_ptr += 8;
        out_ptr += 8;
    }

    volk_32f_sigmoid_32f_generic(out_ptr, in_ptr, num_points - eighth_points * 8);
}

#endif /* LV_HAVE_AVX */


#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_sigmoid_32f_u_avx_fma(float* outVector,
                                                   const float* inVector,
                                                   unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    float* out_ptr = outVector;
    const float* in_ptr = inVector;

    const __m256 clamp = _mm256_set1_ps(SIGMOID_CLAMP);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    const __m256 c0 = _mm256_set1_ps(SIGMOID_C0);
    const __m256 c1 = _mm256_set1_ps(SIGMOID_C1);
    const __m256 c2 = _mm256_set1_ps(SIGMOID_C2);
    const __m256 c3 = _mm256_set1_ps(SIGMOID_C3);
    const __m256 c4 = _mm256_set1_ps(SIGMOID_C4);
    const __m256 c5 = _mm256_set1_ps(SIGMOID_C5);
    const __m256 c6 = _mm256_set1_ps(SIGMOID_C6);
    const __m256 c7 = _mm256_set1_ps(SIGMOID_C7);
    const __m256 c8 = _mm256_set1_ps(SIGMOID_C8);
    const __m256 c9 = _mm256_set1_ps(SIGMOID_C9);
    const __m256 c10 = _mm256_set1_ps(SIGMOID_C10);
    const __m256 c11 = _mm256_set1_ps(SIGMOID_C11);

    for (unsigned int i = 0; i < eighth_points; i++) {
        __m256 x = _mm256_loadu_ps(in_ptr);

        __m256 ax = _mm256_andnot_ps(sign_mask, x);
        __m256 neg_mask = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);

        __m256 y = _mm256_fmadd_ps(c11, ax, c10);
        y = _mm256_fmadd_ps(y, ax, c9);
        y = _mm256_fmadd_ps(y, ax, c8);
        y = _mm256_fmadd_ps(y, ax, c7);
        y = _mm256_fmadd_ps(y, ax, c6);
        y = _mm256_fmadd_ps(y, ax, c5);
        y = _mm256_fmadd_ps(y, ax, c4);
        y = _mm256_fmadd_ps(y, ax, c3);
        y = _mm256_fmadd_ps(y, ax, c2);
        y = _mm256_fmadd_ps(y, ax, c1);
        y = _mm256_fmadd_ps(y, ax, c0);

        __m256 y_neg = _mm256_sub_ps(one, y);
        y = _mm256_blendv_ps(y, y_neg, neg_mask);

        __m256 clamp_mask = _mm256_cmp_ps(ax, clamp, _CMP_GT_OQ);
        __m256 clamp_val = _mm256_blendv_ps(one, zero, neg_mask);
        y = _mm256_blendv_ps(y, clamp_val, clamp_mask);

        _mm256_storeu_ps(out_ptr, y);
        in_ptr += 8;
        out_ptr += 8;
    }

    volk_32f_sigmoid_32f_generic(out_ptr, in_ptr, num_points - eighth_points * 8);
}

#endif /* LV_HAVE_AVX && LV_HAVE_FMA */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_sigmoid_32f_u_avx512f(float* outVector,
                                                   const float* inVector,
                                                   unsigned int num_points)
{
    const unsigned int sixteenth_points = num_points / 16;
    float* out_ptr = outVector;
    const float* in_ptr = inVector;

    const __m512 clamp = _mm512_set1_ps(SIGMOID_CLAMP);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 zero = _mm512_setzero_ps();

    const __m512 c0 = _mm512_set1_ps(SIGMOID_C0);
    const __m512 c1 = _mm512_set1_ps(SIGMOID_C1);
    const __m512 c2 = _mm512_set1_ps(SIGMOID_C2);
    const __m512 c3 = _mm512_set1_ps(SIGMOID_C3);
    const __m512 c4 = _mm512_set1_ps(SIGMOID_C4);
    const __m512 c5 = _mm512_set1_ps(SIGMOID_C5);
    const __m512 c6 = _mm512_set1_ps(SIGMOID_C6);
    const __m512 c7 = _mm512_set1_ps(SIGMOID_C7);
    const __m512 c8 = _mm512_set1_ps(SIGMOID_C8);
    const __m512 c9 = _mm512_set1_ps(SIGMOID_C9);
    const __m512 c10 = _mm512_set1_ps(SIGMOID_C10);
    const __m512 c11 = _mm512_set1_ps(SIGMOID_C11);

    for (unsigned int i = 0; i < sixteenth_points; i++) {
        __m512 x = _mm512_loadu_ps(in_ptr);

        __m512 ax = _mm512_abs_ps(x);
        __mmask16 neg_mask = _mm512_cmp_ps_mask(x, zero, _CMP_LT_OQ);

        __m512 y = _mm512_fmadd_ps(c11, ax, c10);
        y = _mm512_fmadd_ps(y, ax, c9);
        y = _mm512_fmadd_ps(y, ax, c8);
        y = _mm512_fmadd_ps(y, ax, c7);
        y = _mm512_fmadd_ps(y, ax, c6);
        y = _mm512_fmadd_ps(y, ax, c5);
        y = _mm512_fmadd_ps(y, ax, c4);
        y = _mm512_fmadd_ps(y, ax, c3);
        y = _mm512_fmadd_ps(y, ax, c2);
        y = _mm512_fmadd_ps(y, ax, c1);
        y = _mm512_fmadd_ps(y, ax, c0);

        __m512 y_neg = _mm512_sub_ps(one, y);
        y = _mm512_mask_blend_ps(neg_mask, y, y_neg);

        __mmask16 clamp_mask = _mm512_cmp_ps_mask(ax, clamp, _CMP_GT_OQ);
        __m512 clamp_val = _mm512_mask_blend_ps(neg_mask, one, zero);
        y = _mm512_mask_blend_ps(clamp_mask, y, clamp_val);

        _mm512_storeu_ps(out_ptr, y);
        in_ptr += 16;
        out_ptr += 16;
    }

    volk_32f_sigmoid_32f_generic(out_ptr, in_ptr, num_points - sixteenth_points * 16);
}

#endif /* LV_HAVE_AVX512F */


#endif /* INCLUDED_volk_32f_sigmoid_32f_u_H */
