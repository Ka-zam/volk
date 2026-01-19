/* -*- c++ -*- */
/*
 * Copyright 2024 Magnus Lundmark <magnuslundmark@gmail.com>
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
 * in the output vector. SIMD implementations use fast approximate reciprocal
 * instructions with Newton-Raphson refinement for near-single-precision accuracy.
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
/*
 * Double-precision reference implementation for accurate error measurement
 * Uses volatile to prevent compiler from optimizing away the double-precision math
 */
static inline void
volk_32f_reciprocal_32f_generic(float* out, const float* in, unsigned int num_points)
{
    for (unsigned int i = 0; i < num_points; i++) {
        volatile double x = (double)in[i];
        volatile double result = 1.0 / x;
        out[i] = (float)result;
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_GENERIC
static inline void
volk_32f_reciprocal_32f_float(float* out, const float* in, unsigned int num_points)
{
    for (unsigned int i = 0; i < num_points; i++) {
        out[i] = 1.f / in[i];
    }
}
#endif /* LV_HAVE_GENERIC */

/*
 * Magic number reciprocal (Q_rcp) - similar to Q_rsqrt but for 1/x
 * Uses bit manipulation for initial approximation followed by Newton-Raphson
 */

#ifdef LV_HAVE_GENERIC
static inline void
volk_32f_reciprocal_32f_M_rcp(float* out, const float* in, unsigned int num_points)
{
    union {
        float f;
        uint32_t u;
    } conv, result;

    for (unsigned int i = 0; i < num_points; i++) {
        float x = in[i];
        conv.f = x;
        uint32_t input_bits = conv.u;

        // Magic bit trick: approximate 1/x via integer subtraction
        // log2(1/x) = -log2(x), so we manipulate the exponent bits
        conv.u = 0x7EF311C3 - conv.u;
        float y = conv.f;

        // Newton-Raphson iterations: y = y * (2 - x * y)
        y = y * (2.0f - x * y); // Iteration 1
        y = y * (2.0f - x * y); // Iteration 2

        // Branchless special case handling (±0 → ±Inf, ±Inf → ±0)
        result.f = y;
        uint32_t abs_bits = input_bits & 0x7FFFFFFF;
        uint32_t sign_bit = input_bits & 0x80000000;
        uint32_t is_zero = (uint32_t)(-(int32_t)(abs_bits == 0x00000000));
        uint32_t is_inf = (uint32_t)(-(int32_t)(abs_bits == 0x7F800000));
        result.u = (result.u & ~is_zero & ~is_inf) |
                   ((0x7F800000 | sign_bit) & is_zero) | // ±0 → ±Inf
                   (sign_bit & is_inf);                  // ±Inf → ±0
        out[i] = result.f;
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE
#include <volk/volk_sse_intrinsics.h>
static inline void
volk_32f_reciprocal_32f_a_sse(float* out, const float* in, unsigned int num_points)
{
    const unsigned int quarter_points = num_points / 4;

    for (unsigned int number = 0; number < quarter_points; number++) {
        __m128 x = _mm_load_ps(in);
        in += 4;
        __m128 r = _mm_rcp_nr_ps(x);
        _mm_store_ps(out, r);
        out += 4;
    }

    const unsigned int done = quarter_points * 4;

    volk_32f_reciprocal_32f_generic(out, in, num_points - done);
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_AVX
#include <volk/volk_avx_intrinsics.h>
static inline void
volk_32f_reciprocal_32f_a_avx(float* out, const float* in, unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;

    for (unsigned int number = 0; number < eighth_points; number++) {
        __m256 x = _mm256_load_ps(in);
        in += 8;
        __m256 r = _mm256_rcp_nr_ps(x);
        _mm256_store_ps(out, r);
        out += 8;
    }

    const unsigned int done = eighth_points * 8;

    volk_32f_reciprocal_32f_generic(out, in, num_points - done);
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <volk/volk_avx2_intrinsics.h>
static inline void
volk_32f_reciprocal_32f_a_avx2(float* out, const float* in, unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;

    for (unsigned int number = 0; number < eighth_points; number++) {
        __m256 x = _mm256_load_ps(in);
        in += 8;
        __m256 r = _mm256_rcp_nr_avx2(x);
        _mm256_store_ps(out, r);
        out += 8;
    }

    const unsigned int done = eighth_points * 8;

    volk_32f_reciprocal_32f_generic(out, in, num_points - done);
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <volk/volk_avx512_intrinsics.h>
static inline void
volk_32f_reciprocal_32f_a_avx512(float* out, const float* in, unsigned int num_points)
{
    const unsigned int sixteenth_points = num_points / 16;

    for (unsigned int number = 0; number < sixteenth_points; number++) {
        __m512 x = _mm512_load_ps(in);
        in += 16;
        __m512 r = _mm512_rcp_nr_ps(x);
        _mm512_store_ps(out, r);
        out += 16;
    }

    const unsigned int done = sixteenth_points * 16;

    volk_32f_reciprocal_32f_generic(out, in, num_points - done);
}
#endif /* LV_HAVE_AVX512F */

#endif /* INCLUDED_volk_32f_reciprocal_32f_a_H */

#ifndef INCLUDED_volk_32f_reciprocal_32f_u_H
#define INCLUDED_volk_32f_reciprocal_32f_u_H

#ifdef LV_HAVE_SSE
#include <volk/volk_sse_intrinsics.h>
static inline void
volk_32f_reciprocal_32f_u_sse(float* out, const float* in, unsigned int num_points)
{
    const unsigned int quarter_points = num_points / 4;

    for (unsigned int number = 0; number < quarter_points; number++) {
        __m128 x = _mm_loadu_ps(in);
        in += 4;
        __m128 r = _mm_rcp_nr_ps(x);
        _mm_storeu_ps(out, r);
        out += 4;
    }

    const unsigned int done = quarter_points * 4;

    volk_32f_reciprocal_32f_generic(out, in, num_points - done);
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_AVX
#include <volk/volk_avx_intrinsics.h>
static inline void
volk_32f_reciprocal_32f_u_avx(float* out, const float* in, unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;

    for (unsigned int number = 0; number < eighth_points; number++) {
        __m256 x = _mm256_loadu_ps(in);
        in += 8;
        __m256 r = _mm256_rcp_nr_ps(x);
        _mm256_storeu_ps(out, r);
        out += 8;
    }

    const unsigned int done = eighth_points * 8;

    volk_32f_reciprocal_32f_generic(out, in, num_points - done);
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <volk/volk_avx2_intrinsics.h>
static inline void
volk_32f_reciprocal_32f_u_avx2(float* out, const float* in, unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;

    for (unsigned int number = 0; number < eighth_points; number++) {
        __m256 x = _mm256_loadu_ps(in);
        in += 8;
        __m256 r = _mm256_rcp_nr_avx2(x);
        _mm256_storeu_ps(out, r);
        out += 8;
    }

    const unsigned int done = eighth_points * 8;

    volk_32f_reciprocal_32f_generic(out, in, num_points - done);
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <volk/volk_avx512_intrinsics.h>
static inline void
volk_32f_reciprocal_32f_u_avx512(float* out, const float* in, unsigned int num_points)
{
    const unsigned int sixteenth_points = num_points / 16;

    for (unsigned int number = 0; number < sixteenth_points; number++) {
        __m512 x = _mm512_loadu_ps(in);
        in += 16;
        __m512 r = _mm512_rcp_nr_ps(x);
        _mm512_storeu_ps(out, r);
        out += 16;
    }

    const unsigned int done = sixteenth_points * 16;

    volk_32f_reciprocal_32f_generic(out, in, num_points - done);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void
volk_32f_reciprocal_32f_neonv8(float* out, const float* in, unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;

    for (unsigned int number = 0; number < eighth_points; number++) {
        float32x4_t x0 = vld1q_f32(in);
        float32x4_t x1 = vld1q_f32(in + 4);
        __VOLK_PREFETCH(in + 8);
        in += 8;

        // Initial estimate: ~8-bit precision
        float32x4_t y0_0 = vrecpeq_f32(x0);
        float32x4_t y0_1 = vrecpeq_f32(x1);

        // Newton-Raphson iteration 1: y = y * vrecps(x, y)
        // vrecps(a,b) computes 2 - a*b
        y0_0 = vmulq_f32(y0_0, vrecpsq_f32(x0, y0_0));
        y0_1 = vmulq_f32(y0_1, vrecpsq_f32(x1, y0_1));

        // Newton-Raphson iteration 2 for full precision
        y0_0 = vmulq_f32(y0_0, vrecpsq_f32(x0, y0_0));
        y0_1 = vmulq_f32(y0_1, vrecpsq_f32(x1, y0_1));

        vst1q_f32(out, y0_0);
        vst1q_f32(out + 4, y0_1);
        out += 8;
    }

    const unsigned int done = eighth_points * 8;
    for (unsigned int i = done; i < num_points; i++) {
        *out++ = 1.0f / *in++;
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void
volk_32f_reciprocal_32f_rvv(float* out, const float* in, unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, in += vl, out += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(in, vl);
        __riscv_vse32(out, __riscv_vfrdiv(v, 1.0f, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_reciprocal_32f_u_H */
