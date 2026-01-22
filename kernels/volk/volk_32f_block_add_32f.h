/* -*- c++ -*- */
/*
 * Copyright 2025 Magnus Lundin
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_block_add_32f
 *
 * \b Overview
 *
 * Adds together multiple consecutive blocks of a vector element-wise.
 * This is useful for frequency-domain decimation (spectral folding).
 *
 * Given an input of length num_output_points * num_blocks, produces
 * an output of length num_output_points where:
 *
 * output[i] = input[i] + input[i + stride] + input[i + 2*stride] + ...
 *
 * where stride = num_output_points.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_block_add_32f(float* output, const float* input,
 *     unsigned int num_output_points, unsigned int num_blocks)
 * \endcode
 *
 * \b Inputs
 * \li input: The input buffer of length num_output_points * num_blocks
 * \li num_output_points: The number of output points (also the block size)
 * \li num_blocks: The number of blocks to sum together
 *
 * \b Outputs
 * \li output: The output buffer of length num_output_points
 *
 * \b Example
 * Fold a 1024-element vector into 256 elements by summing 4 blocks:
 * \code
 *   unsigned int num_output_points = 256;
 *   unsigned int num_blocks = 4;
 *   unsigned int alignment = volk_get_alignment();
 *   float* input = (float*)volk_malloc(sizeof(float) * num_output_points * num_blocks,
 *                                      alignment);
 *   float* output = (float*)volk_malloc(sizeof(float) * num_output_points, alignment);
 *
 *   // Fill input with test data
 *   for (unsigned int i = 0; i < num_output_points * num_blocks; ++i) {
 *       input[i] = (float)i;
 *   }
 *
 *   volk_32f_block_add_32f(output, input, num_output_points, num_blocks);
 *
 *   // output[i] = input[i] + input[i+256] + input[i+512] + input[i+768]
 *
 *   volk_free(input);
 *   volk_free(output);
 * \endcode
 *
 * \b Note
 * For complex data (lv_32fc_t), cast to float* and double num_output_points.
 * The interleaved real/imag storage ensures correct element-wise addition.
 */

#ifndef INCLUDED_volk_32f_block_add_32f_u_H
#define INCLUDED_volk_32f_block_add_32f_u_H

#include <inttypes.h>
#include <string.h>
#include <volk/volk_common.h>


#ifdef LV_HAVE_GENERIC

static inline void volk_32f_block_add_32f_generic(float* output,
                                                  const float* input,
                                                  unsigned int num_output_points,
                                                  unsigned int num_blocks)
{
    // Copy first block to output
    memcpy(output, input, num_output_points * sizeof(float));

    // Add remaining blocks
    for (unsigned int block = 1; block < num_blocks; block++) {
        const float* block_ptr = input + block * num_output_points;
        for (unsigned int i = 0; i < num_output_points; i++) {
            output[i] += block_ptr[i];
        }
    }
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_block_add_32f_u_sse(float* output,
                                                const float* input,
                                                unsigned int num_output_points,
                                                unsigned int num_blocks)
{
    const unsigned int quarter_points = num_output_points / 4;
    const unsigned int remainder = num_output_points % 4;

    // Copy first block to output
    memcpy(output, input, num_output_points * sizeof(float));

    // Add remaining blocks
    for (unsigned int block = 1; block < num_blocks; block++) {
        const float* block_ptr = input + block * num_output_points;
        float* out_ptr = output;

        for (unsigned int i = 0; i < quarter_points; i++) {
            __m128 out_val = _mm_loadu_ps(out_ptr);
            __m128 in_val = _mm_loadu_ps(block_ptr);
            out_val = _mm_add_ps(out_val, in_val);
            _mm_storeu_ps(out_ptr, out_val);
            out_ptr += 4;
            block_ptr += 4;
        }

        for (unsigned int i = 0; i < remainder; i++) {
            *out_ptr++ += *block_ptr++;
        }
    }
}

#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_block_add_32f_u_avx(float* output,
                                                const float* input,
                                                unsigned int num_output_points,
                                                unsigned int num_blocks)
{
    const unsigned int eighth_points = num_output_points / 8;
    const unsigned int remainder = num_output_points % 8;

    // Copy first block to output
    memcpy(output, input, num_output_points * sizeof(float));

    // Add remaining blocks
    for (unsigned int block = 1; block < num_blocks; block++) {
        const float* block_ptr = input + block * num_output_points;
        float* out_ptr = output;

        for (unsigned int i = 0; i < eighth_points; i++) {
            __m256 out_val = _mm256_loadu_ps(out_ptr);
            __m256 in_val = _mm256_loadu_ps(block_ptr);
            out_val = _mm256_add_ps(out_val, in_val);
            _mm256_storeu_ps(out_ptr, out_val);
            out_ptr += 8;
            block_ptr += 8;
        }

        for (unsigned int i = 0; i < remainder; i++) {
            *out_ptr++ += *block_ptr++;
        }
    }
}

#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_block_add_32f_u_avx512f(float* output,
                                                    const float* input,
                                                    unsigned int num_output_points,
                                                    unsigned int num_blocks)
{
    const unsigned int sixteenth_points = num_output_points / 16;
    const unsigned int remainder = num_output_points % 16;

    // Copy first block to output
    memcpy(output, input, num_output_points * sizeof(float));

    // Add remaining blocks
    for (unsigned int block = 1; block < num_blocks; block++) {
        const float* block_ptr = input + block * num_output_points;
        float* out_ptr = output;

        for (unsigned int i = 0; i < sixteenth_points; i++) {
            __m512 out_val = _mm512_loadu_ps(out_ptr);
            __m512 in_val = _mm512_loadu_ps(block_ptr);
            out_val = _mm512_add_ps(out_val, in_val);
            _mm512_storeu_ps(out_ptr, out_val);
            out_ptr += 16;
            block_ptr += 16;
        }

        for (unsigned int i = 0; i < remainder; i++) {
            *out_ptr++ += *block_ptr++;
        }
    }
}

#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_block_add_32f_rvv(float* output,
                                              const float* input,
                                              unsigned int num_output_points,
                                              unsigned int num_blocks)
{
    // Copy first block to output
    memcpy(output, input, num_output_points * sizeof(float));

    // Add remaining blocks
    for (unsigned int block = 1; block < num_blocks; block++) {
        const float* block_ptr = input + block * num_output_points;
        float* out_ptr = output;
        size_t n = num_output_points;

        for (size_t vl; n > 0; n -= vl, out_ptr += vl, block_ptr += vl) {
            vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t out_val = __riscv_vle32_v_f32m8(out_ptr, vl);
            vfloat32m8_t in_val = __riscv_vle32_v_f32m8(block_ptr, vl);
            __riscv_vse32(out_ptr, __riscv_vfadd(out_val, in_val, vl), vl);
        }
    }
}

#endif /* LV_HAVE_RVV */


#endif /* INCLUDED_volk_32f_block_add_32f_u_H */


#ifndef INCLUDED_volk_32f_block_add_32f_a_H
#define INCLUDED_volk_32f_block_add_32f_a_H

#include <inttypes.h>
#include <string.h>
#include <volk/volk_common.h>


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_block_add_32f_a_sse(float* output,
                                                const float* input,
                                                unsigned int num_output_points,
                                                unsigned int num_blocks)
{
    const unsigned int quarter_points = num_output_points / 4;
    const unsigned int remainder = num_output_points % 4;

    // Copy first block to output
    memcpy(output, input, num_output_points * sizeof(float));

    // Add remaining blocks
    for (unsigned int block = 1; block < num_blocks; block++) {
        const float* block_ptr = input + block * num_output_points;
        float* out_ptr = output;

        for (unsigned int i = 0; i < quarter_points; i++) {
            __m128 out_val = _mm_load_ps(out_ptr);
            __m128 in_val = _mm_load_ps(block_ptr);
            out_val = _mm_add_ps(out_val, in_val);
            _mm_store_ps(out_ptr, out_val);
            out_ptr += 4;
            block_ptr += 4;
        }

        for (unsigned int i = 0; i < remainder; i++) {
            *out_ptr++ += *block_ptr++;
        }
    }
}

#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_block_add_32f_a_avx(float* output,
                                                const float* input,
                                                unsigned int num_output_points,
                                                unsigned int num_blocks)
{
    const unsigned int eighth_points = num_output_points / 8;
    const unsigned int remainder = num_output_points % 8;

    // Copy first block to output
    memcpy(output, input, num_output_points * sizeof(float));

    // Add remaining blocks
    for (unsigned int block = 1; block < num_blocks; block++) {
        const float* block_ptr = input + block * num_output_points;
        float* out_ptr = output;

        for (unsigned int i = 0; i < eighth_points; i++) {
            __m256 out_val = _mm256_load_ps(out_ptr);
            __m256 in_val = _mm256_load_ps(block_ptr);
            out_val = _mm256_add_ps(out_val, in_val);
            _mm256_store_ps(out_ptr, out_val);
            out_ptr += 8;
            block_ptr += 8;
        }

        for (unsigned int i = 0; i < remainder; i++) {
            *out_ptr++ += *block_ptr++;
        }
    }
}

#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_block_add_32f_a_avx512f(float* output,
                                                    const float* input,
                                                    unsigned int num_output_points,
                                                    unsigned int num_blocks)
{
    const unsigned int sixteenth_points = num_output_points / 16;
    const unsigned int remainder = num_output_points % 16;

    // Copy first block to output
    memcpy(output, input, num_output_points * sizeof(float));

    // Add remaining blocks
    for (unsigned int block = 1; block < num_blocks; block++) {
        const float* block_ptr = input + block * num_output_points;
        float* out_ptr = output;

        for (unsigned int i = 0; i < sixteenth_points; i++) {
            __m512 out_val = _mm512_load_ps(out_ptr);
            __m512 in_val = _mm512_load_ps(block_ptr);
            out_val = _mm512_add_ps(out_val, in_val);
            _mm512_store_ps(out_ptr, out_val);
            out_ptr += 16;
            block_ptr += 16;
        }

        for (unsigned int i = 0; i < remainder; i++) {
            *out_ptr++ += *block_ptr++;
        }
    }
}

#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_block_add_32f_neon(float* output,
                                               const float* input,
                                               unsigned int num_output_points,
                                               unsigned int num_blocks)
{
    const unsigned int quarter_points = num_output_points / 4;
    const unsigned int remainder = num_output_points % 4;

    // Copy first block to output
    memcpy(output, input, num_output_points * sizeof(float));

    // Add remaining blocks
    for (unsigned int block = 1; block < num_blocks; block++) {
        const float* block_ptr = input + block * num_output_points;
        float* out_ptr = output;

        for (unsigned int i = 0; i < quarter_points; i++) {
            float32x4_t out_val = vld1q_f32(out_ptr);
            float32x4_t in_val = vld1q_f32(block_ptr);
            __VOLK_PREFETCH(block_ptr + 4);
            out_val = vaddq_f32(out_val, in_val);
            vst1q_f32(out_ptr, out_val);
            out_ptr += 4;
            block_ptr += 4;
        }

        for (unsigned int i = 0; i < remainder; i++) {
            *out_ptr++ += *block_ptr++;
        }
    }
}

#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_block_add_32f_neonv8(float* output,
                                                 const float* input,
                                                 unsigned int num_output_points,
                                                 unsigned int num_blocks)
{
    const unsigned int eighth_points = num_output_points / 8;
    const unsigned int remainder = num_output_points % 8;

    // Copy first block to output
    memcpy(output, input, num_output_points * sizeof(float));

    // Add remaining blocks
    for (unsigned int block = 1; block < num_blocks; block++) {
        const float* block_ptr = input + block * num_output_points;
        float* out_ptr = output;

        // 2x unrolled for better ILP
        for (unsigned int i = 0; i < eighth_points; i++) {
            float32x4_t out_val0 = vld1q_f32(out_ptr);
            float32x4_t out_val1 = vld1q_f32(out_ptr + 4);
            float32x4_t in_val0 = vld1q_f32(block_ptr);
            float32x4_t in_val1 = vld1q_f32(block_ptr + 4);
            __VOLK_PREFETCH(block_ptr + 8);

            out_val0 = vaddq_f32(out_val0, in_val0);
            out_val1 = vaddq_f32(out_val1, in_val1);

            vst1q_f32(out_ptr, out_val0);
            vst1q_f32(out_ptr + 4, out_val1);

            out_ptr += 8;
            block_ptr += 8;
        }

        // Handle remaining 4 floats
        unsigned int remaining = remainder;
        if (remaining >= 4) {
            float32x4_t out_val = vld1q_f32(out_ptr);
            float32x4_t in_val = vld1q_f32(block_ptr);
            vst1q_f32(out_ptr, vaddq_f32(out_val, in_val));
            out_ptr += 4;
            block_ptr += 4;
            remaining -= 4;
        }

        // Scalar tail
        for (unsigned int i = 0; i < remaining; i++) {
            *out_ptr++ += *block_ptr++;
        }
    }
}

#endif /* LV_HAVE_NEONV8 */


#endif /* INCLUDED_volk_32f_block_add_32f_a_H */
