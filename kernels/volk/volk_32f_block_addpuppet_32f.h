/* -*- c++ -*- */
/*
 * Copyright 2025 Magnus Lundin
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_block_addpuppet_32f
 *
 * \b Overview
 *
 * Test puppet for volk_32f_block_add_32f. Uses a fixed num_blocks of 4.
 * The input vector length is used as the total input size, with
 * num_output_points = num_points / 4.
 */

#ifndef INCLUDED_volk_32f_block_addpuppet_32f_u_H
#define INCLUDED_volk_32f_block_addpuppet_32f_u_H

#include <volk/volk_32f_block_add_32f.h>

#define PUPPET_NUM_BLOCKS 4

#ifdef LV_HAVE_GENERIC
static inline void volk_32f_block_addpuppet_32f_generic(float* output,
                                                        const float* input,
                                                        unsigned int num_points)
{
    unsigned int num_output_points = num_points / PUPPET_NUM_BLOCKS;
    volk_32f_block_add_32f_generic(output, input, num_output_points, PUPPET_NUM_BLOCKS);
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE
static inline void volk_32f_block_addpuppet_32f_u_sse(float* output,
                                                      const float* input,
                                                      unsigned int num_points)
{
    unsigned int num_output_points = num_points / PUPPET_NUM_BLOCKS;
    volk_32f_block_add_32f_u_sse(output, input, num_output_points, PUPPET_NUM_BLOCKS);
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_AVX
static inline void volk_32f_block_addpuppet_32f_u_avx(float* output,
                                                      const float* input,
                                                      unsigned int num_points)
{
    unsigned int num_output_points = num_points / PUPPET_NUM_BLOCKS;
    volk_32f_block_add_32f_u_avx(output, input, num_output_points, PUPPET_NUM_BLOCKS);
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_AVX512F
static inline void volk_32f_block_addpuppet_32f_u_avx512f(float* output,
                                                          const float* input,
                                                          unsigned int num_points)
{
    unsigned int num_output_points = num_points / PUPPET_NUM_BLOCKS;
    volk_32f_block_add_32f_u_avx512f(output, input, num_output_points, PUPPET_NUM_BLOCKS);
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_RVV
static inline void volk_32f_block_addpuppet_32f_rvv(float* output,
                                                    const float* input,
                                                    unsigned int num_points)
{
    unsigned int num_output_points = num_points / PUPPET_NUM_BLOCKS;
    volk_32f_block_add_32f_rvv(output, input, num_output_points, PUPPET_NUM_BLOCKS);
}
#endif /* LV_HAVE_RVV */


#endif /* INCLUDED_volk_32f_block_addpuppet_32f_u_H */


#ifndef INCLUDED_volk_32f_block_addpuppet_32f_a_H
#define INCLUDED_volk_32f_block_addpuppet_32f_a_H

#include <volk/volk_32f_block_add_32f.h>

#ifndef PUPPET_NUM_BLOCKS
#define PUPPET_NUM_BLOCKS 4
#endif

#ifdef LV_HAVE_SSE
static inline void volk_32f_block_addpuppet_32f_a_sse(float* output,
                                                      const float* input,
                                                      unsigned int num_points)
{
    unsigned int num_output_points = num_points / PUPPET_NUM_BLOCKS;
    volk_32f_block_add_32f_a_sse(output, input, num_output_points, PUPPET_NUM_BLOCKS);
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_AVX
static inline void volk_32f_block_addpuppet_32f_a_avx(float* output,
                                                      const float* input,
                                                      unsigned int num_points)
{
    unsigned int num_output_points = num_points / PUPPET_NUM_BLOCKS;
    volk_32f_block_add_32f_a_avx(output, input, num_output_points, PUPPET_NUM_BLOCKS);
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_AVX512F
static inline void volk_32f_block_addpuppet_32f_a_avx512f(float* output,
                                                          const float* input,
                                                          unsigned int num_points)
{
    unsigned int num_output_points = num_points / PUPPET_NUM_BLOCKS;
    volk_32f_block_add_32f_a_avx512f(output, input, num_output_points, PUPPET_NUM_BLOCKS);
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_NEON
static inline void volk_32f_block_addpuppet_32f_neon(float* output,
                                                     const float* input,
                                                     unsigned int num_points)
{
    unsigned int num_output_points = num_points / PUPPET_NUM_BLOCKS;
    volk_32f_block_add_32f_neon(output, input, num_output_points, PUPPET_NUM_BLOCKS);
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
static inline void volk_32f_block_addpuppet_32f_neonv8(float* output,
                                                       const float* input,
                                                       unsigned int num_points)
{
    unsigned int num_output_points = num_points / PUPPET_NUM_BLOCKS;
    volk_32f_block_add_32f_neonv8(output, input, num_output_points, PUPPET_NUM_BLOCKS);
}
#endif /* LV_HAVE_NEONV8 */


#endif /* INCLUDED_volk_32f_block_addpuppet_32f_a_H */
