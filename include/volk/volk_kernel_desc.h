/* -*- c++ -*- */
/*
 * Copyright 2026 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#ifndef INCLUDED_VOLK_KERNEL_DESC_H
#define INCLUDED_VOLK_KERNEL_DESC_H

#include <stdbool.h>
#include <math.h> /* INFINITY, NAN */

/*!
 * \brief Maximum number of arguments a kernel can have (excluding num_points).
 *
 * Sized to accommodate the largest existing kernel signatures.
 */
#define VOLK_MAX_ARGS 10

/*!
 * \brief Sentinel value for volk_kernel_desc_t.state_arg when kernel is stateless.
 */
#define VOLK_NO_STATE (-1)

/*!
 * \brief Type of valid-value region for a kernel argument.
 *
 * Used in volk_domain_t to describe the set of valid (or expected) values
 * for a given argument. For input arguments this constrains test data
 * generation and documents caller requirements. For output arguments
 * it describes the expected range of results.
 */
typedef enum {
    /*!< No restriction on values. */
    VOLK_DOMAIN_ANY = 0,

    /*!< Real interval: bounds[0] <= x <= bounds[1].
     *   Use -INFINITY / INFINITY for half-open intervals.
     *   Examples: [-1, 1] for acos input, [0, INFINITY) for sqrt input. */
    VOLK_DOMAIN_SEGMENT,

    /*!< Complex disk: |z| <= bounds[0].
     *   bounds[1] is unused.
     *   Example: bounds[0] = 1.0 for the unit disk. */
    VOLK_DOMAIN_DISK,

    /*!< Complex annulus: bounds[0] <= |z| <= bounds[1].
     *   Set bounds[0] == bounds[1] for a circle.
     *   Example: {1.0, 1.0} for the unit circle. */
    VOLK_DOMAIN_ANNULUS,

    /*!< Complex half-plane through the origin.
     *   The valid region satisfies: bounds[0]*Re(z) + bounds[1]*Im(z) >= 0,
     *   i.e. (bounds[0], bounds[1]) is the inward-pointing normal.
     *   Common cases:
     *     {1, 0}  right half-plane  (Re >= 0)
     *     {-1, 0} left half-plane   (Re <= 0)
     *     {0, 1}  upper half-plane  (Im >= 0)
     *     {0, -1} lower half-plane  (Im <= 0) */
    VOLK_DOMAIN_HALF_PLANE,
} volk_domain_type_t;

/*!
 * \brief Describes the valid-value region for a single kernel argument.
 *
 * Interpretation of bounds depends on the domain type.
 * A zero-initialized volk_domain_t is equivalent to VOLK_DOMAIN_ANY.
 */
typedef struct volk_domain {
    volk_domain_type_t type;
    float bounds[2];
} volk_domain_t;

/*!
 * \brief Kernel operation type — how output count relates to input count.
 */
typedef enum {
    VOLK_OP_MAP = 0, /*!< N elements in -> N elements out */
    VOLK_OP_REDUCE,  /*!< N elements in -> scalar out     */
} volk_op_t;

/*!
 * \brief Kernel descriptor — static metadata embedded in each kernel header.
 *
 * This struct lives in the kernel source file alongside the proto-kernel
 * implementations. It is compiled under VOLK_KERNEL_DESC and is also
 * parsed by the code generator at build time.
 *
 * Domains are indexed by argument position in the kernel function signature
 * (excluding the trailing num_points / count parameter). For output arguments,
 * the domain describes the expected output range. For input arguments, it
 * constrains valid input values. Unused entries default to VOLK_DOMAIN_ANY
 * through zero-initialization.
 *
 * Example — volk_32f_acos_32f(float* out, const float* in, unsigned int N):
 * \code
 * #ifdef VOLK_KERNEL_DESC
 * static const volk_kernel_desc_t volk_32f_acos_32f_desc = {
 *     .op_type = VOLK_OP_MAP,
 *     .in_place = true,
 *     .domains = {
 *         [0] = { VOLK_DOMAIN_SEGMENT, { 0.0f, M_PI } },
 *         [1] = { VOLK_DOMAIN_SEGMENT, { -1.0f, 1.0f } },
 *     },
 *     .tolerance = 1e-5f,
 *     .absolute_tolerance = true,
 * };
 * #endif
 * \endcode
 */
typedef struct volk_kernel_desc {
    /*! Operation type: map (N->N) or reduce (N->scalar). */
    volk_op_t op_type;

    /*! Optimal processing granularity in elements.
     *  Consumers that feed multiples of this value avoid tail-loop overhead.
     *  0 means the kernel handles any count efficiently. */
    unsigned int granularity;

    /*! Element stride. 0 means elements are contiguous in memory. */
    unsigned int stride;

    /*! True if the output buffer may alias the first input buffer. */
    bool in_place;

    /*! Per-argument domain.
     *  Indexed by argument position in the function signature, excluding
     *  the trailing count parameter. Zero-initialized entries (index beyond
     *  the actual argument count, or explicitly zeroed) are VOLK_DOMAIN_ANY. */
    volk_domain_t domains[VOLK_MAX_ARGS];

    /*! Index of the state argument, or VOLK_NO_STATE (-1) if the kernel
     *  is stateless. A state argument is a non-const pointer that persists
     *  and is mutated across calls (e.g. the phase accumulator in rotator). */
    int state_arg;

    /*! Comparison tolerance for correctness testing. 0 means use the
     *  test harness default. */
    float tolerance;

    /*! If true, tolerance is absolute. If false, tolerance is relative. */
    bool absolute_tolerance;

    /*! Name of the replacement kernel if this one is deprecated, or NULL. */
    const char* deprecated_by;

    /*! Optional JSON string for additional metadata not covered by the
     *  struct fields. NULL when unused. */
    const char* extra;
} volk_kernel_desc_t;

#endif /* INCLUDED_VOLK_KERNEL_DESC_H */
