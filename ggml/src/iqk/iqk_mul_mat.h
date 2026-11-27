//
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "iqk_config.h"
#ifdef __cplusplus
extern "C" {
#endif

IQK_API bool iqk_mul_mat(long Nx, long Ny, long ne00,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long stride_C, int ith, int nth);

IQK_API bool iqk_mul_mat_4d(long Nx, long Ny, long ne00,
        long ne02, long ne03, long ne12, long ne13,
        long nb02, long nb03, long nb12, long nb13, long nb2, long nb3,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long stride_C, int ith, int nth);

// A matrix product whose rows the threads take from each other when they are done with their own, so that
// none waits for the slowest. A slot is IQK_SHARE_SLOT_SIZE bytes; a matrix has one per thread, slots[ith].
//  - iqk_mul_mat_share_init(): every thread, for its own slot, before the threads synchronise. It returns
//    false, for every thread alike, if the product is not computed this way.
//  - iqk_mul_mat_share_run(): after that. With steal = 0 a thread computes the rows of its slot, with
//    steal = 1 rows that the others have not started on. A is the thread's own copy of the weights.
#define IQK_SHARE_SLOT_SIZE 64
IQK_API bool iqk_mul_mat_share_init(void * slots, long Nx, long Ny, long ne00, int typeA, long strideA, int typeB, int ith, int nth);
IQK_API bool iqk_mul_mat_share_run(void * slots, int steal, long Nx, long Ny, long ne00,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long stride_C, int ith, int nth);

IQK_API bool iqk_mul_mat_moe(long Nx, long Ny, long ne00, int ne11,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long nb1, long nb2, const void * vrow_mapping, int ith, int nth);

IQK_API bool iqk_moe_fused_up_gate(long Nx, long Ny, long ne00, int ne11, int unary_op,
        int typeA, const void * Aup, const void * Agate, long strideA,
        int typeB, const void * B, long strideB,
        const char * up_b, const char * gate_b,
        float * C, long nb1, long nb2, const void * vrow_mapping, float limit,
        float situ_beta, float situ_beta_linear, int ith, int nth);

// The same sharing where the caller hands out the rows: the up and gate matrices of a feed-forward, computed
// together, and the experts of a token. The slots are the ones above, one per thread.
//  - iqk_share_unit_rows(): the rows the kernel takes at a time for the product of a matrix of Nx rows with Ny
//    tokens (the rows of a thread are made of those), or 0 if that product does not go straight to the kernel
//    of its type: it is then not computed this way.
//  - iqk_share_open(): the thread says which units it is on, [first, end) of run id (below 2^20 - 1; a unit
//    count is below 2^22). What a unit is, is the caller's business. iqk_share_close(): none, or with pending
//    set none yet: a thread closes its slot as pending before it asks for more work, so that the others wait
//    for what it will open.
//  - iqk_share_take(): up to `want` units from the front of the thread's own slot; their number, 0: none left.
//  - iqk_share_steal(): up to `want` units from the end of the slot of thread `from`, and the run they are of;
//    their number, 0: none left, -1: that thread has not said yet what it is on.
//  - iqk_moe_fused_up_gate_rows(), iqk_mul_mat_moe_rows(): rows [first_row, first_row + n_rows) of the product
//    that iqk_moe_fused_up_gate() and iqk_mul_mat_moe() compute; n_rows and first_row are multiples of the unit.
//    The thread reads on up to run_end_row (the weights are asked for ahead up to there); goes_on: the call
//    starts where the last one of this thread stopped.
//  - iqk_share_enable(): 1 or 0 turns all the sharing of rows on or off whatever GGML_SHARE_ROWS says, -1
//    gives the say back to it. For the tests; iqk_share_forced() is what was last asked.
//    iqk_share_stolen(): the units that threads have taken from the slot of another one since the start.
IQK_API void iqk_share_enable(int on);
IQK_API int  iqk_share_forced(void);
IQK_API long iqk_share_stolen(void);
IQK_API int  iqk_share_unit_rows(long Nx, long Ny, long ne00, int typeA, int typeB);
IQK_API void iqk_share_open(void * slots, int ith, uint32_t id, uint32_t first, uint32_t end);
IQK_API void iqk_share_close(void * slots, int ith, int pending);
IQK_API int  iqk_share_take(void * slots, int ith, int want, uint32_t * first);
IQK_API int  iqk_share_steal(void * slots, int from, int want, uint32_t * id, uint32_t * first);
IQK_API bool iqk_moe_fused_up_gate_rows(long first_row, long n_rows, long run_end_row, int goes_on,
        long Ny, long ne00, int ne11, int unary_op,
        int typeA, const void * Aup, const void * Agate, long strideA,
        int typeB, const void * B, long strideB,
        const char * up_b, const char * gate_b,
        float * C, long nb1, long nb2, const void * vrow_mapping, float limit,
        float situ_beta, float situ_beta_linear);
IQK_API bool iqk_mul_mat_moe_rows(long first_row, long n_rows, long run_end_row, int goes_on,
        long Ny, long ne00, int ne11,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long nb1, long nb2, const void * vrow_mapping);

IQK_API int iqk_dequant_type(int type, int Ny);

IQK_API void iqk_vec_dot_f8_e4m3_ks(int n, float * s, size_t bs, const void * x, size_t bx,
        const void * y, size_t by, int nrc);
IQK_API void iqk_vec_dot_f8_e4m3_b128(int n, float * s, size_t bs, const void * x, size_t bx,
        const void * y, size_t by, int nrc);


struct ggml_tensor;

IQK_API size_t iqk_fa_work_buffer_size(const struct ggml_tensor * dst, int nthread);

// adds to the report of GGML_PROFILE how the attention nodes with one K/V head were computed
IQK_API void iqk_fa_report(void);

typedef void (*barrier_t) (void *);

// K, and V which is the same, in two parts that were not put together: the rows 0 .. n0-1 at k0, the others at k1.
struct iqk_fa_two_parts {
    const void * k0;
    long         stride0;
    int          n0;
    const void * k1;
    long         stride1;
};

IQK_API bool iqk_flash_attn_noalibi(int type_q, int type_mask, float max_bias,
                            int neq3, int neq2, long nbq3, long nbq2,
                            int nek3, int nek2, long nbk3, long nbk2,
                            int nev3, int nev2, long nbv3, long nbv2,
                            int ne2,  int ne1,  long nb1,
                            int type_k,             // type of k
                            int type_v,             // type of v
                            int Dk,                 // K head size
                            int Dv,                 // V head size
                            int nq,                 // number of columns in q
                            int nk,                 // number of rows in k
                            int stride_q,           // distance between q columns in bytes
                            int stride_k,           // distance between k rows in bytes
                            int stride_v,           // distance between v rows in bytes
                            int stride_m,           // distance between mask rows (in bytes
                            const void  * q,        // q matrix.
                            const void  * k,        // k matrix. Assumed to be fp16, nq x nk elements
                            const void  * v,        // v matrix. Assumed to be fp16, nq x nk elements
                            const void  * mask,     // mask. If not null, assumed to be fp16. nq x nk elements
                            const void  * sinks,    // mask. If not null, assumed to be fp16. nq x nk elements
                            float         scale,    // scale applied before softmax
                            float         softcap,  // if > 0, a "soft-cap" operation is applied before softmax
                            float       * qkv,      // v*softmax(scale*(k*q))
                            void * work_buffer, barrier_t barrier, void * barrier_data,
                            int ith, int nth, int n_swa, struct ggml_tensor * indexer,
                            // not NULL: k and v do not hold the rows, they are in these two parts. Returns false
                            // when the rows cannot be read there; the caller then puts them together at k
                            const struct iqk_fa_two_parts * two_parts);

IQK_API void iqk_topk_moe(int n_experts, int n_experts_used, int nrows, const float * logits,
        float * weights, int32_t * ids, int ith, int nth);

IQK_API bool iqk_fused_delta_net(int head_dim, int n_heads, int gqa_ratio, int repeat_type, int n_tokens, int n_seqs,
        size_t vnb1, size_t vnb2, size_t vnb3,
        const float * q_data, const float * k_data, const float * v_data, const float * g_data, const float * beta_data,
        const float * state_in, float * out_data, float * state_out, float * saved_steps, int state_step_stride, int ith, int nth);

IQK_API bool iqk_indexer_topk(struct ggml_tensor * dst, void * work_buffer, barrier_t barrier, void * barrier_data, int ith, int nth);
IQK_API size_t iqk_idx_topk_work_buffer_size(const struct ggml_tensor * dst, int nthread);

#ifdef __cplusplus
}
#endif
