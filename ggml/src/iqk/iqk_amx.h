//
// Matrix products of a batch of tokens with the tiles of Intel AMX (Sapphire Rapids and later)
//
// The activations and the weights go through the tiles as bf16: TDPBF16PS multiplies 16 x 32 values by
// 32 x 16 values in one instruction, some 6 times what the AVX-512 kernels of the quantized types reach when
// the weights of a matrix serve many tokens. For Q8_0 weights nothing is lost on the way: the product of an 8-bit
// quant and its block scale is rounded to 8 significant bits, less than the half step Q8_0 already rounds each
// weight by, and the activations keep 8 significant bits each where the usual path gives a block of 32 one
// 8-bit scale.
//
// GGML_AMX=0 in the environment turns it off.
//
// The attention of a batch of tokens can go the same way (GGML_AMX_ATTN=1, see iqk_amx_flash_attn below).
//

#pragma once

#include "iqk_config.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 1 if the tile products can be used: the processor has AMX-BF16 and AVX512-BF16, the system saves the tiles
// and gave this process the permission to use them, and GGML_AMX is not 0
IQK_API int iqk_amx_bf16(void);

// 1 if the product of ny columns (tokens) by a matrix of this type, nx rows of ne00 values, goes through the
// tiles. The caller then hands the activations over as GGML_TYPE_BF16.
IQK_API int iqk_amx_wanted(int typeA, long nx, long ny, long ne00);

// n activations as bf16, for a product that goes through the tiles (only where iqk_amx_bf16() is 1)
IQK_API void iqk_amx_from_float(const float * x, void * y, int64_t n);

// for the tests: 0 = not from now on, 1 = again (where iqk_amx_bf16() was 1 at the start)
IQK_API void iqk_amx_enable(int on);

// 1 if the attention of a batch of tokens goes through the tiles (iqk_amx_flash_attn below): iqk_amx_bf16() is 1
// and GGML_AMX_ATTN=1 is in the environment. It is off unless asked for.
IQK_API int iqk_amx_attn(void);

// for the tests: -1 = as the environment says, 0 = not from now on, 1 = from now on (where iqk_amx_bf16() is 1)
IQK_API void iqk_amx_attn_enable(int on);

// The attention of one token through the tiles: its nq query rows (heads, D floats each, stride_q bytes apart) on
// the rows of its list. K and V are the same rows of D f16 values: row idx[j] of the cache is at
// kv + idx[j]*stride_kv, and idx[j] < 0 names no row. mask: one f16 per row of the cache, a row is seen where it is
// 0; nullptr if all the rows of the list are seen. sinks: one value per query row, or nullptr.
// out: nq rows of D floats, stride_out bytes apart. No soft cap of the scores.
// The scores are scale * q.k with q and k rounded to bf16, the weights exp(score - max) are rounded to bf16 and
// multiply V rounded to bf16; the sums are in fp32.
// Only where iqk_amx_attn_fits() is 1; nothing is written if the list names no row, as without the tiles.
IQK_API void iqk_amx_flash_attn(int D, int nq, int n, const float * q, long stride_q, const void * kv, long stride_kv,
                                const int * idx, const uint16_t * mask, const float * sinks, float scale,
                                float * out, long stride_out);

// 1 if iqk_amx_flash_attn can compute this attention: K and V are the same f16 rows of D values, D a multiple of 32
IQK_API int iqk_amx_attn_fits(int type_k, int type_v, int same_rows, int Dk, int Dv, float softcap);

#ifdef __cplusplus
}

// C[y*stride_C + x] = sum_j A[x][j] * B[y][j], the share of thread ith of nth. B is Ny rows of ne00 bf16 values,
// strideB bytes apart. False if this is not a product for the tiles (then nothing was computed).
bool iqk_amx_mul_mat(long Nx, long Ny, long ne00, int typeA, const void * A, long strideA,
                     const void * B, long strideB, float * C, long stride_C, int ith, int nth);
#endif
