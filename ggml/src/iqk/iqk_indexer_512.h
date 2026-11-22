// INDEXER_TOPK with AVX-512, for index keys in f16 as DeepSeek-V4 and V4.1 keep them.
//
// For every token, score(row) = mask(row) + sum over the heads h of w[h] * relu(K[row] . q[h]), and the rows
// (or groups of rows) with the highest scores are given. See iqk_indexer_512.cpp for the way it is computed.

#pragma once

#include "iqk_config.h"
#include "iqk_mul_mat.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// true when this node can be computed here; iqk_indexer_topk() does it the other way when not.
// GGML_INDEXER_512=0 in the environment makes this false for every node.
bool   iqk_indexer512_usable(const struct ggml_tensor * dst);
size_t iqk_indexer512_work_size(const struct ggml_tensor * dst, int nth);
void   iqk_indexer512(struct ggml_tensor * dst, void * work_buffer, barrier_t barrier, void * barrier_data, int ith, int nth);

#ifdef __cplusplus
}
#endif
