// NUMA mirror: the matrix multiplications must read copy ith % n_copies of mirrored weight memory.
// The copies hold different weights here, so every output element shows which copy produced it.

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "%s:%d: FAILED: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

static std::mt19937 rng(11);

static ggml_tensor * new_weight(ggml_context * ctx, ggml_type type, int n_in, int n_out, int n_expert) {
    std::normal_distribution<float> nd(0.0f, 0.05f);
    std::vector<float> f((size_t) n_in*n_out*n_expert);
    for (auto & v : f) {
        v = nd(rng);
    }
    ggml_tensor * w = n_expert > 1 ? ggml_new_tensor_3d(ctx, type, n_in, n_out, n_expert) : ggml_new_tensor_2d(ctx, type, n_in, n_out);
    if (type == GGML_TYPE_F32) {
        memcpy(w->data, f.data(), f.size()*sizeof(float));
    } else {
        ggml_quantize_chunk(type, f.data(), w->data, 0, (int64_t) n_out*n_expert, n_in, nullptr, nullptr);
    }
    return w;
}

static std::vector<float> compute(ggml_context * ctx, ggml_tensor * y, int n_threads) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    const float * p = (const float *) y->data;
    return std::vector<float>(p, p + ggml_nelements(y));
}

enum op_kind { MUL_MAT, MUL_MAT_ID, MOE_UP_GATE, FUSED_UP_GATE };

static const char * op_name(op_kind op) {
    switch (op) {
        case MUL_MAT:       return "mul_mat";
        case MUL_MAT_ID:    return "mul_mat_id";
        case MOE_UP_GATE:   return "moe_up_gate";
        case FUSED_UP_GATE: return "fused_up_gate";
    }
    return "?";
}

static void test_op(op_kind op, ggml_type type, int n_tokens) {
    const int n_in = 256, n_out = 512, n_used = 4, n_threads = 4;
    const bool moe     = op == MUL_MAT_ID || op == MOE_UP_GATE;
    const bool up_gate = op == MOE_UP_GATE || op == FUSED_UP_GATE;
    const int n_expert = moe ? 8 : 1;

    ggml_init_params ip = { (size_t) 512 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    // w[0] is the tensor the graph uses, w[1] holds other weights of the same shape; w2 is the second matrix of up/gate
    ggml_tensor * w[2], * w2[2] = { nullptr, nullptr };
    for (int c = 0; c < 2; ++c) {
        w[c] = new_weight(ctx, type, n_in, n_out, n_expert);
        if (up_gate) {
            w2[c] = new_weight(ctx, type, n_in, n_out, n_expert);
        }
    }

    std::normal_distribution<float> nd(0.0f, 1.0f);
    ggml_tensor * x = moe ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_in, 1, n_tokens) : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, n_tokens);
    for (int64_t i = 0; i < ggml_nelements(x); ++i) {
        ((float *) x->data)[i] = nd(rng);
    }
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        for (int i = 0; i < n_used; ++i) {
            ((int32_t *) ids->data)[t*n_used + i] = (3*t + i) % n_expert; // distinct experts per token
        }
    }

    auto build = [&](int c) {
        switch (op) {
            case MUL_MAT:       return ggml_mul_mat(ctx, w[c], x);
            case MUL_MAT_ID:    return ggml_mul_mat_id(ctx, w[c], x, ids);
            case MOE_UP_GATE:   return ggml_moe_up_gate(ctx, w[c], w2[c], x, ids, GGML_UNARY_OP_SILU);
            case FUSED_UP_GATE: return ggml_fused_up_gate(ctx, w[c], w2[c], x, GGML_UNARY_OP_SILU);
        }
        return (ggml_tensor *) nullptr;
    };

    const std::vector<float> y0 = compute(ctx, build(0), n_threads); // every thread reads w[0]
    const std::vector<float> y1 = compute(ctx, build(1), n_threads); // every thread reads w[1]

    CHECK(ggml_numa_mirror_add(w[0]->data, ggml_nbytes(w[0]), w[1]->data));
    if (up_gate) {
        CHECK(ggml_numa_mirror_add(w2[0]->data, ggml_nbytes(w2[0]), w2[1]->data));
    }

    // one thread is thread 0: copy 0 only
    const std::vector<float> ys = compute(ctx, build(0), 1);
    const std::vector<float> y0s = [&] {
        ggml_numa_mirror_remove(w[0]->data);
        if (up_gate) ggml_numa_mirror_remove(w2[0]->data);
        auto r = compute(ctx, build(0), 1);
        CHECK(ggml_numa_mirror_add(w[0]->data, ggml_nbytes(w[0]), w[1]->data));
        if (up_gate) CHECK(ggml_numa_mirror_add(w2[0]->data, ggml_nbytes(w2[0]), w2[1]->data));
        return r;
    }();
    CHECK(ys == y0s);

    // several threads: each element comes from the copy of the thread that computed it, and both copies are used.
    // The expert operations hand out work dynamically, so give the other threads a few runs to take part.
    size_t n0 = 0, n1 = 0, n_other = 0;
    for (int attempt = 0; attempt < 50 && (n0 == 0 || n1 == 0) && n_other == 0; ++attempt) {
        const std::vector<float> ym = compute(ctx, build(0), n_threads);
        n0 = n1 = n_other = 0;
        for (size_t i = 0; i < ym.size(); ++i) {
            const bool is0 = std::fabs(ym[i] - y0[i]) <= 1e-5f*std::fabs(y0[i]) + 1e-7f;
            const bool is1 = std::fabs(ym[i] - y1[i]) <= 1e-5f*std::fabs(y1[i]) + 1e-7f;
            n0      += is0;
            n1      += is1 && !is0;
            n_other += !is0 && !is1;
        }
    }
    printf("%-13s %-6s %2d token(s): %zu elements from copy 0, %zu from copy 1, %zu from neither\n",
            op_name(op), ggml_type_name(type), n_tokens, n0, n1, n_other);
    CHECK(n_other == 0);
    CHECK(n0 > 0);
    CHECK(n1 > 0);

    // without the copies the tensor is read as before
    ggml_numa_mirror_remove(w[0]->data);
    if (up_gate) {
        ggml_numa_mirror_remove(w2[0]->data);
    }
    CHECK(compute(ctx, build(0), n_threads) == y0 || op == MUL_MAT_ID || op == MOE_UP_GATE); // dynamic chunks may differ in the last bits
    const std::vector<float> yr = compute(ctx, build(0), n_threads);
    size_t n_bad = 0;
    for (size_t i = 0; i < yr.size(); ++i) {
        n_bad += !(std::fabs(yr[i] - y0[i]) <= 1e-5f*std::fabs(y0[i]) + 1e-7f);
    }
    CHECK(n_bad == 0);

    ggml_free(ctx);
}

int main() {
    for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_MXFP4 }) {
        for (int n_tokens : { 1, 5 }) {
            test_op(MUL_MAT, type, n_tokens);
            test_op(MUL_MAT_ID, type, n_tokens);
            if (type != GGML_TYPE_F32) {
                test_op(MOE_UP_GATE, type, n_tokens);
                test_op(FUSED_UP_GATE, type, n_tokens);
            }
        }
    }

    // a range takes GGML_NUMA_MAX_NODES copies at most, and its size has to match
    static char base[64], copy[64];
    CHECK(ggml_numa_mirror_add(base, sizeof(base), copy));
    CHECK(!ggml_numa_mirror_add(base, sizeof(base)/2, copy));
    int n_copies = 2;
    while (ggml_numa_mirror_add(base, sizeof(base), copy)) {
        ++n_copies;
    }
    CHECK(n_copies == 8);
    ggml_numa_mirror_remove(base);

    printf(n_fail ? "%d check(s) FAILED\n" : "OK\n", n_fail);
    return n_fail ? 1 : 0;
}
