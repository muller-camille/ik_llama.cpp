// flash_attn_ext must not depend on the number of threads: every thread count has to give the result of one
// thread, and one thread the result of the generic implementation. Covers head sizes with and without an IQK
// kernel, GQA and MLA head layouts, and cache lengths that are not a multiple of 32.

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int n_fail = 0;
static bool verbose = false; // name every run before it starts: shows where a hang is

static std::mt19937 rng(21);

static ggml_tensor * new_cache(ggml_context * ctx, ggml_type type, int D, int n_kv, int n_head_kv) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> f((size_t) D*n_kv*n_head_kv);
    for (auto & v : f) {
        v = nd(rng);
    }
    ggml_tensor * t = ggml_new_tensor_3d(ctx, type, D, n_kv, n_head_kv);
    if (type == GGML_TYPE_F16) {
        for (size_t i = 0; i < f.size(); ++i) {
            ((ggml_fp16_t *) t->data)[i] = ggml_fp32_to_fp16(f[i]);
        }
    } else {
        ggml_quantize_chunk(type, f.data(), t->data, 0, (int64_t) n_kv*n_head_kv, D, nullptr, nullptr);
    }
    return t;
}

static void test_shape(ggml_type type_kv, int D, int n_head, int n_head_kv, int n_q, int n_kv) {
    ggml_init_params ip = { (size_t) 1 << 30, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    std::normal_distribution<float> nd(0.0f, 1.0f);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, n_q, n_head);
    for (int64_t i = 0; i < ggml_nelements(q); ++i) {
        ((float *) q->data)[i] = nd(rng);
    }
    ggml_tensor * k = new_cache(ctx, type_kv, D, n_kv, n_head_kv);
    ggml_tensor * v = new_cache(ctx, type_kv, D, n_kv, n_head_kv);

    // causal mask over a cache that is three quarters full
    const int n_used = std::max(n_q, 3*n_kv/4);
    const int n_mask = GGML_PAD(n_q, GGML_KQ_MASK_PAD);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_mask);
    for (int iq = 0; iq < n_mask; ++iq) {
        for (int ik = 0; ik < n_kv; ++ik) {
            const bool visible = iq < n_q && ik < n_used - (n_q - 1 - iq);
            ((ggml_fp16_t *) mask->data)[(size_t) iq*n_kv + ik] = ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
        }
    }

    auto run = [&](int n_threads, bool generic) {
        ggml_tensor * y = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f/std::sqrt((float) D), 0.0f, 0.0f);
        if (generic) {
            y->op_params[4] = GGML_FLASH_ATTN_EXT_IQK_DISABLED;
        }
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_graph_compute_with_ctx(ctx, gf, n_threads);
        const float * p = (const float *) y->data;
        return std::vector<float>(p, p + ggml_nelements(y));
    };

    auto rel_err = [](const std::vector<float> & y, const std::vector<float> & ref) {
        double num = 0, den = 0;
        for (size_t i = 0; i < y.size(); ++i) {
            num += (double) (y[i] - ref[i])*(y[i] - ref[i]);
            den += (double) ref[i]*ref[i];
        }
        return std::sqrt(num/den);
    };

    const std::vector<float> ref = run(1, false);
    std::string bad;
    // one thread against the generic implementation: the two differ in precision only (the generic one
    // accumulates in f16). It has no dot product for a quantized cache, so f16 caches only.
    if (type_kv == GGML_TYPE_F16) {
        const double err = rel_err(ref, run(1, true));
        if (!(err < 2e-2)) {
            char buf[64];
            snprintf(buf, sizeof(buf), " [1 thread against generic: %.1e]", err);
            bad += buf;
        }
    }
    for (int n_threads : { 2, 3, 4, 5, 6, 7, 8, 12, 16 }) {
        if (verbose) {
            fprintf(stderr, "%s head size %d, %d heads, %d kv heads, %d query row(s), cache %d, %d threads\n",
                    ggml_type_name(type_kv), D, n_head, n_head_kv, n_q, n_kv, n_threads);
        }
        const double err = rel_err(run(n_threads, false), ref);
        if (!(err < 1e-3)) {
            char buf[64];
            snprintf(buf, sizeof(buf), " [%d threads: %.1e]", n_threads, err);
            bad += buf;
        }
    }
    if (!bad.empty()) {
        fprintf(stderr, "FAILED: %s head size %3d, %2d heads, %2d kv heads, %d query row(s), cache %4d:%s\n",
                ggml_type_name(type_kv), D, n_head, n_head_kv, n_q, n_kv, bad.c_str());
        n_fail++;
    }
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    verbose = argc > 1 && strcmp(argv[1], "-v") == 0;
    struct shape { ggml_type type; int D, n_head, n_head_kv; };
    const shape shapes[] = {
        { GGML_TYPE_F16,   48,  6,  6 }, // no IQK kernel for this head size
        { GGML_TYPE_F16,   80, 10,  2 }, // no IQK kernel, GQA
        { GGML_TYPE_F16,   64,  8,  8 },
        { GGML_TYPE_F16,   64, 12,  4 }, // GQA
        { GGML_TYPE_F16,  128,  6,  2 }, // GQA
        { GGML_TYPE_F16,  512, 64,  1 }, // MLA
        { GGML_TYPE_Q8_0,  64,  8,  8 },
        { GGML_TYPE_Q8_0,  96,  6,  6 },
    };
    int n_tests = 0;
    for (const auto & s : shapes) {
        for (int n_q : { 1, 5 }) {
            for (int n_kv : { 32, 40, 72, 256, 1024 }) {
                if (s.type != GGML_TYPE_F16 && n_kv % 32 != 0) {
                    continue; // only the IQK kernels read a quantized cache, and they take whole blocks of 32 rows
                }
                test_shape(s.type, s.D, s.n_head, s.n_head_kv, n_q, n_kv);
                ++n_tests;
            }
        }
    }
    printf("%d of %d cases failed\n", n_fail, n_tests);
    return n_fail ? 1 : 0;
}
