// Small nodes are computed by thread 0 alone, and the other threads skip them without a barrier: they only
// wait, before the next node they take part in, until thread 0 got there (ggml_node_is_serial).
// A graph that mixes such nodes with multi-threaded ones, with serial runs at the start, in the middle and
// at the end, must give the result of one thread for every thread count, every time.

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static std::mt19937 rng(31);

static ggml_tensor * rnd(ggml_context * ctx, int n0, int n1, float scale, float offset = 0.0f) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    ggml_tensor * t = n1 > 0 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1) : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n0);
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
        ((float *) t->data)[i] = offset + scale*nd(rng);
    }
    return t;
}

int main() {
    ggml_init_params ip = { (size_t) 512 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    const int n = 512, n_big = 1024, n_cols = 3;

    // inputs and weights
    ggml_tensor * x   = rnd(ctx, n, 1, 1.0f);
    ggml_tensor * w1  = rnd(ctx, n, n, 0.05f);
    ggml_tensor * b1  = rnd(ctx, n, 1, 0.5f);
    ggml_tensor * g   = rnd(ctx, n, 1, 0.2f, 1.0f);
    ggml_tensor * w2  = rnd(ctx, 2*n, 256, 0.05f);
    ggml_tensor * tab = rnd(ctx, 256, 100, 1.0f);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_cols);
    for (int i = 0; i < n_cols; ++i) {
        ((int32_t *) idx->data)[i] = 7 + 31*i;
    }
    ggml_tensor * w3  = rnd(ctx, 256, 128, 0.05f);
    ggml_tensor * big = rnd(ctx, n_big, 128, 1.0f);   // too large for a serial node: stays multi-threaded
    ggml_tensor * big_b = rnd(ctx, n_big, 128, 1.0f);
    ggml_tensor * w4  = rnd(ctx, n_big, 64, 0.05f);

    auto build = [&]() {
        // serial run at the start of the graph
        ggml_tensor * h = ggml_add(ctx, x, x);
        h = ggml_mul(ctx, h, g);
        // multi-threaded
        h = ggml_mul_mat(ctx, w1, h);
        // serial run in the middle
        ggml_tensor * h2 = ggml_add(ctx, h, b1);
        ggml_tensor * h3 = ggml_rms_norm(ctx, h2, 1e-6f);
        ggml_tensor * h4 = ggml_mul(ctx, h3, g);
        ggml_tensor * h5 = ggml_concat(ctx, h4, h2, 0);
        ggml_tensor * h6 = ggml_sqrt(ctx, ggml_clamp(ctx, ggml_mul(ctx, h5, h5), 0.01f, 10.0f));
        // multi-threaded
        ggml_tensor * h7 = ggml_mul_mat(ctx, w2, h6);                 // [256, 1]
        // serial: gather, repeat, add
        ggml_tensor * rows = ggml_get_rows(ctx, tab, idx);            // [256, n_cols]
        ggml_tensor * h8 = ggml_add(ctx, ggml_repeat(ctx, h7, rows), rows);
        // two multi-threaded nodes in a row, the second one a large add
        ggml_tensor * o1 = ggml_mul_mat(ctx, w3, h8);                 // [128, n_cols]
        ggml_tensor * o2 = ggml_mul_mat(ctx, w4, ggml_add(ctx, big, big_b)); // [64, 128]
        // serial run at the end of the graph
        o1 = ggml_add(ctx, o1, o1);
        o1 = ggml_rms_norm(ctx, o1, 1e-6f);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, o1);
        ggml_build_forward_expand(gf, o2);
        return std::make_pair(gf, std::make_pair(o1, o2));
    };

    auto run = [&](int n_threads) {
        auto b = build();
        ggml_graph_compute_with_ctx(ctx, b.first, n_threads);
        std::vector<float> r;
        for (ggml_tensor * t : { b.second.first, b.second.second }) {
            const float * p = (const float *) t->data;
            r.insert(r.end(), p, p + ggml_nelements(t));
        }
        return r;
    };

    const std::vector<float> ref = run(1);
    int n_fail = 0, n_runs = 0;
    for (int n_threads : { 2, 3, 4, 5, 8, 16 }) {
        for (int rep = 0; rep < 40; ++rep) {
            const std::vector<float> r = run(n_threads);
            double num = 0, den = 0;
            for (size_t i = 0; i < r.size(); ++i) {
                num += (double) (r[i] - ref[i])*(r[i] - ref[i]);
                den += (double) ref[i]*ref[i];
            }
            ++n_runs;
            if (!(std::sqrt(num/den) < 1e-5)) {
                if (n_fail++ < 5) {
                    fprintf(stderr, "FAILED: %d threads, run %d: relative difference %.2e\n", n_threads, rep, std::sqrt(num/den));
                }
            }
        }
    }
    printf("%d of %d runs differ from the single-thread result\n", n_fail, n_runs);
    ggml_free(ctx);
    return n_fail ? 1 : 0;
}
