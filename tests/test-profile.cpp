// GGML_PROFILE=1: the graphs give the same results as without it, and the table printed at exit adds up the
// layers: CTest checks its text (see CMakeLists.txt), this program checks the results.

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main() {
#ifdef _WIN32
    _putenv_s("GGML_PROFILE", "1");
#else
    setenv("GGML_PROFILE", "1", 1);
#endif

    ggml_init_params ip = { (size_t) 256 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    std::mt19937 rng(41);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto rnd = [&](const char * name, int n0, int n1) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1);
        for (int64_t i = 0; i < ggml_nelements(t); ++i) {
            ((float *) t->data)[i] = 0.05f*nd(rng);
        }
        ggml_set_name(t, name);
        return t;
    };

    const int n = 384;
    ggml_tensor * x  = rnd("x", n, 1);
    ggml_tensor * w3 = rnd("blk.3.w.weight", n, n);
    ggml_tensor * w7 = rnd("blk.7.w.weight", n, n);
    ggml_tensor * b  = rnd("b", n, 1);

    // what the graph computes, by hand
    std::vector<float> ref(n), tmp(n);
    for (int i = 0; i < n; ++i) {
        double s = 0;
        for (int j = 0; j < n; ++j) s += (double) ((float *) w3->data)[(size_t) i*n + j] * ((float *) x->data)[j];
        tmp[i] = (float) s + ((float *) b->data)[i];
    }
    for (int i = 0; i < n; ++i) {
        double s = 0;
        for (int j = 0; j < n; ++j) s += (double) ((float *) w7->data)[(size_t) i*n + j] * tmp[j];
        ref[i] = (float) s;
    }

    int n_fail = 0;
    auto run = [&](int n_threads) {
        ggml_tensor * h = ggml_mul_mat(ctx, w3, x);
        h = ggml_add(ctx, h, b);
        ggml_set_name(h, "sum-3");
        ggml_tensor * y = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, w7, n, n), h); // weights behind a view
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_graph_compute_with_ctx(ctx, gf, n_threads);
        double num = 0, den = 0;
        for (int i = 0; i < n; ++i) {
            const double d = ((float *) y->data)[i] - ref[i];
            num += d*d;
            den += (double) ref[i]*ref[i];
        }
        if (!(std::sqrt(num/den) < 1e-4)) {
            fprintf(stderr, "FAILED: %d threads: relative difference %.2e\n", n_threads, std::sqrt(num/den));
            n_fail++;
        }
    };
    // the first graph of each kind is not timed: 5 graphs with 4 threads, 2 with 1 thread
    for (int i = 0; i < 6; ++i) run(4);
    for (int i = 0; i < 3; ++i) run(1);

    ggml_free(ctx);
    fprintf(stderr, "%d run(s) FAILED\n", n_fail); // stderr, as the table: it comes first
    return n_fail ? 1 : 0;
}
