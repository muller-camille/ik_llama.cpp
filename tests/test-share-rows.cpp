// The threads take rows of a quantized matrix product from each other when they are done with their own
// (iqk_mul_mat_share_run). Whoever computes a row, its value is the same: products that share one input and
// are computed in one go, large and small ones, must give the values of one thread for every thread count,
// every time. For one token that is bit for bit; for several tokens the kernel of a group of rows can differ.

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static std::mt19937 rng(43);
static std::normal_distribution<float> nd(0.0f, 1.0f);

static ggml_tensor * weights(ggml_context * ctx, ggml_type type, int n_in, int n_out) {
    ggml_tensor * w = ggml_new_tensor_2d(ctx, type, n_in, n_out);
    std::vector<float> f((size_t) n_in*n_out);
    for (auto & v : f) v = 0.05f*nd(rng);
    if (type == GGML_TYPE_F32) {
        memcpy(w->data, f.data(), f.size()*sizeof(float));
    } else {
        ggml_quantize_chunk(type, f.data(), w->data, 0, n_out, n_in, nullptr, nullptr);
    }
    return w;
}

int main() {
    const int n_in = 2048; // 8192 rows of Q8_0_R8 are 17 MB: shared up to 64 threads
    // one input for all: the quantized products that follow each other are computed in one go
    struct Mat { ggml_type type; int n_out; };
    const std::vector<Mat> mats = {
        { GGML_TYPE_Q8_0_R8, 4096 }, { GGML_TYPE_Q8_0_R8, 2304 }, { GGML_TYPE_Q8_0_R8, 48 }, // 48 rows: too small to share
        { GGML_TYPE_Q8_0_R8, 1280 }, { GGML_TYPE_F32, 64 },                                   // another input type
        { GGML_TYPE_Q4_K_R4, 2048 }, { GGML_TYPE_Q6_K_R4, 1024 }, { GGML_TYPE_Q8_0_R8, 8192 },
    };
    int n_fail = 0, n_runs = 0;
    for (int n_tok : { 1, 2, 5, 8, 9 }) {
        ggml_init_params ip = { (size_t) 512 << 20, nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        std::vector<ggml_tensor *> w;
        for (const auto & m : mats) {
            w.push_back(weights(ctx, m.type, n_in, m.n_out));
        }
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, n_tok);
        for (int i = 0; i < n_in*n_tok; ++i) {
            ((float *) x->data)[i] = nd(rng);
        }
        auto run = [&] (int n_threads) {
            ggml_cgraph * gf = ggml_new_graph(ctx);
            std::vector<ggml_tensor *> y;
            for (auto * wi : w) {
                y.push_back(ggml_mul_mat(ctx, wi, x));
                ggml_build_forward_expand(gf, y.back());
            }
            ggml_graph_compute_with_ctx(ctx, gf, n_threads);
            std::vector<float> out;
            for (auto * yi : y) {
                const float * p = (const float *) yi->data;
                out.insert(out.end(), p, p + ggml_nelements(yi));
            }
            return out;
        };
        const std::vector<float> ref = run(1);
        double scale = 0;
        for (float v : ref) {
            if (!std::isfinite(v)) { fprintf(stderr, "FAILED: %d token(s): not finite\n", n_tok); n_fail++; break; }
            scale = std::max(scale, (double) std::fabs(v));
        }
        for (int n_threads : { 2, 3, 4, 7, 8, 16, 32, 64 }) {
            for (int rep = 0; rep < 3; ++rep) {
                const std::vector<float> out = run(n_threads);
                ++n_runs;
                double diff = 0;
                for (size_t i = 0; i < ref.size(); ++i) {
                    diff = std::max(diff, (double) std::fabs(out[i] - ref[i]));
                }
                const bool ok = n_tok == 1 ? memcmp(out.data(), ref.data(), ref.size()*sizeof(float)) == 0 : diff < 2e-5*scale;
                if (!ok) {
                    fprintf(stderr, "FAILED: %d token(s), %d threads, run %d: differs from one thread by %.2e of %.2e\n",
                            n_tok, n_threads, rep, diff, scale);
                    n_fail++;
                }
            }
        }
        ggml_free(ctx);
    }
    printf("%d of %d runs differ from the single-thread result\n", n_fail, n_runs);
    return n_fail ? 1 : 0;
}
