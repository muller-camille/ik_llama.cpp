// The threads take rows of a quantized matrix product from each other when they are done with their own
// (iqk_mul_mat_share_run). Whoever computes a row, its value is the same: products that share one input and
// are computed in one go, large and small ones, must give the values of one thread for every thread count,
// every time. For one token that is bit for bit; for several tokens the kernel of a group of rows can differ.
//
// The same holds for the up and gate matrices of a feed-forward computed together, and for the experts of a
// few tokens, where a thread that finds no chunk left takes the end of the chunks the others are still on
// (iqk_share_take, iqk_share_steal): second part. There the values are compared, bit for bit and for any number
// of tokens, with the ones the same number of threads gives when every thread keeps to its own rows: the
// activation does not round alike for the last rows of a range, so one thread is not the reference.

#include "ggml.h"
#include "../ggml/src/iqk/iqk_mul_mat.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
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

    // The fused up and gate product of a feed-forward (the shared expert of DeepSeek-V4), and experts: 4 of 12
    // for each token, up and gate fused, then down. 2048 rows of 1024 values per expert, as rows of 8. The
    // experts twice, as MXFP4_R8 and as Q8_0_R8: for 2 to 7 tokens the Q8_0_R8 kernel takes another way for a call
    // that is not a multiple of 16 rows, and with 3, 7 or 9 threads a chunk of an expert is not one.
    int n_fail2 = 0, n_runs2 = 0;
    {
        const int n_embd = 1024, n_ff = 2048, n_expert = 12, n_used = 4, n_ff_dense = 4096;
        ggml_init_params ip = { (size_t) 1024 << 20, nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        auto expert_weights = [&] (ggml_type type, int n_in, int n_out) {
            ggml_tensor * w = ggml_new_tensor_3d(ctx, type, n_in, n_out, n_expert);
            std::vector<float> f((size_t) n_in*n_out*n_expert);
            for (auto & v : f) v = 0.05f*nd(rng);
            ggml_quantize_chunk(type, f.data(), w->data, 0, (int64_t) n_out*n_expert, n_in, nullptr, nullptr);
            return w;
        };
        ggml_tensor * up   = expert_weights(GGML_TYPE_MXFP4_R8, n_embd, n_ff);
        ggml_tensor * gate = expert_weights(GGML_TYPE_MXFP4_R8, n_embd, n_ff);
        ggml_tensor * down = expert_weights(GGML_TYPE_MXFP4_R8, n_ff, n_embd);
        ggml_tensor * up_q8   = expert_weights(GGML_TYPE_Q8_0_R8, n_embd, n_ff);
        ggml_tensor * gate_q8 = expert_weights(GGML_TYPE_Q8_0_R8, n_embd, n_ff);
        ggml_tensor * down_q8 = expert_weights(GGML_TYPE_Q8_0_R8, n_ff, n_embd);
        // 2 x 4096 rows of 2048 values of Q8_0_R8 are 18 MB: shared up to 64 threads
        ggml_tensor * up_d   = weights(ctx, GGML_TYPE_Q8_0_R8, 2*n_embd, n_ff_dense);
        ggml_tensor * gate_d = weights(ctx, GGML_TYPE_Q8_0_R8, 2*n_embd, n_ff_dense);
        for (int n_tok : { 1, 2, 3, 5, 8, 9 }) {
            ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tok);
            ggml_tensor * x_d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2*n_embd, n_tok);
            ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
            for (int i = 0; i < n_embd*n_tok; ++i)   ((float *) x->data)[i]   = nd(rng);
            for (int i = 0; i < 2*n_embd*n_tok; ++i) ((float *) x_d->data)[i] = nd(rng);
            for (int t = 0; t < n_tok; ++t) {
                // n_used experts that differ, some of them the ones of the token before
                std::vector<int> order(n_expert);
                for (int e = 0; e < n_expert; ++e) order[e] = e;
                for (int e = 0; e < n_used; ++e) {
                    std::swap(order[e], order[e + rng()%(n_expert - e)]);
                    ((int32_t *) ids->data)[t*n_used + e] = order[e];
                }
            }
            auto run = [&] (int n_threads) {
                ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_tensor * h = ggml_moe_up_gate(ctx, up, gate, x, ids, GGML_UNARY_OP_SILU);
                ggml_tensor * y = ggml_mul_mat_id(ctx, down, h, ids);
                ggml_tensor * z = ggml_fused_up_gate(ctx, up_d, gate_d, x_d, GGML_UNARY_OP_SILU);
                ggml_tensor * h_q8 = ggml_moe_up_gate(ctx, up_q8, gate_q8, x, ids, GGML_UNARY_OP_SILU);
                ggml_tensor * y_q8 = ggml_mul_mat_id(ctx, down_q8, h_q8, ids);
                ggml_build_forward_expand(gf, y);
                ggml_build_forward_expand(gf, z);
                ggml_build_forward_expand(gf, y_q8);
                ggml_graph_compute_with_ctx(ctx, gf, n_threads);
                std::vector<float> out;
                for (auto * t : { h, y, z, h_q8, y_q8 }) {
                    const float * p = (const float *) t->data;
                    out.insert(out.end(), p, p + ggml_nelements(t));
                }
                return out;
            };
            for (int n_threads : { 2, 3, 4, 5, 6, 7, 8, 9, 16, 32, 64 }) {
                iqk_share_enable(0);
                const std::vector<float> ref = run(n_threads);
                iqk_share_enable(1);
                double scale = 0;
                bool finite = true;
                for (float v : ref) {
                    finite = finite && std::isfinite(v);
                    scale = std::max(scale, (double) std::fabs(v));
                }
                if (!finite || scale == 0) {
                    fprintf(stderr, "FAILED: experts, %d token(s), %d threads: the result without sharing is %s\n",
                            n_tok, n_threads, finite ? "all zero" : "not finite");
                    n_fail2++;
                }
                for (int rep = 0; rep < 4; ++rep) {
                    const std::vector<float> out = run(n_threads);
                    ++n_runs2;
                    if (out.size() != ref.size() || memcmp(out.data(), ref.data(), ref.size()*sizeof(float)) != 0) {
                        double diff = 0;
                        size_t n_diff = 0, first = ref.size(), last = 0;
                        for (size_t i = 0; i < ref.size() && i < out.size(); ++i) {
                            diff = std::max(diff, (double) std::fabs(out[i] - ref[i]));
                            if (out[i] != ref[i]) {
                                ++n_diff;
                                first = std::min(first, i);
                                last = i;
                            }
                        }
                        // where: the fused experts (n_ff rows for each of n_used experts and each token), the down
                        // matrices (n_embd rows each), the fused up and gate (n_ff_dense rows for each token)
                        const size_t n_h = (size_t) n_ff*n_used*n_tok, n_y = (size_t) n_embd*n_used*n_tok, n_z = (size_t) n_ff_dense*n_tok;
                        auto where = [&] (size_t i) {
                            char buf[96];
                            if (i < n_h)            { snprintf(buf, sizeof(buf), "fused experts, row %zu", i%n_ff); return std::string(buf); }
                            i -= n_h;
                            if (i < n_y)            { snprintf(buf, sizeof(buf), "down, row %zu", i%n_embd); return std::string(buf); }
                            i -= n_y;
                            if (i < n_z)            { snprintf(buf, sizeof(buf), "fused up and gate, row %zu", i%n_ff_dense); return std::string(buf); }
                            i -= n_z;
                            if (i < n_h)            { snprintf(buf, sizeof(buf), "fused Q8_0_R8 experts, row %zu", i%n_ff); return std::string(buf); }
                            snprintf(buf, sizeof(buf), "Q8_0_R8 down, row %zu", (i - n_h)%n_embd);
                            return std::string(buf);
                        };
                        fprintf(stderr, "FAILED: experts, %d token(s), %d threads, run %d: %zu values differ from the ones without sharing, by up to %.2e of %.2e, from %s to %s\n",
                                n_tok, n_threads, rep, n_diff, diff, scale, where(first).c_str(), where(last).c_str());
                        n_fail2++;
                    }
                }
            }
            iqk_share_enable(-1);
        }
        ggml_free(ctx);
    }
    printf("%d of %d runs of the fused and expert products differ from the result without sharing\n", n_fail2, n_runs2);
    // a comparison that no thread ever took a row for would say nothing
    const long n_stolen = iqk_share_stolen();
    printf("%ld units of 16 rows were computed by another thread than the one that had them\n", n_stolen);
    if (n_stolen == 0) {
        fprintf(stderr, "FAILED: no thread took rows from another one\n");
        n_fail2++;
    }
    return n_fail || n_fail2 ? 1 : 0;
}
