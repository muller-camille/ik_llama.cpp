// The small operations that were rewritten for speed give the values of their plain definitions, for one
// thread and for several: rms_norm (sum of squares on several chains), hc_post (vectorized per token), hc_proj
// (norm, projection and collapse in one node), rope (cos/sin tables kept between nodes) and argmax (AVX512).

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

static int n_fail = 0;
static std::mt19937 rng(57);

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "%s:%d: FAILED: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

static ggml_tensor * rnd(ggml_context * ctx, int n0, int n1 = 1, int n2 = 1, float scale = 1.0f) {
    std::normal_distribution<float> nd(0.0f, scale);
    ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n0, n1, n2);
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
        ((float *) t->data)[i] = nd(rng);
    }
    return t;
}

static std::vector<float> compute(ggml_context * ctx, ggml_tensor * y, int n_threads) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    const float * p = (const float *) y->data;
    return std::vector<float>(p, p + ggml_nelements(y));
}

// largest difference, relative to the largest reference value
static double max_err(const std::vector<float> & y, const std::vector<double> & ref) {
    double err = 0, scale = 1e-30;
    for (size_t i = 0; i < ref.size(); ++i) {
        err   = std::max(err, std::fabs(y[i] - ref[i]));
        scale = std::max(scale, std::fabs(ref[i]));
    }
    return y.size() == ref.size() ? err/scale : 1.0;
}

static const int k_threads[] = { 1, 2, 3, 4, 8 };

static void test_rms_norm(int n, int n_rows) {
    ggml_init_params ip = { (size_t) 64 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    const float eps = 1e-6f;
    ggml_tensor * x = rnd(ctx, n, n_rows, 1, 3.0f);
    ggml_tensor * w = rnd(ctx, n);
    std::vector<double> ref((size_t) n*n_rows), ref_w(ref.size());
    for (int r = 0; r < n_rows; ++r) {
        const float * xr = (const float *) x->data + (size_t) r*n;
        double sum = 0;
        for (int i = 0; i < n; ++i) sum += (double) xr[i]*xr[i];
        const double scale = 1.0/std::sqrt(sum/n + eps);
        for (int i = 0; i < n; ++i) {
            ref  [(size_t) r*n + i] = xr[i]*scale;
            ref_w[(size_t) r*n + i] = xr[i]*scale*((const float *) w->data)[i];
        }
    }
    for (int n_threads : k_threads) {
        const double e1 = max_err(compute(ctx, ggml_rms_norm(ctx, x, eps), n_threads), ref);
        const double e2 = max_err(compute(ctx, ggml_fused_rms_norm(ctx, x, w, eps), n_threads), ref_w);
        if (!(e1 < 1e-5 && e2 < 1e-5)) {
            fprintf(stderr, "FAILED: rms_norm %d x %d, %d threads: %.1e, fused %.1e\n", n, n_rows, n_threads, e1, e2);
            n_fail++;
        }
    }
    ggml_free(ctx);
}

static void test_hc_post(int ne0, int S, int T) {
    ggml_init_params ip = { (size_t) 64 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x    = rnd(ctx, ne0, T);
    ggml_tensor * post = rnd(ctx, S, T);
    ggml_tensor * res  = rnd(ctx, ne0, S, T);
    ggml_tensor * comb = rnd(ctx, S, S, T);
    std::vector<double> ref((size_t) ne0*S*T);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < S; ++i) {
            for (int i0 = 0; i0 < ne0; ++i0) {
                double sum = (double) ((const float *) x->data)[(size_t) t*ne0 + i0] * ((const float *) post->data)[t*S + i];
                for (int j = 0; j < S; ++j) {
                    sum += (double) ((const float *) comb->data)[(size_t) t*S*S + j*S + i] *
                                    ((const float *) res->data)[((size_t) t*S + j)*ne0 + i0];
                }
                ref[((size_t) t*S + i)*ne0 + i0] = sum;
            }
        }
    }
    for (int n_threads : k_threads) {
        const double e = max_err(compute(ctx, ggml_hc_post(ctx, x, post, res, comb), n_threads), ref);
        if (!(e < 1e-5)) {
            fprintf(stderr, "FAILED: hc_post %d x %d streams x %d tokens, %d threads: %.1e\n", ne0, S, T, n_threads, e);
            n_fail++;
        }
    }
    ggml_free(ctx);
}

// hc_proj against rms_norm, then the product with fn, then the weighted sum of the streams
static void test_hc_proj(int ne0, int S, int M, int T, bool with_pre) {
    ggml_init_params ip = { (size_t) 64 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    const float eps = 1e-6f;
    ggml_tensor * x   = rnd(ctx, ne0, S, T, 3.0f);
    ggml_tensor * fn  = rnd(ctx, ne0*S, M);
    ggml_tensor * pre = with_pre ? rnd(ctx, S, T) : nullptr;
    const float * xd = (const float *) x->data;
    std::vector<double> ref_mix((size_t) M*T), ref_sum((size_t) ne0*T);
    for (int t = 0; t < T; ++t) {
        const float * xt = xd + (size_t) t*ne0*S;
        double sq = 0;
        for (int i = 0; i < ne0*S; ++i) sq += (double) xt[i]*xt[i];
        const double scale = 1.0/std::sqrt(sq/(ne0*S) + eps);
        for (int m = 0; m < M; ++m) {
            const float * w = (const float *) fn->data + (size_t) m*ne0*S;
            double sum = 0;
            for (int i = 0; i < ne0*S; ++i) sum += (double) w[i]*xt[i];
            ref_mix[(size_t) t*M + m] = sum*scale;
        }
        for (int i = 0; with_pre && i < ne0; ++i) {
            double sum = 0;
            for (int s = 0; s < S; ++s) sum += (double) ((const float *) pre->data)[t*S + s]*xt[(size_t) s*ne0 + i];
            ref_sum[(size_t) t*ne0 + i] = sum;
        }
    }
    for (int n_threads : k_threads) {
        ggml_tensor * y = ggml_hc_proj(ctx, fn, x, pre, eps);
        const std::vector<float> all = compute(ctx, y, n_threads);
        const double e1 = max_err(std::vector<float>(all.begin(), all.begin() + M*T), ref_mix);
        double e2 = 0;
        if (with_pre) {
            const size_t off = ggml_hc_proj_sum_offset(M, T);
            e2 = all.size() == off + (size_t) ne0*T ? max_err(std::vector<float>(all.begin() + off, all.end()), ref_sum) : 1.0;
        } else if (all.size() != (size_t) M*T) {
            e2 = 1.0;
        }
        if (!(e1 < 1e-5 && e2 < 1e-5)) {
            fprintf(stderr, "FAILED: hc_proj %d x %d streams -> %d, %d tokens%s, %d threads: %.1e, sum %.1e\n",
                    ne0, S, M, T, with_pre ? ", pre" : "", n_threads, e1, e2);
            n_fail++;
        }
    }
    ggml_free(ctx);
}

// The nodes the DeepSeek-V4 graph builds around hc_proj against the ones it built before: rms_norm, mul_mat and
// mul_multi_add. What hc_pre makes of the mixes and the collapsed streams have to agree.
static void test_hc_chain(int ne0, int S, int T) {
    ggml_init_params ip = { (size_t) 256 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    const float eps = 1e-6f;
    const int M = S*S + 2*S;
    ggml_tensor * x     = rnd(ctx, ne0, S, T, 2.0f);
    ggml_tensor * fn    = rnd(ctx, ne0*S, M, 1, 0.02f);
    ggml_tensor * pre   = rnd(ctx, S, T);
    ggml_tensor * scale = rnd(ctx, 3);
    ggml_tensor * bias  = rnd(ctx, M);
    for (int n_threads : k_threads) {
        // before
        ggml_tensor * flat   = ggml_reshape_2d(ctx, x, ne0*S, T);
        ggml_tensor * mixes0 = ggml_mul_mat(ctx, fn, ggml_rms_norm(ctx, flat, eps));
        ggml_tensor * all0   = ggml_hc_pre(ctx, mixes0, scale, bias, S, 20, 1e-6f, 0.0f);
        ggml_tensor * sum0   = ggml_mul_multi_add(ctx, x, ggml_reshape_4d(ctx, pre, 1, S, T, 1));
        // now
        ggml_tensor * proj   = ggml_hc_proj(ctx, fn, x, pre, eps);
        ggml_tensor * mixes1 = ggml_view_2d(ctx, proj, M,   T, M*sizeof(float),   0);
        ggml_tensor * sum1   = ggml_view_2d(ctx, proj, ne0, T, ne0*sizeof(float), ggml_hc_proj_sum_offset(M, T)*sizeof(float));
        ggml_tensor * all1   = ggml_hc_pre(ctx, mixes1, scale, bias, S, 20, 1e-6f, 0.0f);
        ggml_tensor * sum1c  = ggml_cont(ctx, sum1);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        for (ggml_tensor * t : { all0, sum0, all1, sum1c }) {
            ggml_build_forward_expand(gf, t);
        }
        ggml_graph_compute_with_ctx(ctx, gf, n_threads);

        auto values = [] (const ggml_tensor * t) {
            const float * p = (const float *) t->data;
            return std::vector<float>(p, p + ggml_nelements(t));
        };
        auto as_ref = [&values] (const ggml_tensor * t) {
            const std::vector<float> v = values(t);
            return std::vector<double>(v.begin(), v.end());
        };
        const double e1 = max_err(values(all1), as_ref(all0));
        const double e2 = max_err(values(sum1c), as_ref(sum0));
        if (!(e1 < 1e-4 && e2 < 1e-6)) {
            fprintf(stderr, "FAILED: hc chain %d x %d streams, %d tokens, %d threads: hc_pre %.1e, sum %.1e\n",
                    ne0, S, T, n_threads, e1, e2);
            n_fail++;
        }
    }
    ggml_free(ctx);
}

// plain rope on the last n_dims of each row when flipped, on the first ones otherwise; back = inverse rotation
static void test_rope(int ne0, int n_dims, int n_head, int T, bool back) {
    ggml_init_params ip = { (size_t) 64 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    const float freq_base = 10000.0f;
    ggml_tensor * x   = rnd(ctx, ne0, n_head, T);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    // the positions of three graphs: the tables of a thread are reused, replaced and come back
    const int first_pos[] = { 5, 1000, 5 };
    for (int round = 0; round < 3; ++round) {
        for (int t = 0; t < T; ++t) {
            ((int32_t *) pos->data)[t] = first_pos[round] + 3*t;
        }
        const int offset = back ? ne0 - n_dims : 0;
        std::vector<double> ref((size_t) ne0*n_head*T);
        const float theta_scale = powf(freq_base, -2.0f/n_dims);
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < n_head; ++h) {
                const float * xr = (const float *) x->data + ((size_t) t*n_head + h)*ne0;
                double * yr = ref.data() + ((size_t) t*n_head + h)*ne0;
                for (int i = 0; i < ne0; ++i) yr[i] = xr[i];
                float theta = (float) ((int32_t *) pos->data)[t];
                for (int i = 0; i < n_dims; i += 2) {
                    const float c = cosf(theta), s = (back ? -1.0f : 1.0f)*sinf(theta);
                    yr[offset + i]     = (double) xr[offset + i]*c - (double) xr[offset + i + 1]*s;
                    yr[offset + i + 1] = (double) xr[offset + i]*s + (double) xr[offset + i + 1]*c;
                    theta *= theta_scale;
                }
            }
        }
        for (int n_threads : k_threads) {
            // two rope nodes in one graph, as two layers have
            ggml_tensor * y1 = ggml_rope_ext(ctx, x, pos, nullptr, n_dims, 0, 0, freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            ggml_tensor * y2 = ggml_rope_ext(ctx, x, pos, nullptr, n_dims, 0, 0, freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            if (back) {
                y1->op = y2->op = GGML_OP_ROPE_BACK;
                y1->op_params[15] = y2->op_params[15] = 1;
            }
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, y1);
            ggml_build_forward_expand(gf, y2);
            ggml_graph_compute_with_ctx(ctx, gf, n_threads);
            for (ggml_tensor * y : { y1, y2 }) {
                const float * p = (const float *) y->data;
                const double e = max_err(std::vector<float>(p, p + ggml_nelements(y)), ref);
                if (!(e < 1e-5)) {
                    fprintf(stderr, "FAILED: rope%s %d of %d dims x %d heads x %d tokens, round %d, %d threads: %.1e\n",
                            back ? "_back" : "", n_dims, ne0, n_head, T, round, n_threads, e);
                    n_fail++;
                }
            }
        }
    }
    ggml_free(ctx);
}

// what ggml computed before: the last position of the largest value, and its own answer when there is a NaN
static int argmax_ref(const std::vector<float> & x) {
    float max = -INFINITY;
    int idx = 0;
    for (int i = 0; i < (int) x.size(); ++i) {
        max = max > x[i] ? max : x[i];
        if (max == x[i]) idx = i;
    }
    return idx;
}

static void test_argmax() {
    ggml_init_params ip = { (size_t) 64 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    int n_cases = 0;
    for (int n : { 1, 10, 63, 64, 100, 1000, 4097, 129280 }) {
        for (int kind = 0; kind < 8; ++kind) {
            std::vector<float> x(n);
            for (auto & v : x) v = nd(rng);
            std::uniform_int_distribution<int> at(0, n - 1);
            switch (kind) {
                case 0: break;
                case 1: { const float m = 9.0f; x[at(rng)] = m; x[at(rng)] = m; x[at(rng)] = m; } break; // ties
                case 2: x[0] = 9.0f; break;
                case 3: x[n - 1] = 9.0f; break;
                case 4: for (auto & v : x) v = -INFINITY; break;
                case 5: x[at(rng)] = nan; break;
                case 6: x[n - 1] = nan; x[at(rng)] = 9.0f; break;
                case 7: for (auto & v : x) v = 2.5f; break; // all equal
            }
            ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            memcpy(t->data, x.data(), n*sizeof(float));
            ggml_tensor * y = ggml_argmax(ctx, t);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, y);
            ggml_graph_compute_with_ctx(ctx, gf, 4);
            const int got = ((const int32_t *) y->data)[0], want = argmax_ref(x);
            ++n_cases;
            if (got != want) {
                fprintf(stderr, "FAILED: argmax of %d values, case %d: %d, expected %d\n", n, kind, got, want);
                n_fail++;
            }
        }
    }
    printf("argmax: %d cases\n", n_cases);
    ggml_free(ctx);
}

int main() {
    for (int n : { 37, 64, 5120, 20480 }) {
        for (int n_rows : { 1, 3, 7 }) {
            test_rms_norm(n, n_rows);
        }
    }
    for (int ne0 : { 64, 100, 5120 }) {
        for (int T : { 1, 2, 3, 7 }) {
            test_hc_post(ne0, 4, T);
        }
    }
    test_hc_post(1000, 8, 2);
    for (bool with_pre : { false, true }) {
        for (int T : { 1, 2, 5, 9 }) {
            test_hc_proj(5120, 4, 24, T, with_pre); // DeepSeek-V4
            test_hc_proj(100,  4, 24, T, with_pre); // elements that do not fill the last cache line
        }
        test_hc_proj(16, 3, 5, 3, with_pre);        // fewer cache lines than threads
        test_hc_proj(7,  8, 64, 1, with_pre);
    }
    for (int T : { 1, 5, 32 }) {
        test_hc_chain(5120, 4, T);
    }
    for (bool back : { false, true }) {
        test_rope(512,  64, 64, 1, back); // DeepSeek-V4 query: many rows, one position
        test_rope(512,  64,  1, 1, back); // one row: thread 0 alone
        test_rope(512,  64,  8, 5, back); // more positions than a thread keeps tables for
        test_rope(128, 128,  4, 2, back);
    }
    test_argmax();

    printf(n_fail ? "%d check(s) FAILED\n" : "OK\n", n_fail);
    return n_fail ? 1 : 0;
}
