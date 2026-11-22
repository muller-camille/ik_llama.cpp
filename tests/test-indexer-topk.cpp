// INDEXER_TOPK gives, for every token of a batch, the rows of the index keys with the highest scores:
//   score(row) = mask(row) + sum over the heads h of w[h] * relu(K[row] . q[h])
// The rows it gives are compared with scores written out here in double precision, for batches with at least
// as many tokens as threads (each thread takes whole tokens) and with fewer (the threads share the rows of a
// token), for the three kinds of masks, and for the selection by groups of rows that DeepSeek-V4.1 does in one
// layer. A row may replace another one only where their scores are equal within the rounding.
//
//   test-indexer-topk --bench [rows [tokens [threads [repeats]]]]   times the operation at the sizes of
//                                                                   DeepSeek-V4.1 Flash (32 heads of 128)

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

static int n_fail = 0;

static std::mt19937 rng(41);

static bool equal_keys = false; // every key row is the same: the scores of the rows in view are equal

enum mask_kind {
    MASK_VISIBLE, // I32: the number of rows each token sees
    MASK_F32,     // 0 or -inf for every row
    MASK_F16,
};

struct indexer_case {
    int       d;
    int       n_head;
    int       n_kv;
    int       n_tok;
    int       n_top;        // rows to give, or groups with a group size
    mask_kind mask;
    int       group_size;   // > 0: the groups with the best row are chosen, then the rows inside them
    int       n_top_out;    // rows to give after the groups were chosen
    int       n_threads;
};

struct indexer_data {
    ggml_context * ctx = nullptr;
    ggml_tensor  * k = nullptr, * q = nullptr, * w = nullptr, * m = nullptr, * top_out = nullptr, * ids = nullptr;
    std::vector<float> kf;       // the keys as the operation reads them (after the rounding to f16)
    std::vector<int>   visible;  // per token
    std::vector<char>  hidden;   // per token and row, for the masks that are not a count
};

static indexer_data build(const indexer_case & c) {
    indexer_data s;
    ggml_init_params ip = { (size_t) 256 << 20, nullptr, false };
    ip.mem_size += (size_t) c.d*c.n_kv*2 + (size_t) c.d*c.n_head*c.n_tok*4 + (size_t) c.n_kv*c.n_tok*4;
    s.ctx = ggml_init(ip);

    std::normal_distribution<float> nd(0.0f, 1.0f);
    s.k = ggml_new_tensor_2d(s.ctx, GGML_TYPE_F16, c.d, c.n_kv);
    s.kf.resize((size_t) c.d*c.n_kv);
    for (size_t i = 0; i < s.kf.size(); ++i) {
        const ggml_fp16_t h = equal_keys && i >= (size_t) c.d ? ((ggml_fp16_t *) s.k->data)[i - c.d] : ggml_fp32_to_fp16(nd(rng));
        ((ggml_fp16_t *) s.k->data)[i] = h;
        s.kf[i] = ggml_fp16_to_fp32(h);
    }
    s.q = ggml_new_tensor_3d(s.ctx, GGML_TYPE_F32, c.d, c.n_head, c.n_tok);
    for (int64_t i = 0; i < ggml_nelements(s.q); ++i) {
        ((float *) s.q->data)[i] = nd(rng);
    }
    s.w = ggml_new_tensor_2d(s.ctx, GGML_TYPE_F32, c.n_head, c.n_tok);
    for (int64_t i = 0; i < ggml_nelements(s.w); ++i) {
        ((float *) s.w->data)[i] = nd(rng) / std::sqrt((float) c.d*c.n_head);
    }

    // every token sees a few rows more than the one before, as in a prompt; the rows after them are hidden
    s.visible.resize(c.n_tok);
    const int first = std::max(1, c.n_kv - c.n_kv/4 - c.n_tok/4);
    for (int it = 0; it < c.n_tok; ++it) {
        s.visible[it] = std::min(c.n_kv, first + it/4);
    }
    if (c.mask == MASK_VISIBLE) {
        s.m = ggml_new_tensor_2d(s.ctx, GGML_TYPE_I32, 1, c.n_tok);
        for (int it = 0; it < c.n_tok; ++it) {
            ((int32_t *) s.m->data)[it] = s.visible[it];
        }
    } else {
        // and some rows before them too, as when a layer keeps the rows of the chosen groups only
        s.hidden.assign((size_t) c.n_kv*c.n_tok, 0);
        std::uniform_int_distribution<int> one_in(0, 6);
        s.m = ggml_new_tensor_2d(s.ctx, c.mask == MASK_F32 ? GGML_TYPE_F32 : GGML_TYPE_F16, c.n_kv, c.n_tok);
        const ggml_fp16_t ninf16 = ggml_fp32_to_fp16(-INFINITY);
        for (int it = 0; it < c.n_tok; ++it) {
            for (int j = 0; j < c.n_kv; ++j) {
                const bool hide = j >= s.visible[it] || one_in(rng) == 0;
                s.hidden[(size_t) it*c.n_kv + j] = hide;
                if (c.mask == MASK_F32) {
                    ((float *) s.m->data)[(size_t) it*c.n_kv + j] = hide ? -INFINITY : 0.0f;
                } else {
                    ((ggml_fp16_t *) s.m->data)[(size_t) it*c.n_kv + j] = hide ? ninf16 : 0;
                }
            }
        }
    }

    if (c.group_size > 0 && c.n_top_out > 0) {
        s.top_out = ggml_new_tensor_2d(s.ctx, GGML_TYPE_I32, c.n_top_out, c.n_tok);
    }
    s.ids = ggml_indexer_topk(s.ctx, s.k, s.q, s.w, s.m, nullptr, GGML_UNARY_OP_RELU, c.n_top, c.group_size,
            c.group_size > 0, s.top_out);
    return s;
}

static void compute(const indexer_data & s, int n_threads) {
    ggml_cgraph * gf = ggml_new_graph(s.ctx);
    ggml_build_forward_expand(gf, s.ids);
    ggml_graph_compute_with_ctx(s.ctx, gf, n_threads);
}

static std::string name_of(const indexer_case & c) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%d heads of %d, %d rows, %d tokens, top %d, %s mask, groups of %d (top %d rows), %d threads",
            c.n_head, c.d, c.n_kv, c.n_tok, c.n_top,
            c.mask == MASK_VISIBLE ? "count" : c.mask == MASK_F32 ? "f32" : "f16", c.group_size, c.n_top_out, c.n_threads);
    return buf;
}

// `got` holds the n best of `score`: no row twice, and no row left out that scores more than one that is in
static bool is_top(const int32_t * got, int n, const std::vector<double> & score, double eps, std::string & why) {
    const int n_all = (int) score.size();
    std::vector<char> in(n_all, 0);
    double worst_in = INFINITY;
    for (int i = 0; i < n; ++i) {
        const int j = got[i];
        if (j < 0 || j >= n_all) {
            why = "row " + std::to_string(j) + " does not exist";
            return false;
        }
        if (in[j]) {
            why = "row " + std::to_string(j) + " is given twice";
            return false;
        }
        in[j] = 1;
        worst_in = std::min(worst_in, score[j]);
    }
    for (int j = 0; j < n_all; ++j) {
        if (!in[j] && score[j] > worst_in + eps) {
            char buf[160];
            snprintf(buf, sizeof(buf), "row %d (score %g) is left out, a row with score %g is in", j, score[j], worst_in);
            why = buf;
            return false;
        }
    }
    return true;
}

static void test_case(const indexer_case & c) {
    indexer_data s = build(c);
    compute(s, c.n_threads);

    const double eps = 1e-4;
    std::vector<double> score(c.n_kv);
    bool ok = true;
    std::string why;
    for (int it = 0; it < c.n_tok && ok; ++it) {
        const float * q = (const float *) s.q->data + (size_t) it*c.n_head*c.d;
        const float * w = (const float *) s.w->data + (size_t) it*c.n_head;
        for (int j = 0; j < c.n_kv; ++j) {
            const bool hide = c.mask == MASK_VISIBLE ? j >= s.visible[it] : s.hidden[(size_t) it*c.n_kv + j] != 0;
            if (hide) {
                score[j] = -INFINITY;
                continue;
            }
            const float * kr = s.kf.data() + (size_t) j*c.d;
            double sum = 0;
            for (int h = 0; h < c.n_head; ++h) {
                double dot = 0;
                for (int i = 0; i < c.d; ++i) {
                    dot += (double) kr[i] * q[(size_t) h*c.d + i];
                }
                sum += w[h] * std::max(dot, 0.0);
            }
            score[j] = sum;
        }
        const int32_t * got = (const int32_t *) s.ids->data + (size_t) it*c.n_top;
        if (c.group_size <= 0) {
            ok = is_top(got, c.n_top, score, eps, why);
            continue;
        }
        // the score of a group is the one of its best row; the last group with a row in view is always taken
        const int n_b = c.n_kv/c.group_size;
        std::vector<double> group(n_b);
        for (int b = 0; b < n_b; ++b) {
            group[b] = *std::max_element(score.begin() + (size_t) b*c.group_size, score.begin() + (size_t) (b + 1)*c.group_size);
        }
        for (int b = n_b - 1; b >= 0; --b) {
            if (group[b] > -INFINITY) {
                group[b] = INFINITY;
                break;
            }
        }
        ok = is_top(got, c.n_top, group, eps, why);
        if (ok && s.top_out) {
            // then the best rows of the groups that were taken
            std::vector<char> kept(n_b, 0);
            for (int i = 0; i < c.n_top; ++i) {
                kept[got[i]] = 1;
            }
            for (int j = 0; j < c.n_kv; ++j) {
                if (!kept[j/c.group_size]) {
                    score[j] = -INFINITY;
                }
            }
            ok = is_top((const int32_t *) s.top_out->data + (size_t) it*c.n_top_out, c.n_top_out, score, eps, why);
            if (!ok) {
                why = "rows of the groups: " + why;
            }
        }
        if (!ok) {
            why = "token " + std::to_string(it) + ": " + why;
        }
    }
    if (!ok) {
        fprintf(stderr, "FAILED: %s\n    %s\n", name_of(c).c_str(), why.c_str());
        n_fail++;
    }
    ggml_free(s.ctx);
}

static int bench(int argc, char ** argv) {
    indexer_case c = { 128, 32, 5120, 512, 512, MASK_VISIBLE, 0, 0, 4 };
    int n_rep = 3;
    if (argc > 2) c.n_kv      = atoi(argv[2]);
    if (argc > 3) c.n_tok     = atoi(argv[3]);
    if (argc > 4) c.n_threads = atoi(argv[4]);
    if (argc > 5) n_rep       = atoi(argv[5]);
    for (const int group_size : { 0, 8 }) {
        c.group_size = group_size;
        c.n_top      = group_size > 0 ? std::min(2048, c.n_kv/group_size) : 512;
        c.n_top_out  = group_size > 0 ? 512 : 0;
        indexer_data s = build(c);
        double best = INFINITY;
        for (int r = 0; r < n_rep; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            compute(s, c.n_threads);
            best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        printf("%s\n    %.2f ms, %.1f us of a thread per token\n", name_of(c).c_str(), best, 1e3*best*c.n_threads/c.n_tok);
        ggml_free(s.ctx);
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        return bench(argc, argv);
    }

    int n_cases = 0;
    for (const int n_threads : { 1, 3, 4, 8 }) {
        for (const mask_kind mask : { MASK_VISIBLE, MASK_F32, MASK_F16 }) {
            // tokens >= threads: whole tokens per thread; fewer: the rows of a token shared by the threads
            for (const int n_tok : { 1, 2, 5, 16, 37 }) {
                // heads by 8 and rows by 32 have their own kernel; 6 heads and 100 rows go the general way
                for (const int n_head : { 8, 32, 6 }) {
                    for (const int n_kv : { 256, 608, 100 }) {
                        if (n_kv % 32 != 0 && n_tok < n_threads) {
                            continue; // the shared way wants the rows by 32, as a cache has them
                        }
                        test_case({ 128, n_head, n_kv, n_tok, 48, mask, 0, 0, n_threads });
                        n_cases++;
                        if (n_kv % 8 == 0) {
                            // groups of 8 rows: all of them, as with a short context, then a part of them
                            test_case({ 128, n_head, n_kv, n_tok, n_kv/8, mask, 8, 40, n_threads });
                            test_case({ 128, n_head, n_kv, n_tok, n_kv/32, mask, 8, 24, n_threads });
                            test_case({ 128, n_head, n_kv, n_tok, n_kv/32, mask, 8, 0, n_threads });
                            n_cases += 3;
                        }
                    }
                }
            }
        }
    }
    for (const int n_threads : { 1, 4 }) {
        for (const int n_tok : { 2, 9 }) {
            // other sizes of the keys
            test_case({ 64, 16, 512, n_tok, 100, MASK_VISIBLE, 0, 0, n_threads });
            test_case({ 48, 32, 512, n_tok, 100, MASK_F16, 0, 0, n_threads });
            // heads that do not fill 16 or 32, and more than 32 of them
            for (const int n_head : { 1, 17, 24, 40, 70 }) {
                test_case({ 128, n_head, 320, n_tok, 60, MASK_F32, 0, 0, n_threads });
                test_case({ 128, n_head, 320, n_tok, 10, MASK_VISIBLE, 8, 30, n_threads });
                n_cases += 2;
            }
            // other sizes of the groups, one of them not a divisor of 32
            for (const int group_size : { 2, 4, 16, 32, 5 }) {
                test_case({ 128, 32, 640, n_tok, 640/group_size/3, MASK_VISIBLE, group_size, 50, n_threads });
                test_case({ 128, 32, 640, n_tok, 640/group_size, MASK_F16, group_size, 50, n_threads });
                n_cases += 2;
            }
            // nearly every row asked for, and one row only
            test_case({ 128, 32, 256, n_tok, 255, MASK_F32, 0, 0, n_threads });
            test_case({ 128, 32, 256, n_tok, 1, MASK_VISIBLE, 0, 0, n_threads });
            test_case({ 128, 32, 256, n_tok, 1, MASK_VISIBLE, 8, 1, n_threads });
            n_cases += 5;
        }
    }
    // many scores that are equal: every row is the same key, so the rows in view all score the same
    equal_keys = true;
    for (const int n_threads : { 1, 4 }) {
        for (const int n_tok : { 2, 9 }) {
            test_case({ 128, 32, 512, n_tok, 100, MASK_VISIBLE, 0, 0, n_threads });
            test_case({ 128, 32, 512, n_tok, 100, MASK_F32, 0, 0, n_threads });
            test_case({ 128, 32, 512, n_tok, 16, MASK_F16, 8, 60, n_threads });
            n_cases += 3;
        }
    }
    equal_keys = false;

    if (n_fail == 0) {
        printf("the indexer gives the best rows in %d cases\n", n_cases);
    }
    return n_fail ? 1 : 0;
}
