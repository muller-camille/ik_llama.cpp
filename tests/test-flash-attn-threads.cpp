// flash_attn_ext must not depend on the number of threads: every thread count has to give the result of one
// thread, and one thread the result of the generic implementation. Covers head sizes with and without an IQK
// kernel, GQA and MLA head layouts, and cache lengths that are not a multiple of 32.
//
// A second part gives a cache with one K/V head a mask that leaves out most of the rows, in one or several
// places, as a cache made of parts does, with the list of the rows of each query or that list alone, and compares with a
// sum written out here in double precision. GGML_FA_SKIP_MASKED=0 in the environment sends these cases the other
// way through the code, and GGML_FA_ROW_GROUPS=<n> sets how many groups of rows the threads form.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

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

// Which cache rows query iq sees, of n_q queries over n_kv rows.
//   0: a cache made of two halves, each filled from its start, the first one causal
//   1: a window that moves with the query, in the middle of the cache
//   2: six separate blocks of rows, more than are taken as separate runs
//   3: as 0, and the first query sees nothing at all
//   4: a window that moves with the query, and one row in five of the rest, a little different for every query
static bool sees(int layout, int iq, int n_q, int ik, int n_kv) {
    const int half = n_kv/2;
    switch (layout) {
        case 0: return ik < half ? ik < 40 - (n_q - 1 - iq) : ik - half < 21;
        case 1: return ik >= half - 50 + iq && ik < half + 9 + iq;
        case 2: return (ik/32) % 3 == 1 && ik/32 < 18 && ik % 32 >= iq && ik % 32 < 20 + iq;
        case 3: return iq > 0 && sees(0, iq, n_q, ik, n_kv);
        // 4: a window, and rows picked all over the rest of the cache, not the same for every query
        default: return (ik >= 100 + iq && ik < 140 + iq) || (ik >= 256 && (ik*7 + iq/2) % 5 == 0);
    }
}

// index: 0 = the mask alone, 1 = the mask and the list of the rows of each query, 2 = the lists alone
// n_split > 0: K, which is V then, is the concatenation of its first n_split rows and of the others, left undone
// (ggml_flash_attn_ext_lazy_kv). The attention reads the two parts where they are, or makes the concatenation first.
static int n_lazy = 0;
static void test_masked(ggml_type type_kv, int D, int n_head, int n_q, int n_kv, int layout, bool with_sinks, bool k_is_v, int index_mode,
        int n_split = 0) {
    const bool with_index = index_mode != 0;
    ggml_init_params ip = { (size_t) 1 << 30, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    std::normal_distribution<float> nd(0.0f, 1.0f);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, n_q, n_head);
    for (int64_t i = 0; i < ggml_nelements(q); ++i) {
        ((float *) q->data)[i] = nd(rng);
    }
    ggml_tensor * k = new_cache(ctx, type_kv, D, n_kv, 1);
    ggml_tensor * v = k_is_v ? k : new_cache(ctx, type_kv, D, n_kv, 1);
    // the two parts, each with rows of its own
    ggml_tensor * part[2] = { nullptr, nullptr };
    if (n_split > 0) {
        const size_t row = ggml_row_size(type_kv, D);
        part[0] = ggml_new_tensor_4d(ctx, type_kv, D, 1, n_split, 1);
        part[1] = ggml_new_tensor_4d(ctx, type_kv, D, 1, n_kv - n_split, 1);
        memcpy(part[0]->data, k->data, row*n_split);
        memcpy(part[1]->data, (const char *) k->data + row*n_split, row*(n_kv - n_split));
    }
    ggml_tensor * sinks = nullptr;
    if (with_sinks) {
        sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_head);
        for (int h = 0; h < n_head; ++h) {
            ((float *) sinks->data)[h] = nd(rng);
        }
    }
    const int n_mask = GGML_PAD(n_q, GGML_KQ_MASK_PAD);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_mask);
    for (int iq = 0; iq < n_mask; ++iq) {
        for (int ik = 0; ik < n_kv; ++ik) {
            const bool visible = iq < n_q && sees(layout, iq, n_q, ik, n_kv);
            ((ggml_fp16_t *) mask->data)[(size_t) iq*n_kv + ik] = ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
        }
    }
    // the rows of every query as a list, as ggml_mask_to_index makes it: the rows in view, then -1
    ggml_tensor * index = nullptr;
    if (with_index) {
        // long enough for the rows in view, and shorter than the cache: a list as long as the cache is left to
        // the generic implementation
        const int n_idx = layout == 4 ? GGML_PAD(48 + (n_kv - 256)/5, 32) : 256;
        index = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_idx, GGML_PAD(n_q, 16));
        for (int iq = 0; iq < (int) index->ne[1]; ++iq) {
            int32_t * id = (int32_t *) index->data + (size_t) iq*n_idx;
            int n = 0;
            for (int ik = 0; ik < n_kv && n < n_idx; ++ik) {
                if (iq < n_q && sees(layout, iq, n_q, ik, n_kv)) {
                    id[n++] = ik;
                }
            }
            while (n < n_idx) {
                id[n++] = -1;
            }
        }
    }
    const float scale = 1.0f/std::sqrt((float) D);

    // the cache rows as the kernels read them
    std::vector<float> kf((size_t) D*n_kv), vf((size_t) D*n_kv);
    ggml_internal_get_type_traits(type_kv).to_float(k->data, kf.data(), (int64_t) D*n_kv);
    ggml_internal_get_type_traits(type_kv).to_float(v->data, vf.data(), (int64_t) D*n_kv);
    std::vector<float> ref((size_t) D*n_head*n_q);
    std::vector<double> w(n_kv), acc(D);
    for (int iq = 0; iq < n_q; ++iq) {
        for (int h = 0; h < n_head; ++h) {
            const float * qr = (const float *) q->data + ((size_t) h*n_q + iq)*D;
            double m = with_sinks ? ((float *) sinks->data)[h] : -INFINITY;
            for (int ik = 0; ik < n_kv; ++ik) {
                w[ik] = -INFINITY;
                if (!sees(layout, iq, n_q, ik, n_kv)) {
                    continue;
                }
                double d = 0;
                for (int i = 0; i < D; ++i) {
                    d += (double) qr[i]*kf[(size_t) ik*D + i];
                }
                w[ik] = d*scale;
                m = std::max(m, w[ik]);
            }
            double sum = with_sinks ? std::exp(((float *) sinks->data)[h] - m) : 0.0;
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int ik = 0; ik < n_kv; ++ik) {
                if (w[ik] == -INFINITY) {
                    continue;
                }
                const double e = std::exp(w[ik] - m);
                sum += e;
                for (int i = 0; i < D; ++i) {
                    acc[i] += e*vf[(size_t) ik*D + i];
                }
            }
            // the result is [D, n_head, n_q]
            float * r = ref.data() + ((size_t) iq*n_head + h)*D;
            for (int i = 0; i < D; ++i) {
                r[i] = sum > 0 ? (float) (acc[i]/sum) : 0.0f;
            }
        }
    }

    // The f16 kernels compute in f32. A quantized cache is met with a quantized query, and without a kernel for
    // the head size the generic implementation adds up in f16.
    const double tolerance = type_kv != GGML_TYPE_F16 ? 2e-2 : D == 48 ? 1e-2 : 1e-4;
    std::string bad;
    for (int n_threads : { 1, 2, 3, 4, 7, 8, 16, 32, 64 }) {
        if (verbose) {
            fprintf(stderr, "masked: %s head size %d, %d heads, %d query row(s), cache %d, layout %d, %d threads\n",
                    ggml_type_name(type_kv), D, n_head, n_q, n_kv, layout, n_threads);
        }
        ggml_tensor * kk = k, * vv = v, * cat = nullptr;
        if (n_split > 0) {
            cat = ggml_concat(ctx, part[0], part[1], 2);
            kk  = ggml_permute(ctx, cat, 0, 2, 1, 3);
            vv  = ggml_permute(ctx, cat, 0, 2, 1, 3);
        }
        ggml_tensor * y = ggml_flash_attn_ext(ctx, q, kk, vv, index_mode == 2 ? nullptr : mask, scale, 0.0f, 0.0f);
        if (sinks) {
            ggml_flash_attn_ext_add_sinks(y, sinks);
        }
        if (index) {
            y->src[5] = index;
        }
        if (cat && ggml_flash_attn_ext_lazy_kv(y)) {
            // nobody may read the rows at the place of the concatenation unless the attention has put them there
            memset(cat->data, 0xff, ggml_nbytes(cat));
            n_lazy++;
        }
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_graph_compute_with_ctx(ctx, gf, n_threads);
        const float * p = (const float *) y->data;
        double num = 0, den = 0;
        bool finite = true;
        for (size_t i = 0; i < ref.size(); ++i) {
            finite = finite && std::isfinite(p[i]);
            num += (double) (p[i] - ref[i])*(p[i] - ref[i]);
            den += (double) ref[i]*ref[i];
        }
        const double err = den > 0 ? std::sqrt(num/den) : std::sqrt(num);
        if (!finite || !(err < tolerance)) {
            char buf[64];
            snprintf(buf, sizeof(buf), " [%d threads: %.1e]", n_threads, finite ? err : NAN);
            bad += buf;
        }
    }
    if (!bad.empty()) {
        fprintf(stderr, "FAILED: masked %s head size %3d, %2d heads, %d query row(s), cache %4d, layout %d%s%s%s%s:%s\n",
                ggml_type_name(type_kv), D, n_head, n_q, n_kv, layout, with_sinks ? ", sinks" : "", k_is_v ? ", K == V" : "",
                index_mode == 1 ? ", index list" : index_mode == 2 ? ", index list alone" : "",
                n_split > 0 ? (", in two parts, the first of " + std::to_string(n_split) + " rows").c_str() : "", bad.c_str());
        n_fail++;
    }
    ggml_free(ctx);
}

// The two parts of K as the graph of a model has them: results of nodes, in memory that the allocator of the graph
// hands out, and takes back once a tensor has no reader left. The attention node reads the parts after the
// concatenation that was left undone, which is their only other reader. Unless it names them as its own sources,
// their memory is free by then, and the first tensor to be put there is the result of the node itself: with one
// row group a thread writes the result of a query row before it reads the rows of the next one. That is how a
// server ended in NaN after a prompt of 20000 tokens (the last five tokens of a prompt are decoded together).
static void test_allocated(int n_q, int n_kv, int n_split) {
    const int D = 512, n_head = 64, layout = 4;
    ggml_init_params ip_in = { (size_t) 1 << 28, nullptr, false };
    ggml_context * ctx_in = ggml_init(ip_in);

    std::normal_distribution<float> nd(0.0f, 1.0f);
    ggml_tensor * q = ggml_new_tensor_3d(ctx_in, GGML_TYPE_F32, D, n_q, n_head);
    for (int64_t i = 0; i < ggml_nelements(q); ++i) {
        ((float *) q->data)[i] = nd(rng);
    }
    ggml_tensor * src[2] = {
        ggml_new_tensor_4d(ctx_in, GGML_TYPE_F16, D, 1, n_split, 1),
        ggml_new_tensor_4d(ctx_in, GGML_TYPE_F16, D, 1, n_kv - n_split, 1),
    };
    for (ggml_tensor * t : src) {
        for (int64_t i = 0; i < ggml_nelements(t); ++i) {
            ((ggml_fp16_t *) t->data)[i] = ggml_fp32_to_fp16(nd(rng));
        }
    }
    const int n_idx = GGML_PAD(48 + (n_kv - 256)/5, 32);
    ggml_tensor * index = ggml_new_tensor_2d(ctx_in, GGML_TYPE_I32, n_idx, GGML_PAD(n_q, 16));
    for (int iq = 0; iq < (int) index->ne[1]; ++iq) {
        int32_t * id = (int32_t *) index->data + (size_t) iq*n_idx;
        int n = 0;
        for (int ik = 0; ik < n_kv && n < n_idx; ++ik) {
            if (iq < n_q && sees(layout, iq, n_q, ik, n_kv)) {
                id[n++] = ik;
            }
        }
        while (n < n_idx) {
            id[n++] = -1;
        }
    }
    const float scale = 1.0f/std::sqrt((float) D);

    // parts are graph results when `allocated`, tensors with memory of their own otherwise: the cases above
    auto attention = [&](ggml_context * ctx, bool allocated) {
        ggml_tensor * part[2] = { src[0], src[1] };
        if (allocated) {
            part[0] = ggml_dup(ctx, src[0]);
            part[1] = ggml_dup(ctx, src[1]);
        }
        ggml_tensor * cat = ggml_concat(ctx, part[0], part[1], 2);
        ggml_tensor * kk  = ggml_permute(ctx, cat, 0, 2, 1, 3);
        ggml_tensor * y   = ggml_flash_attn_ext(ctx, q, kk, kk, nullptr, scale, 0.0f, 0.0f);
        y->src[5] = index;
        if (ggml_flash_attn_ext_lazy_kv(y)) {
            n_lazy++;
        }
        return y;
    };

    std::string bad;
    for (int n_threads : { 1, 4, 32 }) {
        ggml_tensor * y_ref = attention(ctx_in, false);
        ggml_cgraph * gf_ref = ggml_new_graph(ctx_in);
        ggml_build_forward_expand(gf_ref, y_ref);
        ggml_graph_compute_with_ctx(ctx_in, gf_ref, n_threads);

        ggml_init_params ip = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * y = attention(ctx, true);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        ggml_gallocr_alloc_graph(galloc, gf);
        ggml_cplan plan = ggml_graph_plan(gf, n_threads);
        std::vector<uint8_t> work(plan.work_size);
        plan.work_data = work.data();
        ggml_graph_compute(gf, &plan);

        const float * p = (const float *) y->data, * r = (const float *) y_ref->data;
        double num = 0, den = 0;
        bool finite = true;
        for (int64_t i = 0; i < ggml_nelements(y); ++i) {
            finite = finite && std::isfinite(p[i]);
            num += (double) (p[i] - r[i])*(p[i] - r[i]);
            den += (double) r[i]*r[i];
        }
        const double err = den > 0 ? std::sqrt(num/den) : std::sqrt(num);
        if (!finite || !(err < 1e-5)) {
            char buf[64];
            snprintf(buf, sizeof(buf), " [%d threads: %.1e]", n_threads, finite ? err : NAN);
            bad += buf;
        }
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
    }
    if (!bad.empty()) {
        fprintf(stderr, "FAILED: parts in memory of the graph, %d query row(s), cache %d, the first part of %d rows:%s\n",
                n_q, n_kv, n_split, bad.c_str());
        n_fail++;
    }
    ggml_free(ctx_in);
}

int main(int argc, char ** argv) {
    verbose = argc > 1 && strcmp(argv[1], "-v") == 0;
    if (argc > 1 && strcmp(argv[1], "--allocated") == 0) {
        // these cases alone: they take a second
        for (int n_q : { 1, 5 }) {
            for (int n_split : { 704, 160 }) {
                test_allocated(n_q, 2048, n_split);
            }
        }
        printf("%d of 4 cases failed (%d attention nodes had their cache in two parts)\n", n_fail, n_lazy);
        return n_fail ? 1 : 0;
    }
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
    // one K/V head, a mask that leaves out most of the cache
    const shape masked[] = {
        { GGML_TYPE_F16,  512, 64, 1 }, // DeepSeek-V4
        { GGML_TYPE_F16,  128, 16, 1 },
        { GGML_TYPE_F16,   64, 12, 1 },
        { GGML_TYPE_F16,   48,  6, 1 }, // no IQK kernel
        { GGML_TYPE_Q8_0,  64,  8, 1 },
    };
    for (const auto & s : masked) {
        for (int n_q : { 1, 2, 3, 5, 7, 9 }) {
            for (int n_kv : { 512, 768 }) {
                for (int layout = 0; layout < 5; ++layout) {
                    if (layout == 3 && n_q == 1) {
                        continue; // a token sees at least itself
                    }
                    const bool variant = (n_q + layout) % 2 == 0;
                    test_masked(s.type, s.D, s.n_head, n_q, n_kv, layout, variant, !variant, 0);
                    ++n_tests;
                    if (layout != 3 && s.D != 48) {
                        // the same rows as index lists, with the mask and without it; every query needs one
                        // row at least, and a kernel
                        test_masked(s.type, s.D, s.n_head, n_q, n_kv, layout, !variant, variant, 1);
                        test_masked(s.type, s.D, s.n_head, n_q, n_kv, layout, variant, variant, 2);
                        n_tests += 2;
                    }
                }
            }
        }
    }
    // lists alone, over a long cache: more rows than the 8192 whose mask every thread is willing to scan
    for (int n_q : { 1, 3 }) {
        test_masked(GGML_TYPE_F16, 512, 64, n_q, 9216, 4, n_q == 1, true, 2);
        ++n_tests;
    }
    // K and V in two parts that are not put together: the first part ends at a whole block of 32 rows or not, the
    // rows in view lie in one part or in both, and with 9 query rows the kernels do not read through at all
    for (int n_q : { 1, 3, 5, 9 }) {
        for (int n_kv : { 768, 2048 }) {
            for (int n_split : { 256, 352, 250 }) {
                for (int layout : { 0, 1, 4 }) {
                    for (int index_mode = 0; index_mode < 3; ++index_mode) {
                        test_masked(GGML_TYPE_F16, 512, 64, n_q, n_kv, layout, (n_q + layout) % 2 == 0, true, index_mode, n_split);
                        ++n_tests;
                    }
                }
            }
        }
    }
    // the same over a long cache: the last tokens of a long prompt are decoded a few at a time, each on the rows
    // of its list, raw rows in one part and all the compressed rows of the context in the other
    for (int n_q : { 1, 3, 5 }) {
        for (int n_split : { 256, 250 }) {
            for (int index_mode = 1; index_mode < 3; ++index_mode) {
                test_masked(GGML_TYPE_F16, 512, 64, n_q, 12800, 4, (n_q + index_mode) % 2 == 0, true, index_mode, n_split);
                ++n_tests;
            }
        }
    }
    // the two parts in memory that the allocator of a graph hands out: the result of the node fits where the
    // first part is (704 rows), or only where the second one is (160 rows first)
    for (int n_q : { 1, 5 }) {
        for (int n_split : { 704, 160 }) {
            test_allocated(n_q, 2048, n_split);
            ++n_tests;
        }
    }
    printf("%d of %d cases failed (%d attention nodes had their cache in two parts)\n", n_fail, n_tests, n_lazy);
    return n_fail ? 1 : 0;
}
