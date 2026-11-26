// The attention of a batch of tokens through the tiles of Intel AMX (iqk_amx_flash_attn in
// ggml/src/iqk/iqk_amx.cpp, GGML_AMX_ATTN=1) against the same attention written out here in double precision:
//   - with the values as the model has them: how far the rounding to bf16 takes the result;
//   - with the queries, the cache rows and the weights rounded to bf16 as the tiles round them: the sums the tiles
//     make, which leaves only the last bits of fp32 and a weight rounded the other way here and there.
// Then the attention node of a graph, with K = V in f16 and the list of the rows of each token: with the tiles it
// must give, for any number of threads, what the kernel gives when called here for each token, and with fewer than
// 8 tokens what it gives without the tiles.
//
// Results only: nothing is timed here.

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" {
int  iqk_amx_bf16(void);
int  iqk_amx_attn(void);
void iqk_amx_attn_enable(int on);
void iqk_amx_flash_attn(int D, int nq, int n, const float * q, long stride_q, const void * kv, long stride_kv,
                        const int * idx, const uint16_t * mask, const float * sinks, float scale,
                        float * out, long stride_out);
}

static int n_fail = 0, n_cases = 0;

static std::mt19937 rng(1017);

// to the nearest bf16, ties to even
static float bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    u += 0x7fff + ((u >> 16) & 1);
    u &= 0xffff0000u;
    memcpy(&x, &u, 4);
    return x;
}

struct attn_case {
    int D;         // values per row
    int nq;        // query rows (heads) of the token
    int n;         // length of the list
    int n_cache;   // rows of the cache
    int n_listed;  // entries of the list that name a row
    bool holes;    // the entries that name no row are spread over the list, not at its end
    int  masked;   // one row in this many of the list is masked (0: no mask at all)
    bool sinks;
    bool sharp;    // each query row looks like one row of the cache: a few weights carry the sum
    int  pad;      // values between two rows of the queries, of the cache and of the result
};

struct attn_data {
    std::vector<float>       q, sinks;
    std::vector<ggml_fp16_t> kv, mask;
    std::vector<int>         idx;
    long stride_q, stride_kv, stride_out;   // in values
    float scale;
};

static attn_data build(const attn_case & c) {
    attn_data s;
    std::normal_distribution<float> gauss(0.f, 1.f);
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    s.stride_q = s.stride_out = c.D + c.pad;
    s.stride_kv = c.D + 2*c.pad;
    s.scale = 1.0f/std::sqrt((float)c.D);
    s.kv.resize((size_t)c.n_cache*s.stride_kv);
    for (auto & v : s.kv) {
        float x = gauss(rng);
        if (u01(rng) < 0.01f) x *= 6;
        v = ggml_fp32_to_fp16(0.7f*x);
    }
    // the rows of the list: distinct rows of the cache in increasing order, as an index list has them
    std::vector<int> all(c.n_cache);
    for (int i = 0; i < c.n_cache; ++i) all[i] = i;
    std::shuffle(all.begin(), all.end(), rng);
    std::vector<int> picked(all.begin(), all.begin() + c.n_listed);
    std::sort(picked.begin(), picked.end());
    s.idx.assign(c.n, -1);
    if (c.holes) {
        std::vector<int> pos(c.n);
        for (int i = 0; i < c.n; ++i) pos[i] = i;
        std::shuffle(pos.begin(), pos.end(), rng);
        std::sort(pos.begin(), pos.begin() + c.n_listed);
        for (int i = 0; i < c.n_listed; ++i) s.idx[pos[i]] = picked[i];
    } else {
        for (int i = 0; i < c.n_listed; ++i) s.idx[i] = picked[i];
    }
    if (c.masked > 0) {
        // the rows that are not in the list are masked too, and some of the mask values are not -inf
        s.mask.assign(c.n_cache, ggml_fp32_to_fp16(-INFINITY));
        for (int i = 0; i < c.n_listed; ++i) {
            s.mask[picked[i]] = i%c.masked == c.masked - 1 ? ggml_fp32_to_fp16(i%2 ? -INFINITY : -1.0f) : ggml_fp32_to_fp16(0.0f);
        }
    }
    s.q.assign((size_t)c.nq*s.stride_q, 0.f);
    for (int j = 0; j < c.nq; ++j) {
        float * qr = s.q.data() + (size_t)j*s.stride_q;
        for (int i = 0; i < c.D; ++i) qr[i] = gauss(rng);
        if (c.sharp && c.n_listed > 0) {
            const ggml_fp16_t * kr = s.kv.data() + (size_t)picked[rng()%c.n_listed]*s.stride_kv;
            for (int i = 0; i < c.D; ++i) qr[i] += 1.2f*ggml_fp16_to_fp32(kr[i]);
        }
    }
    if (c.sinks) {
        s.sinks.resize(c.nq);
        for (auto & v : s.sinks) v = (c.sharp ? 6.f : 1.f)*gauss(rng);
    }
    return s;
}

// The attention in double precision. rounded: with the queries, the rows and the weights as the tiles take them.
static std::vector<float> reference(const attn_case & c, const attn_data & s, bool rounded) {
    std::vector<float> out((size_t)c.nq*c.D, 0.f);
    std::vector<int> seen;
    for (int j = 0; j < c.n; ++j) {
        if (s.idx[j] >= 0 && (s.mask.empty() || s.mask[s.idx[j]] == 0)) seen.push_back(s.idx[j]);
    }
    std::vector<float> kf(seen.size()*c.D);
    for (size_t r = 0; r < seen.size(); ++r) for (int i = 0; i < c.D; ++i) {
        const float x = ggml_fp16_to_fp32(s.kv[(size_t)seen[r]*s.stride_kv + i]);
        kf[r*c.D + i] = rounded ? bf16(x) : x;
    }
    std::vector<double> w(seen.size()), acc(c.D);
    std::vector<float> qf(c.D);
    for (int j = 0; j < c.nq; ++j) {
        for (int i = 0; i < c.D; ++i) {
            const float x = s.q[(size_t)j*s.stride_q + i];
            qf[i] = rounded ? bf16(x) : x;
        }
        double m = -INFINITY;
        for (size_t r = 0; r < seen.size(); ++r) {
            double d = 0;
            for (int i = 0; i < c.D; ++i) d += (double)qf[i]*kf[r*c.D + i];
            w[r] = d*s.scale;
            m = std::max(m, w[r]);
        }
        double sum = 0;
        std::fill(acc.begin(), acc.end(), 0.0);
        for (size_t r = 0; r < seen.size(); ++r) {
            double e = std::exp(w[r] - m);
            if (rounded) e = bf16((float)std::exp(std::max(w[r] - m, -87.0)));
            sum += e;
            for (int i = 0; i < c.D; ++i) acc[i] += e*kf[r*c.D + i];
        }
        double f = 1;
        if (c.sinks) {
            const double sk = s.sinks[j];
            if (sk > m) { f = std::exp(m - sk); sum = sum*f + 1; }
            else sum += std::exp(sk - m);
        }
        for (int i = 0; i < c.D; ++i) out[(size_t)j*c.D + i] = sum > 0 ? (float)(acc[i]*f/sum) : 0.f;
    }
    return out;
}

// root of the mean square of a - b over that of b, and whether a is finite
static double distance(const float * a, const float * b, size_t n, bool & finite) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        finite = finite && std::isfinite(a[i]);
        num += ((double)a[i] - b[i])*((double)a[i] - b[i]);
        den += (double)b[i]*b[i];
    }
    return den > 0 ? std::sqrt(num/den) : std::sqrt(num);
}

static std::vector<float> run_kernel(const attn_case & c, const attn_data & s, float fill) {
    std::vector<float> padded((size_t)c.nq*s.stride_out, fill);
    iqk_amx_flash_attn(c.D, c.nq, c.n, s.q.data(), s.stride_q*(long)sizeof(float), s.kv.data(), s.stride_kv*(long)sizeof(ggml_fp16_t),
            s.idx.data(), s.mask.empty() ? nullptr : s.mask.data(), s.sinks.empty() ? nullptr : s.sinks.data(), s.scale,
            padded.data(), s.stride_out*(long)sizeof(float));
    std::vector<float> out((size_t)c.nq*c.D);
    for (int j = 0; j < c.nq; ++j) {
        memcpy(out.data() + (size_t)j*c.D, padded.data() + (size_t)j*s.stride_out, c.D*sizeof(float));
        // nothing may be written between two rows of the result
        for (int i = c.D; i < s.stride_out; ++i) {
            if (padded[(size_t)j*s.stride_out + i] != fill) { out[0] = NAN; }
        }
    }
    return out;
}

static void check_kernel(const attn_case & c) {
    const attn_data s = build(c);
    const std::vector<float> y = run_kernel(c, s, 7.f);
    const std::vector<float> exact = reference(c, s, false), same = reference(c, s, true);
    bool finite = true;
    const double d_exact = distance(y.data(), exact.data(), y.size(), finite);
    const double d_same  = distance(y.data(), same.data(),  y.size(), finite);
    // 8 significant bits on each factor: 1.4e-3 to 3e-3 of the result with 512 values per row, up to 9e-3 with 32,
    // where fewer roundings average out. With the same rounding here, what is left is the last bits of fp32 and a
    // weight rounded the other way now and then: 6e-5 at most in these cases.
    const bool ok = finite && d_exact < (c.D >= 256 ? 5e-3 : 1.5e-2) && d_same < 5e-4;
    ++n_cases;
    if (!ok) ++n_fail;
    printf("%s kernel: %3d values, %2d query rows, list of %3d with %3d rows%s, cache %4d%s%s%s%s: %.1e from the exact result, %.1e from the same sums\n",
           ok ? "ok  " : "FAIL", c.D, c.nq, c.n, c.n_listed, c.holes ? " spread" : "", c.n_cache,
           c.masked ? (", 1 in " + std::to_string(c.masked) + " masked").c_str() : "", c.sinks ? ", sinks" : "",
           c.sharp ? ", sharp" : "", c.pad ? ", rows apart" : "", d_exact, d_same);
}

// A list that names no row leaves the result as it is, and a list whose rows are all masked gives zeros: as the
// usual kernels do.
static void check_empty() {
    attn_case c = { 64, 20, 64, 300, 0, false, 0, false, false, 0 };
    attn_data s = build(c);
    std::vector<float> y = run_kernel(c, s, 7.f);
    bool untouched = true;
    for (float v : y) untouched = untouched && v == 7.f;
    c.n_listed = 40; c.masked = 1; c.sinks = true;
    s = build(c);
    y = run_kernel(c, s, 7.f);
    bool zeros = true;
    for (float v : y) zeros = zeros && v == 0.f;
    n_cases += 2;
    if (!untouched) ++n_fail;
    if (!zeros) ++n_fail;
    printf("%s kernel: a list without rows leaves the result as it was\n", untouched ? "ok  " : "FAIL");
    printf("%s kernel: a list of masked rows gives zeros\n", zeros ? "ok  " : "FAIL");
}

// The attention node of a graph: n_q tokens of n_head heads on a cache of n_kv rows, K = V, every token with its
// list. mode 1: the lists and the mask; 2: the lists alone.
static void check_graph(int D, int n_head, int n_q, int n_kv, int n_idx, int mode, bool with_sinks) {
    ggml_init_params ip = { (size_t)1 << 30, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    std::normal_distribution<float> gauss(0.f, 1.f);

    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, n_q, n_head);
    for (int64_t i = 0; i < ggml_nelements(q); ++i) ((float *)q->data)[i] = gauss(rng);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, n_kv, 1);
    for (int64_t i = 0; i < ggml_nelements(k); ++i) ((ggml_fp16_t *)k->data)[i] = ggml_fp32_to_fp16(0.7f*gauss(rng));
    ggml_tensor * sinks = nullptr;
    if (with_sinks) {
        sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_head);
        for (int h = 0; h < n_head; ++h) ((float *)sinks->data)[h] = gauss(rng);
    }
    // each token sees a window of rows that moves with it and rows picked all over the cache; in mode 1 the list
    // also names rows that the mask hides
    const int n_mask = GGML_PAD(n_q, GGML_KQ_MASK_PAD);
    ggml_tensor * mask  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_mask);
    ggml_tensor * index = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_idx, GGML_PAD(n_q, 16));
    for (int iq = 0; iq < (int)index->ne[1]; ++iq) {
        int32_t * id = (int32_t *)index->data + (size_t)iq*n_idx;
        ggml_fp16_t * m = iq < n_mask ? (ggml_fp16_t *)mask->data + (size_t)iq*n_kv : nullptr;
        int n = 0;
        for (int ik = 0; ik < n_kv; ++ik) {
            const bool window = ik >= 40 + iq && ik < 100 + 3*iq;
            const bool picked = ik >= 256 && (ik*7 + iq/2)%5 == 0;
            const bool hidden = mode == 1 && ik%11 == 3;
            const bool listed = iq < n_q && (window || picked) && n < n_idx - (iq%3 ? 0 : 5);
            if (listed) id[n++] = ik;
            if (m) m[ik] = ggml_fp32_to_fp16(listed && !hidden ? 0.0f : -INFINITY);
        }
        while (n < n_idx) id[n++] = -1;
    }
    const float scale = 1.0f/std::sqrt((float)D);

    // what the kernel gives for each token, called here: [D, n_head, n_q]
    std::vector<float> direct((size_t)D*n_head*n_q);
    for (int iq = 0; iq < n_q; ++iq) {
        iqk_amx_flash_attn(D, n_head, n_idx, (const float *)q->data + (size_t)iq*D, (long)q->nb[2], k->data, (long)k->nb[1],
                (const int *)index->data + (size_t)iq*n_idx, mode == 1 ? (const uint16_t *)mask->data + (size_t)iq*n_kv : nullptr,
                sinks ? (const float *)sinks->data : nullptr, scale, direct.data() + (size_t)iq*n_head*D, D*(long)sizeof(float));
    }

    auto run = [&] (int tiles, int n_threads) {
        iqk_amx_attn_enable(tiles);
        ggml_tensor * y = ggml_flash_attn_ext(ctx, q, k, k, mode == 2 ? nullptr : mask, scale, 0.0f, 0.0f);
        if (sinks) ggml_flash_attn_ext_add_sinks(y, sinks);
        y->src[5] = index;
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_graph_compute_with_ctx(ctx, gf, n_threads);
        iqk_amx_attn_enable(-1);
        return std::vector<float>((const float *)y->data, (const float *)y->data + ggml_nelements(y));
    };

    const std::vector<float> usual = run(0, 4);
    // the exact result, token by token
    std::vector<float> exact((size_t)D*n_head*n_q);
    for (int iq = 0; iq < n_q; ++iq) {
        attn_case c = { D, n_head, n_idx, n_kv, 0, false, mode == 1, with_sinks, false, 0 };
        attn_data s;
        s.stride_q = s.stride_out = s.stride_kv = D;
        s.scale = scale;
        s.q.resize((size_t)n_head*D);
        for (int h = 0; h < n_head; ++h) memcpy(s.q.data() + (size_t)h*D, (const float *)q->data + ((size_t)h*n_q + iq)*D, D*sizeof(float));
        s.kv.assign((const ggml_fp16_t *)k->data, (const ggml_fp16_t *)k->data + (size_t)n_kv*D);
        s.idx.assign((const int *)index->data + (size_t)iq*n_idx, (const int *)index->data + (size_t)(iq + 1)*n_idx);
        if (mode == 1) s.mask.assign((const ggml_fp16_t *)mask->data + (size_t)iq*n_kv, (const ggml_fp16_t *)mask->data + (size_t)(iq + 1)*n_kv);
        if (sinks) s.sinks.assign((const float *)sinks->data, (const float *)sinks->data + n_head);
        const std::vector<float> r = reference(c, s, false);
        memcpy(exact.data() + (size_t)iq*n_head*D, r.data(), r.size()*sizeof(float));
    }
    bool finite = true;
    const double e_tiles = distance(direct.data(), exact.data(), exact.size(), finite);
    const double e_usual = distance(usual.data(),  exact.data(), exact.size(), finite);
    std::string bad;
    // with fewer than 8 tokens the node does not go through the tiles
    const bool through = n_q >= 8;
    for (int n_threads : { 1, 2, 3, 4, 7, 8, 16, 32, 64 }) {
        const std::vector<float> y    = run(1, n_threads);
        const std::vector<float> want = through ? direct : run(0, n_threads);
        if (y.size() != want.size() || memcmp(y.data(), want.data(), y.size()*sizeof(float)) != 0) bad += " " + std::to_string(n_threads);
    }
    const bool ok = finite && bad.empty() && e_tiles < 8e-3 && e_usual < 1e-4;
    ++n_cases;
    if (!ok) ++n_fail;
    printf("%s graph: %3d values, %2d heads, %2d tokens, cache %4d, lists of %3d%s%s: ", ok ? "ok  " : "FAIL", D, n_head, n_q, n_kv, n_idx,
           mode == 1 ? " and the mask" : "", with_sinks ? ", sinks" : "");
    if (!bad.empty()) printf("not the result of the %s with%s threads, ", through ? "kernel called here" : "usual kernels", bad.c_str());
    if (through) printf("%.1e from the exact result with the tiles, %.1e with the usual kernels\n", e_tiles, e_usual);
    else         printf("the usual kernels with or without GGML_AMX_ATTN, %.1e from the exact result\n", e_usual);
    ggml_free(ctx);
}

int main() {
    if (!iqk_amx_bf16()) {
        printf("no AMX-BF16 here (or GGML_AMX=0): nothing to check\n");
        return 0;
    }
    if (iqk_amx_attn()) printf("GGML_AMX_ATTN is set\n");
    // the table that ggml_fp16_to_fp32 reads is filled by the first context
    ggml_free(ggml_init({ (size_t)1 << 20, nullptr, false }));

    //            D  nq    n  cache listed holes masked sinks sharp pad
    check_kernel({ 512, 64, 768, 4096,  768, false, 0, false, false, 0 });   // a token of DeepSeek-V4.1: 64 heads, 768 rows
    check_kernel({ 512, 64, 768, 4096,  768, false, 0, false, true,  0 });
    check_kernel({ 512, 64, 768, 4096,  700, false, 0, true,  true,  0 });
    check_kernel({ 512, 64, 768, 4096,  601, true,  7, false, true,  0 });
    check_kernel({ 512, 64, 512, 2048,   37, false, 0, false, false, 0 });   // the first tokens of a prompt: few rows
    check_kernel({ 512, 64, 256, 2048,    1, false, 0, false, false, 0 });
    check_kernel({ 512,  8, 768, 4096,  768, false, 0, false, true,  0 });   // some of the heads, as when threads share a token
    check_kernel({ 512, 13, 768, 4096,  755, true,  5, true,  true,  3 });
    check_kernel({ 512,  1, 100, 1000,   99, true,  0, true,  false, 1 });
    for (int D : { 32, 64, 96, 128, 256 }) {
        for (int nq : { 1, 5, 16, 17, 31, 32, 33, 48, 50 }) {
            const int n = 1 + rng()%300, cache = n + 1 + rng()%500;
            const attn_case c = { D, nq, n, cache, 1 + (int)(rng()%n), rng()%2 == 0, (int)(rng()%3 ? 0 : 2 + rng()%5),
                                  rng()%2 == 0, rng()%2 == 0, (int)(rng()%3) };
            check_kernel(c);
        }
    }
    // every number of rows around the blocks of 16 and 32
    for (int n = 1; n <= 70; ++n) {
        if (n > 4 && n%16 > 2 && n%16 < 14) continue;
        check_kernel({ 64, 19, n + n%3, 200, n, n%2 == 0, 0, n%4 == 0, n%5 == 0, 0 });
    }
    check_empty();

    check_graph(512, 64, 12, 2048, 768, 2, false);
    check_graph(512, 64,  9, 2048, 512, 1, false);
    check_graph(128, 32, 40, 1500, 256, 1, true);
    check_graph(128, 24, 33, 1500, 192, 2, true);
    check_graph(128, 24, 33, 1500, 200, 2, true);    // lists whose length is not a multiple of 32
    check_graph(512, 64,  9, 2048, 333, 1, false);
    check_graph( 64,  8,  8,  800, 128, 2, false);
    check_graph( 64,  5, 70,  800, 160, 1, false);
    check_graph(128, 32,  4, 1500, 256, 1, false);   // a draft: not through the tiles
    check_graph(128, 32,  1, 1500, 256, 2, false);

    printf("%d of %d cases failed\n", n_fail, n_cases);
    return n_fail ? 1 : 0;
}
