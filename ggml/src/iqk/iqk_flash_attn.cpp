//
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "iqk_config.h"
#include "iqk_mul_mat.h"
#include "iqk_flash_impl.h"
#include "ggml.h"

#if defined IQK_IMPLEMENT && defined GGML_IQK_FLASH_ATTENTION

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <unordered_set>

namespace {
inline uint32_t simple_gcd(uint32_t a, uint32_t b) {
    while (a != b) {
        if (a > b) a -= b;
        else b -= a;
    }
    return a;
}
inline void accumulate_qkv(int Dv, float& M, float& S, float Mj, float Sj, float * Racc, const float * R) {
    if (Mj == -INFINITY) return;
    if (Mj > M) {
        if (M == -INFINITY) {
            std::memcpy(Racc, R, Dv*sizeof(float));
            S = Sj;
        } else {
            float c = exp(M - Mj);
            S = c*S + Sj;
            for (int i = 0; i < Dv; ++i) Racc[i] = c*Racc[i] + R[i];
        }
        M = Mj;
    } else {
        float c = exp(Mj - M);
        S += c*Sj;
        for (int i = 0; i < Dv; ++i) Racc[i] += c*R[i];
    }
}
// What is left to do once the parts of a query are combined: the attention sink of its head, which counts in the sum
// of the weights and brings no value, and the division by that sum. A query that sees no row gets zeros.
inline void finish_qkv(int Dv, float M, float S, const float * sink, float * Racc) {
    if (M == -INFINITY) {
        // no part has written Racc
        std::memset(Racc, 0, Dv*sizeof(float));
        S = 0;
    }
    if (sink) {
        const float s = *sink;
        if (s > M) {
            const float m = expf(M - s);
            for (int i = 0; i < Dv; ++i) Racc[i] *= m;
            S = S*m + 1;
        } else {
            S += expf(s - M);
        }
    }
    const float norm = S > 0 ? 1/S : 0;
    for (int i = 0; i < Dv; ++i) Racc[i] *= norm;
}

// --- One K/V head and few query rows: only the cache rows that a query sees -------------------------------------
//
// The mask leaves out the rows that are not filled yet, and with a cache made of parts (DeepSeek-V4: the last raw
// rows, then the compressed ones) the padding of each part. The kernels compute every row they are given, the masked
// ones come out as zeros. Here the blocks of 32 rows in which no query sees anything are left out: the kernel is
// called on the runs of the other blocks.
//
// The threads form a grid: R groups of rows by G groups of heads. A thread computes the heads of its head group
// on the rows of its row group; with R > 1 the R threads of a head group then combine what they found, after a
// barrier. R = 1 needs neither, but every thread then reads all the rows.
constexpr int    k_fa_seen_max_runs   = 4;
constexpr int    k_fa_seen_max_pieces = k_fa_seen_max_runs + 1;   // a run may span the two parts of K
constexpr int    k_fa_seen_max_q      = 8;     // query rows: with 8 and more, K is repacked for all the threads
constexpr int    k_fa_seen_max_rows   = 8192;  // of the cache: the mask is scanned by every thread
constexpr int    k_fa_seen_max_groups = 8;     // of rows
constexpr int    k_fa_listed_ahead    = 3;     // rows of a list asked for before they are copied

// What the choice of R goes by, in microseconds. Measured on 2 x 16 cores with 32 threads, DeepSeek-V4.1 (64 heads
// of 512, 1 kB per row of K), by letting every node draw its R (GGML_FA_ROW_GROUPS=-1) and timing it:
//
//                                               R = 1     2      4      8
//   one token, 672 rows read in place             99*   73.6   59.5   65.1
//   about 3 tokens, 640 rows of each gathered    258*  210.6  171.7  186.5      (* without the wait for the others)
//
// The rows are in no cache of the core when a node starts. A thread gets those it reads in place at 0.13 us per
// kB and those it copies at 0.26, while the threads of its row group ask for the same ones; that time comes on
// top of the kernel's, which does about 60 products per nanosecond. What another thread has just written costs
// 1 us per kB to read, eight times as much as a row: it is still in the other core's own cache.
constexpr double k_fa_us_per_kb_rows     = 0.134;
constexpr double k_fa_us_per_kb_gathered = 0.26;
constexpr double k_fa_us_per_kb_parts    = 1.0;
constexpr double k_fa_us_barrier         = 5;
constexpr double k_fa_products_per_us    = 60000;

#if defined(__GNUC__) || defined(__clang__)
#define FA_PREFETCH(p) __builtin_prefetch((p), 0, 2)
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#define FA_PREFETCH(p) _mm_prefetch((const char *)(p), _MM_HINT_T1)
#else
#define FA_PREFETCH(p) ((void)(p))
#endif

inline bool fa_skip_masked_enabled() {
    static const bool on = [] { const char * e = getenv("GGML_FA_SKIP_MASKED"); return !(e && e[0] == '0'); }();
    return on;
}
// GGML_FA_ROW_GROUPS=<n> sets R where n can be used; 1 is the split without groups of rows. With -1 every node
// draws its R among those that can be used, to compare them: the profile tells how long the nodes took with each.
inline int fa_env_row_groups() {
    static const int n = [] { const char * e = getenv("GGML_FA_ROW_GROUPS"); return e ? atoi(e) : 0; }();
    return n;
}
// GGML_FA_PARTS_AHEAD=0: a thread that combines what the threads of its head group left does not ask for the next
// part while it adds one. When R is drawn, so is this.
inline bool fa_env_parts_ahead() {
    static const bool on = [] { const char * e = getenv("GGML_FA_PARTS_AHEAD"); return !(e && e[0] == '0'); }();
    return on;
}
// GGML_FA_ROWS_AHEAD=0: rows gathered from the two parts of K are not asked for in advance.
inline bool fa_env_rows_ahead() {
    static const bool on = [] { const char * e = getenv("GGML_FA_ROWS_AHEAD"); return !(e && e[0] == '0'); }();
    return on;
}
// A number that all the threads of a node find and that changes from node to node: the bits of the first query.
inline uint32_t fa_draw(const void * q) {
    uint32_t bits;
    std::memcpy(&bits, q, sizeof(bits));
    return (bits*2654435761u) >> 12;
}
// with one row group there are no parts: counted with the nodes that ask in advance
inline bool fa_parts_ahead(int R, uint32_t draw) {
    return R == 1 || (fa_env_row_groups() < 0 ? (draw >> 8 & 1) == 0 : fa_env_parts_ahead());
}

// The heads a thread can have (R at its largest), and from that the queries of one kernel call and the
// (head, query row) pairs whose result a thread leaves for its head group.
inline int fa_seen_heads_cap(int neq2, int nth) {
    const int R = std::min(k_fa_seen_max_groups, nth);
    return std::min(neq2, (neq2*R + nth - 1)/nth);
}
inline size_t fa_seen_part_size(int Dv, int nq) { return size_t(Dv + 16)*nq*sizeof(float); }
inline size_t fa_seen_pieces_size(int Dv, int neq1, int neq2, int nth) {
    return k_fa_seen_max_pieces*fa_seen_part_size(Dv, std::max(fa_seen_heads_cap(neq2, nth), neq1));
}
inline size_t fa_seen_thread_size(int Dv, int neq1, int neq2, int nth) {
    return fa_seen_pieces_size(Dv, neq1, neq2, nth) + fa_seen_part_size(Dv, fa_seen_heads_cap(neq2, nth)*neq1) + 64;
}

// The runs of 32-row blocks in which at least one of the n_rows mask rows lets a cache row through (anything but
// -inf). Runs in excess of k_fa_seen_max_runs are joined across the smallest gaps. Returns the number of runs, or
// -1 when there are too many to bother.
int fa_seen_runs(const char * mask, int n_rows, size_t stride_m, int nek1, int * first, int * last) {
    constexpr uint16_t h_inf = 0xfc00;
    constexpr int k_max = 32;
    int f[k_max], l[k_max];
    int n = 0;
    bool in_run = false;
    for (int b = 0; b < nek1/32; ++b) {
        uint16_t seen = 0;
        for (int r = 0; r < n_rows; ++r) {
            auto m = (const uint16_t *)(mask + r*stride_m) + 32*b;
            for (int j = 0; j < 32; ++j) seen |= m[j] ^ h_inf;
        }
        if (seen) {
            if (!in_run) {
                if (n == k_max) return -1;
                f[n] = b; in_run = true;
            }
            l[n] = b + 1;
        } else if (in_run) {
            ++n; in_run = false;
        }
    }
    if (in_run) ++n;
    while (n > k_fa_seen_max_runs) {
        int best = 1;
        for (int i = 2; i < n; ++i) if (f[i] - l[i-1] < f[best] - l[best-1]) best = i;
        l[best-1] = l[best];
        for (int i = best; i + 1 < n; ++i) { f[i] = f[i+1]; l[i] = l[i+1]; }
        --n;
    }
    for (int i = 0; i < n; ++i) { first[i] = f[i]; last[i] = l[i]; }
    return n;
}

// Whether the index list of every query row names exactly the rows that its mask lets through, in rising order:
// the list then says nothing that the mask does not. n_seen gets the largest number of rows a query sees.
bool fa_index_is_mask(const ggml_tensor * indexer, const char * mask, int n_rows, size_t stride_m, int nek1, int & n_seen) {
    constexpr uint16_t h_inf = 0xfc00;
    n_seen = 0;
    if (indexer->type != GGML_TYPE_I32 || indexer->ne[1] < n_rows || indexer->ne[2] != 1) return false;
    const int n_idx = int(indexer->ne[0]);
    for (int t = 0; t < n_rows; ++t) {
        auto m  = (const uint16_t *)(mask + t*stride_m);
        auto id = (const int32_t *)((const char *)indexer->data + t*indexer->nb[1]);
        int n_mask = 0;
        for (int j = 0; j < nek1; ++j) n_mask += m[j] != h_inf;
        int n_list = 0, prev = -1;
        for (int j = 0; j < n_idx; ++j) {
            if (id[j] < 0) continue;
            if (id[j] <= prev || id[j] >= nek1 || m[id[j]] == h_inf) return false;
            prev = id[j];
            ++n_list;
        }
        if (n_list != n_mask) return false;
        n_seen = std::max(n_seen, n_mask);
    }
    return true;
}

// For the profile: how often each way was taken, by the number of row groups, and the time of thread 0 in it.
struct fa_stat {
    std::atomic<uint64_t> n{0}, ns{0};
    void add(uint64_t t) { n.fetch_add(1, std::memory_order_relaxed); ns.fetch_add(t, std::memory_order_relaxed); }
};
enum { k_fa_in_place, k_fa_in_place_not_ahead, k_fa_listed, k_fa_listed_not_ahead, k_fa_n_ways };
static fa_stat fa_stats[k_fa_n_ways][k_fa_seen_max_groups + 1];
static std::atomic<uint64_t> fa_n_neither, fa_n_two_parts;
inline uint64_t fa_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// The number of row groups for n_rows rows in view: the one with the least estimated time. A thread fetches
// n_fetched rows in all, 1/R of them with R row groups, reading them in place or copying them. Returns 0 when the
// threads cannot be arranged in a grid with at least one head per thread.
int fa_seen_row_groups(int n_rows, double n_fetched, size_t row_bytes, bool gathered, int Dk, int Dv, int neq1, int neq2, int nth,
        uint32_t draw) {
    auto usable = [n_rows, neq2, nth] (int R) {
        return R >= 1 && R <= k_fa_seen_max_groups && 32*R <= std::max(32, n_rows) && nth%R == 0 && nth/R <= neq2;
    };
    if (usable(fa_env_row_groups())) return fa_env_row_groups();
    if (fa_env_row_groups() < 0) {
        int can[8], n = 0;
        for (int R = 1; R <= k_fa_seen_max_groups; R *= 2) if (usable(R)) can[n++] = R;
        return n > 0 ? can[draw%n] : 0;
    }
    const double pairs   = double(neq1)*neq2/nth;   // (head, query row) pairs per thread, whatever R is
    const double compute = n_rows*pairs*(Dk + Dv)/k_fa_products_per_us;
    int best = 0;
    double best_us = 0;
    for (int R = 1; R <= k_fa_seen_max_groups; R *= 2) {
        if (!usable(R)) continue;
        double us = (gathered ? k_fa_us_per_kb_gathered : k_fa_us_per_kb_rows)*n_fetched*row_bytes/1024/R + compute;
        if (R > 1) us += k_fa_us_per_kb_parts*pairs*(R - 1)*Dv*sizeof(float)/1024 + k_fa_us_barrier;
        if (best == 0 || us < best_us) { best = R; best_us = us; }
    }
    return best;
}

// Some rows of a thread: nk of them (a multiple of 32) in K and V, and the mask of the first query row over them.
struct fa_piece {
    const char * k;
    const char * v;
    const char * mask;
    int          nk;
    int          stride_k, stride_v;
};

// Where the rows of K are, and of V, which is K then: at k, or in two parts that were not put together.
struct fa_rows {
    const char * k;        // nullptr with two parts
    const char * v;
    int          stride_k, stride_v;
    const iqk_fa_two_parts * two = nullptr;
    const char * row_k(int row) const {
        return !two ? k + size_t(row)*stride_k :
               row < two->n0 ? (const char *)two->k0 + size_t(row)*two->stride0
                             : (const char *)two->k1 + size_t(row - two->n0)*two->stride1;
    }
    const char * row_v(int row) const { return two ? row_k(row) : v + size_t(row)*stride_v; }
    bool k_is_v() const { return two || k == v; }
};

// What does not depend on where the rows are.
struct fa_grid_args {
    int int_type_k, int_type_v, Dk, Dv, neq1, neq2, R;
    long nbq2;
    int stride_q, ne1;
    long nb1;
    const char  * q;
    const float * sinks;
    float scale, softcap;
    float * qkv;
    char  * work;          // size_thread bytes per thread, the grid's part first
    size_t  size_thread;
    barrier_t barrier;
    void * barrier_data;
    int ith, nth;
    bool parts_ahead = true;   // ask for the next part while one is added
    bool rows_ahead  = true;   // and, where rows are gathered from the two parts of K, for the next rows
};

// Hands out the rows of query row t, with their mask, and tells whether there are any.
typedef bool (*fa_own_rows_t)(void * ctx, int t, fa_piece & piece);

// The thread computes the heads of its head group, for every query row, on the rows of its row group. Either
// these are the same for every query row: the pieces it is given, and mask_stride is the distance between the
// masks of two query rows. Or own_rows hands out those of each query row. With one row group the results are
// final; otherwise the threads of the head group meet after a barrier and share out the pairs to combine.
bool fa_grid_run(const fa_grid_args & a, const fa_piece * piece, int n_pieces, size_t mask_stride,
        fa_own_rows_t own_rows = nullptr, void * own_ctx = nullptr) {
    const int R = a.R, Dv = a.Dv, neq1 = a.neq1, neq2 = a.neq2;
    const int G = a.nth/R, g = a.ith/R, r = a.ith%R;   // the head group and the row group of this thread
    const int h_first = int(int64_t(neq2)*g/G);
    const int nh      = int(int64_t(neq2)*(g + 1)/G) - h_first;

    // The queries of one kernel call share the conversion of K: the heads of this thread for one token, or the
    // tokens of one head, whichever are more. Tokens with rows of their own are taken one by one.
    const bool by_token = own_rows || nh >= neq1;
    const int  n_call   = by_token ? neq1 : nh;
    const int  nq       = by_token ? nh : neq1;
    const size_t size_part   = fa_seen_part_size(Dv, nq);
    const size_t offset_left = fa_seen_pieces_size(Dv, neq1, neq2, a.nth);
    const int    left_stride = Dv + 16;                       // what a thread leaves per pair: the sum, then M and S
    auto parts = a.work + size_t(a.ith)*a.size_thread;
    auto left  = (float *)(parts + offset_left);
    for (int c = 0; c < n_call; ++c) {
        const int h = by_token ? h_first : h_first + c;
        const int t = by_token ? c : 0;
        auto qc  = (const float *)(a.q + size_t(t)*a.stride_q + size_t(h)*a.nbq2);
        auto sc  = a.sinks ? a.sinks + h : nullptr;
        auto out = a.qkv + (size_t(t)*a.ne1 + h)*a.nb1/sizeof(float);
        const int sq = by_token ? int(a.nbq2) : a.stride_q;
        const int sm = by_token ? 0 : int(mask_stride);
        const int ss = by_token ? 1 : 0;
        const int so = by_token ? Dv : int(a.ne1*a.nb1/sizeof(float));
        fa_piece own;
        const fa_piece * pc = piece;
        int    np       = n_pieces;
        size_t mask_off = size_t(t)*mask_stride;
        if (own_rows) {
            np = own_rows(own_ctx, t, own) ? 1 : 0;
            pc = &own;
            mask_off = 0;
        }
        if (R == 1 && np == 1) {
            if (!iqk_flash_attn_impl(a.int_type_k, a.int_type_v, a.Dk, Dv, nq, pc[0].nk, sq, pc[0].stride_k, pc[0].stride_v, sm, so,
                        qc, pc[0].k, pc[0].v, pc[0].mask + mask_off, sc, ss, a.scale, a.softcap,
                        out, nullptr, nullptr)) return false;
            continue;
        }
        for (int p = 0; p < np; ++p) {
            auto P = (float *)(parts + p*size_part);
            if (!iqk_flash_attn_impl(a.int_type_k, a.int_type_v, a.Dk, Dv, nq, pc[p].nk, sq, pc[p].stride_k, pc[p].stride_v, sm, Dv,
                        qc, pc[p].k, pc[p].v, pc[p].mask + mask_off, nullptr, 0, a.scale, a.softcap,
                        P, P + size_t(Dv)*nq, P + size_t(Dv + 1)*nq)) return false;
        }
        for (int j = 0; j < nq; ++j) {
            // the pair: head h_first + jh of the group, query row jt
            const int jh = by_token ? j : c, jt = by_token ? c : j;
            auto acc = R == 1 ? out + size_t(j)*so : left + (size_t(jt)*nh + jh)*left_stride;
            float M = -INFINITY, S = 0;
            for (int p = 0; p < np; ++p) {
                auto P = (const float *)(parts + p*size_part);
                accumulate_qkv(Dv, M, S, P[size_t(Dv)*nq + j], P[size_t(Dv + 1)*nq + j], acc, P + size_t(j)*Dv);
            }
            if (R == 1) {
                finish_qkv(Dv, M, S, sc ? sc + j*ss : nullptr, acc);
            } else {
                acc[Dv] = M; acc[Dv + 1] = S;
            }
        }
    }
    if (R == 1) return true;

    a.barrier(a.barrier_data);

    // The pairs of the head group go to its R threads in turn; each combines what the R threads left for a pair.
    // A part lies in the cache of the core that wrote it: the next one is asked for while this one is added.
    auto part = [&] (int rr, int u) {
        return (const float *)(a.work + size_t(g*R + rr)*a.size_thread + offset_left) + size_t(u)*left_stride;
    };
    for (int u = r; u < nh*neq1; u += R) {
        const int jt = u/nh, jh = u - jt*nh;
        auto acc = a.qkv + (size_t(jt)*a.ne1 + h_first + jh)*a.nb1/sizeof(float);
        float M = -INFINITY, S = 0;
        for (int rr = 0; rr < R; ++rr) {
            if (a.parts_ahead) {
                const float * next = rr + 1 < R ? part(rr + 1, u) : u + R < nh*neq1 ? part(0, u + R) : nullptr;
                if (next) {
                    for (int i = 0; i < left_stride; i += 16) FA_PREFETCH(next + i);
                }
            }
            auto P = part(rr, u);
            accumulate_qkv(Dv, M, S, P[Dv], P[Dv + 1], acc, P);
        }
        finish_qkv(Dv, M, S, a.sinks ? a.sinks + h_first + jh : nullptr, acc);
    }
    return true;
}

// The rows in view are whole blocks of the cache, read where they are: row group a.ith % R takes its share of the
// blocks, as pieces of the runs. With K in two parts, a piece ends where the first part does: that is at a whole
// block (the caller sees to it).
bool iqk_fa_seen_rows(const fa_grid_args & a, int n_runs, const int * first, const int * last,
        const fa_rows & rows, const char * mask, int stride_m) {
    const int r = a.ith%a.R;
    fa_piece piece[k_fa_seen_max_pieces];
    int n_pieces = 0, n_blocks = 0;
    for (int i = 0; i < n_runs; ++i) n_blocks += last[i] - first[i];
    const int b0 = int(int64_t(n_blocks)*r/a.R), b1 = int(int64_t(n_blocks)*(r + 1)/a.R);
    const int cut = rows.two ? rows.two->n0/32 : -1;   // the block where the second part begins
    auto add = [&] (int block_lo, int block_hi) {
        if (block_lo >= block_hi) return;
        const int row = 32*block_lo;
        const bool second = rows.two && block_lo >= cut;
        piece[n_pieces++] = { rows.row_k(row), rows.row_v(row), mask + size_t(row)*sizeof(uint16_t), 32*(block_hi - block_lo),
            rows.two ? int(second ? rows.two->stride1 : rows.two->stride0) : rows.stride_k,
            rows.two ? int(second ? rows.two->stride1 : rows.two->stride0) : rows.stride_v };
    };
    bool cut_done = false;
    for (int i = 0, pos = 0; i < n_runs; pos += last[i] - first[i], ++i) {
        const int lo = std::max(b0, pos), hi = std::min(b1, pos + last[i] - first[i]);
        if (lo >= hi) continue;
        const int block_lo = first[i] + lo - pos, block_hi = first[i] + hi - pos;
        if (rows.two && !cut_done && block_lo < cut && cut < block_hi) {
            add(block_lo, cut);
            add(cut, block_hi);
            cut_done = true;
        } else {
            add(block_lo, block_hi);
        }
    }
    return fa_grid_run(a, piece, n_pieces, stride_m);
}

// --- The rows come as index lists ------------------------------------------------------------------------------
//
// One list per query row, as for the tokens of a draft once the context is long: the last raw rows and the rows
// that the indexer picked among the compressed ones, which lie anywhere in the cache and are not the same for
// every token. The cache is cut into R ranges of rows, one per row group, that hold about as many rows of the
// first list each. For every query row in turn, a thread copies the rows of its list that lie in the range of its
// row group next to each other, and the kernel runs on them for the heads of the thread. No row is computed for a
// query row that does not name it. A row that several lists name is copied once per list: after the first time
// it comes from the cache of the core.
struct fa_listed {
    const char * lists;    // n_idx rows per query row, the others are < 0
    size_t       stride_l;
    int          n_idx;
    const char * mask;     // or nullptr: every listed row is in view
    size_t       stride_m;
    const fa_rows * rows;  // where K and V are
    size_t       row_size_k, row_size_v;
    int          lo, hi;   // the range of the row group of the thread: lo <= row < hi
    bool         ahead;    // ask for the rows a few places further in the list while one is copied
    char       * buf;      // for the rows of one query row: K, V if it is not K, the mask
    int          cap;      // rows that buf holds
};

// what a thread needs to gather the rows of one list: K and V, and a mask
inline int fa_listed_rows_cap(int n_idx) { return 32*((n_idx + 31)/32); }
inline size_t fa_listed_gather_size(int n_idx, size_t row_size_k, size_t row_size_v) {
    return size_t(fa_listed_rows_cap(n_idx))*(row_size_k + row_size_v + sizeof(uint16_t)) + 64;
}

inline bool fa_listed_in_view(int row, const uint16_t * m, int lo, int hi) {
    return row >= lo && row < hi && !(m && m[row] == 0xfc00);
}

// The rows of query row t in the range of the thread, in the order of the list, up to a whole block of 32 rows.
static bool fa_listed_own_rows(void * ctx, int t, fa_piece & piece) {
    constexpr uint16_t h_inf = 0xfc00;
    const auto & L = *(const fa_listed *)ctx;
    auto id = (const int32_t *)(L.lists + size_t(t)*L.stride_l);
    auto m  = L.mask ? (const uint16_t *)(L.mask + size_t(t)*L.stride_m) : nullptr;
    const bool own_v = !L.rows->k_is_v();
    auto local_k = L.buf;
    auto local_v = own_v ? local_k + size_t(L.cap)*L.row_size_k : local_k;
    auto local_m = (uint16_t *)(local_k + size_t(L.cap)*(L.row_size_k + (own_v ? L.row_size_v : 0)));
    int j = 0;
    for (int i = 0; i < L.n_idx; ++i) {
        if (L.ahead && i + k_fa_listed_ahead < L.n_idx) {
            // Rows that were not just copied next to each other come from the memory, one cache miss after the
            // other unless several are on their way at a time.
            const int next = id[i + k_fa_listed_ahead];
            if (next >= L.lo && next < L.hi) {
                auto pk = L.rows->row_k(next);
                for (size_t o = 0; o < L.row_size_k; o += 64) FA_PREFETCH(pk + o);
                if (own_v) {
                    auto pv = L.rows->row_v(next);
                    for (size_t o = 0; o < L.row_size_v; o += 64) FA_PREFETCH(pv + o);
                }
            }
        }
        const int row = id[i];
        if (!fa_listed_in_view(row, m, L.lo, L.hi)) continue;
        std::memcpy(local_k + size_t(j)*L.row_size_k, L.rows->row_k(row), L.row_size_k);
        if (own_v) std::memcpy(local_v + size_t(j)*L.row_size_v, L.rows->row_v(row), L.row_size_v);
        // a mask may hold other values than 0 for a row in view
        local_m[j++] = m ? m[row] : 0;
    }
    if (j == 0) return false;
    // up to a whole block: rows of zeros that the query does not see
    const int nk = 32*((j + 31)/32);
    if (j < nk) {
        std::memset(local_k + size_t(j)*L.row_size_k, 0, size_t(nk - j)*L.row_size_k);
        if (own_v) std::memset(local_v + size_t(j)*L.row_size_v, 0, size_t(nk - j)*L.row_size_v);
        for (int i = j; i < nk; ++i) local_m[i] = h_inf;
    }
    piece = { local_k, local_v, (const char *)local_m, nk, int(L.row_size_k), int(L.row_size_v) };
    return true;
}

// How many rows the first list has in view.
int fa_listed_count(const ggml_tensor * indexer, const char * mask, int nek1) {
    auto id = (const int32_t *)indexer->data;
    int n = 0;
    for (int i = 0; i < int(indexer->ne[0]); ++i) n += fa_listed_in_view(id[i], (const uint16_t *)mask, 0, nek1);
    return n;
}

bool iqk_fa_listed_rows(const fa_grid_args & a, const ggml_tensor * indexer, int n_first, int nek1,
        const fa_rows & rows, size_t row_size_k, size_t row_size_v, const char * mask, int stride_m) {
    const int R = a.R, r = a.ith%R;
    // The ranges: the rows of the first list in view, R parts of them. They need not be in rising order for the
    // ranges to cover the cache without overlap, only for the parts to be alike.
    int lo = 0, hi = nek1;
    if (R > 1) {
        auto id = (const int32_t *)indexer->data;
        int count = 0, cut = 1, at = 0;
        for (int i = 0; i < int(indexer->ne[0]) && cut <= r + 1 && cut < R; ++i) {
            if (!fa_listed_in_view(id[i], (const uint16_t *)mask, 0, nek1)) continue;
            while (cut < R && cut <= r + 1 && count == int(int64_t(n_first)*cut/R)) {
                at = std::max(at, id[i]);
                if (cut == r) lo = at;
                if (cut == r + 1) hi = at;
                ++cut;
            }
            ++count;
        }
    }
    const int n_idx = int(indexer->ne[0]);
    // the rows start on a cache line
    auto buf = a.work + size_t(a.ith)*a.size_thread + fa_seen_thread_size(a.Dv, a.neq1, a.neq2, a.nth);
    buf += (64 - (uintptr_t(buf) & 63)) & 63;
    // rows in two parts were not copied next to each other a moment ago: they are asked for in advance
    fa_listed L = { (const char *)indexer->data, size_t(indexer->nb[1]), n_idx, mask, size_t(stride_m), &rows,
        row_size_k, row_size_v, lo, hi, rows.two != nullptr && a.rows_ahead, buf, fa_listed_rows_cap(n_idx) };
    return fa_grid_run(a, nullptr, 0, 0, fa_listed_own_rows, &L);
}

// What the profile says about the ways above.
extern "C" IQK_API void iqk_fa_report(void) {
    static const char * names[k_fa_n_ways] = {
        "rows in view read in place", "rows in view read in place, nothing asked for in advance",
        "rows of the index lists gathered", "rows of the index lists gathered, nothing asked for in advance" };
    uint64_t n = fa_n_neither.load();
    for (auto & way : fa_stats) for (auto & st : way) n += st.n.load();
    if (n == 0) return;
    fprintf(stderr, "ggml_profile:\nggml_profile: attention with one K/V head and few query rows: nodes by the way they were computed,\n");
    fprintf(stderr, "ggml_profile: us = time of thread 0 in a node (with one row group it does not wait for the others)\n");
    fprintf(stderr, "ggml_profile:      nodes       us  way\n");
    for (int way = 0; way < k_fa_n_ways; ++way) {
        for (int R = 1; R <= k_fa_seen_max_groups; ++R) {
            const auto & st = fa_stats[way][R];
            if (st.n.load() == 0) continue;
            fprintf(stderr, "ggml_profile: %10llu %8.1f  %s, %d row group%s\n", (unsigned long long)st.n.load(),
                    1e-3*st.ns.load()/st.n.load(), names[way], R, R > 1 ? "s" : "");
        }
    }
    if (fa_n_neither.load() > 0) {
        fprintf(stderr, "ggml_profile: %10llu        -  another way\n", (unsigned long long)fa_n_neither.load());
    }
    if (fa_n_two_parts.load() > 0) {
        fprintf(stderr, "ggml_profile: %10llu        -  of the nodes above read the cache in its two parts, which were not put together\n",
                (unsigned long long)fa_n_two_parts.load());
    }
}

// Decode, one index list per KV head (indexer {nkv, nek2}). The lists are whole blocks of 128 cells,
// so every 32-entry chunk names 32 consecutive cells and is read in place.
bool iqk_fa_indexed_kv_heads(int int_type_k, int int_type_v, int Dk, int Dv, int neq2, int nek2, int nkv,
        long nbq2, long nbk2, long nbv2, long nb1, int stride_k, int stride_v,
        const char * q, const char * k, const char * v, const ggml_fp16_t * mask, const char * idx, size_t idx_stride,
        const float * sinks, float scale, float softcap, float * qkv, char * work,
        barrier_t barrier, void * barrier_data, int ith, int nth) {
    const int rk2    = neq2/nek2;
    const int nchunk = nkv/32;
    const int nunit  = nek2*nchunk;
    const int upt    = (nunit + nth - 1)/nth;
    const int u_first = std::min(ith*upt, nunit);
    const int u_last  = std::min(u_first + upt, nunit);
    const size_t size_part = (Dv + 2)*rk2*sizeof(float);
    auto part = [&](int t, int g) { return (float *)(work + (size_t(t)*nek2 + g)*size_part); };
    auto tmp  = (float *)(work + size_t(nth)*nek2*size_part + ith*size_part);

    for (int g = 0; g < nek2; ++g) {
        auto PM = part(ith, g) + rk2*Dv;
        for (int j = 0; j < rk2; ++j) PM[j] = -INFINITY;
    }
    bool ok = true;
    for (int u = u_first; u < u_last; ) {
        const int g = u/nchunk;
        const int * ig = (const int *)(idx + g*idx_stride);
        const int c = u - g*nchunk;
        const int c_end = std::min(u_last - g*nchunk, nchunk);
        const int c0 = ig[32*c];
        int e = c + 1;
        while (e < c_end && ig[32*e] == c0 + 32*(e - c)) ++e;
        auto Mt = tmp + rk2*Dv, St = Mt + rk2;
        if (!iqk_flash_attn_impl(int_type_k, int_type_v, Dk, Dv, rk2, 32*(e - c), nbq2, stride_k, stride_v, 0, Dv,
                (const float *)(q + g*rk2*nbq2), k + g*nbk2 + size_t(c0)*stride_k, v + g*nbv2 + size_t(c0)*stride_v,
                mask + c0, nullptr, 0, scale, softcap, tmp, Mt, St)) {
            ok = false;
            break;
        }
        auto P = part(ith, g);
        auto PM = P + rk2*Dv, PS = PM + rk2;
        for (int j = 0; j < rk2; ++j) accumulate_qkv(Dv, PM[j], PS[j], Mt[j], St[j], P + j*Dv, tmp + j*Dv);
        u = g*nchunk + e;
    }

    barrier(barrier_data);
    if (!ok) return false;

    for (int h = ith; h < neq2; h += nth) {
        const int g = h/rk2, j = h - g*rk2;
        auto Racc = qkv + h*nb1/sizeof(float);
        float M = -INFINITY, S = 0;
        for (int t = 0; t < nth; ++t) {
            auto P = part(t, g);
            accumulate_qkv(Dv, M, S, P[rk2*Dv + j], P[rk2*Dv + rk2 + j], Racc, P + j*Dv);
        }
        if (sinks) {
            float s = sinks[h];
            if (s > M) {
                float m = expf(M - s);
                for (int i = 0; i < Dv; ++i) Racc[i] *= m;
                S = S*m + 1;
            } else {
                S += expf(s - M);
            }
        }
        float norm = 1/S;
        for (int i = 0; i < Dv; ++i) Racc[i] *= norm;
    }
    return true;
}
// Prefill, one index list per query row and KV head (indexer {nkv, nek2, neq1}), gathered as the nek2 == 1 path does per row.
bool iqk_fa_indexed_kv_heads_rows(int int_type_k, int int_type_v, int Dk, int Dv, int neq1, int neq2, int nek2, int nkv,
        long nbq2, long nbk2, long nbv2, long nb1, int ne1, int stride_q, int stride_k, int stride_v, int stride_m,
        size_t row_size_k, size_t row_size_v, const char * q, const char * k, const char * v, const ggml_fp16_t * mask,
        const char * idx, size_t idx_nb1, size_t idx_nb2, const float * sinks, float scale, float softcap, float * qkv,
        char * work, int ith, int nth) {
    const int rk2     = neq2/nek2;
    const int nunit   = neq1*nek2;
    const int upt     = (nunit + nth - 1)/nth;
    const int u_first = std::min(ith*upt, nunit);
    const int u_last  = std::min(u_first + upt, nunit);
    auto work_k = work + size_t(ith)*(row_size_k + row_size_v + 64)*nkv;
    auto work_v = work_k + row_size_k*nkv;
    auto work_m = (ggml_fp16_t *)(work_v + row_size_v*nkv);

    for (int u = u_first; u < u_last; ++u) {
        const int iq = u/nek2, g = u - iq*nek2;
        auto ig = (const int *)(idx + iq*idx_nb2 + g*idx_nb1);
        auto M  = (const ggml_fp16_t *)((const char *)mask + size_t(iq)*stride_m);
        for (int j = 0; j < nkv; ++j) {
            std::memcpy(work_k + row_size_k*j, k + g*nbk2 + size_t(ig[j])*stride_k, row_size_k);
            std::memcpy(work_v + row_size_v*j, v + g*nbv2 + size_t(ig[j])*stride_v, row_size_v);
            work_m[j] = M[ig[j]];
        }
        if (!iqk_flash_attn_impl(int_type_k, int_type_v, Dk, Dv, rk2, nkv, nbq2, row_size_k, row_size_v, 0, Dv,
                 (const float *)(q + iq*stride_q + g*rk2*nbq2), work_k, work_v, work_m, sinks ? sinks + g*rk2 : nullptr, 1,
                 scale, softcap, qkv + (size_t(iq)*ne1 + g*rk2)*nb1/sizeof(float), nullptr, nullptr)) return false;
    }
    return true;
}
}

static size_t iqk_fa_work_buffer_size_base(const struct ggml_tensor * dst, int nth);

size_t iqk_fa_work_buffer_size(const struct ggml_tensor * dst, int nth) {
    size_t size = iqk_fa_work_buffer_size_base(dst, nth);
    auto Q = dst->src[0];
    auto K = dst->src[1];
    auto V = dst->src[2];
    auto indexer = dst->src[5];
    // must cover the entries of iqk_fa_seen_rows and of iqk_fa_listed_rows in iqk_flash_attn_noalibi
    if (Q->ne[3] == 1 && K->ne[3] == 1 && V->ne[3] == 1 && K->ne[2] == 1 && V->ne[2] == 1 && Q->ne[1] < k_fa_seen_max_q) {
        size_t size_thread = fa_seen_thread_size(int(V->ne[0]), int(Q->ne[1]), int(Q->ne[2]), nth);
        if (indexer && indexer->type == GGML_TYPE_I32) {
            size_thread += fa_listed_gather_size(int(indexer->ne[0]),
                    ggml_row_size(K->type, K->ne[0]), ggml_row_size(V->type, V->ne[0]));
        }
        if (dst->src[3] || indexer) size = std::max(size, nth*size_thread);
    }
    return size;
}

static size_t iqk_fa_work_buffer_size_base(const struct ggml_tensor * dst, int nth) {
    auto Q = dst->src[0];
    auto K = dst->src[1];
    auto V = dst->src[2];
    auto indexer = dst->src[5];
    // must match the sparse entry in iqk_flash_attn_noalibi (an unsupported K/V type enters it there, then aborts)
    if (indexer && indexer->type == GGML_TYPE_I32 && indexer->ne[0] < K->ne[1] && indexer->ne[0] % 32 == 0 && dst->src[3] &&
        Q->ne[3] == 1 && K->ne[3] == 1 && V->ne[3] == 1 &&
        K->ne[2] > 1 && indexer->ne[1] == K->ne[2] && indexer->ne[2] == Q->ne[1] && Q->ne[2] % K->ne[2] == 0) {
        auto row_size_k = ggml_row_size(K->type, K->ne[0]);
        auto row_size_v = ggml_row_size(V->type, V->ne[0]);
        if (Q->ne[1] > 1) {
            // iqk_fa_indexed_kv_heads_rows
            return nth*(row_size_k + row_size_v + 64)*indexer->ne[0];
        }
        // iqk_fa_indexed_kv_heads
        size_t size_part = (V->ne[0] + 2)*(Q->ne[2]/K->ne[2])*sizeof(float);
        return nth*(K->ne[2] + 1)*size_part;
    }
    if (indexer && indexer->type == GGML_TYPE_I32 && indexer->ne[0] < K->ne[1] &&
        Q->ne[3] == 1 && K->ne[3] == 1 && V->ne[3] == 1 && (K->ne[2] == 1 || (Q->ne[1] > 1 &&
        K->nb[2] == ggml_row_size(K->type, K->ne[0]) && V->nb[2] == ggml_row_size(V->type, V->ne[0])))) {
        auto row_size_k = ggml_row_size(K->type, K->ne[0])*K->ne[2];
        auto row_size_v = ggml_row_size(V->type, V->ne[0])*K->ne[2];
        auto work_size  = (row_size_k + row_size_v + 64) * indexer->ne[0];
        size_t result = work_size * nth;
        if (Q->ne[1] == 1) result += 512*sizeof(float);
        return result;
        //return work_size * nth;
    }
    int rk2 = Q->ne[2]/K->ne[2];
    size_t size = 0;
    if (Q->ne[1] >= 8 && K->type == GGML_TYPE_Q8_0) {
        size = ggml_row_size(GGML_TYPE_Q8_0, K->ne[0]) * K->ne[1]*K->ne[2]*K->ne[3];
    }
    if (Q->ne[1] == 1 && Q->ne[3] == 1 && Q->ne[2]/K->ne[2] > 1 && nth >= 1 && K->ne[1]/32 > 1) {
        if (K->ne[2] > 1) {
            // Cover any nek1 that iqk_flash_attn_noalibi narrows to at run time.
            int nstep_k = K->ne[2]*((K->ne[1] + 31)/32);
            if (nstep_k > 4*nth + K->ne[2]) nstep_k = 4*nth + K->ne[2];
            size_t result_size = (V->ne[0] + 16)*Q->ne[2]/K->ne[2]*sizeof(float);
            size += nstep_k*result_size;
            return size;
        }
        // With nth >= rk2 the consumer can fall through to the chunked loop, which uses up to 4*nth slots.
        int nstep_k = nth;
        if (nth >= rk2) {
            nstep_k = (K->ne[1] + 31)/32;
            if (nstep_k > 4*nth) nstep_k = 4*nth;
            if (nstep_k < nth) nstep_k = nth;
        }
        size_t result_size = (V->ne[0] + 16)*rk2*sizeof(float);
        size += nstep_k*result_size;
        return size;
    }
    return size;
}

static inline const std::unordered_set<ggml_type> & supported_kv_types() {
#ifdef GGML_IQK_FA_ALL_QUANTS
    static std::unordered_set<ggml_type> k_supported = {
        GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q8_KV, GGML_TYPE_Q6_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL
    };
#else
    static std::unordered_set<ggml_type> k_supported = {
        GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q8_KV, GGML_TYPE_Q6_0,
    };
#endif
    return k_supported;
}

static inline bool are_kv_types_supported(ggml_type type_k, ggml_type type_v) {
    if (type_k == GGML_TYPE_BF16) {
        if (type_v != type_k) {
            return false;
        }
#ifdef __AVX512BF16__
        return true;
#else
        return false;
#endif
    }
    auto & supported = supported_kv_types();
    auto it_k = supported.find(type_k);
    auto it_v = supported.find(type_v);
    return it_k != supported.end() && it_v != supported.end();
}

// Whether iqk_flash_attn_impl has a kernel for these head sizes and cache types. The kernel dispatch itself is
// asked, with one query row over 32 empty cache rows, and the answer is remembered per thread.
static bool fa_kernel_available(int type_k, int type_v, int Dk, int Dv) {
    struct entry { int type_k, type_v, Dk, Dv; bool ok; };
    constexpr int k_max = 8;
    static thread_local entry cache[k_max];
    static thread_local int   n_cache = 0;
    for (int i = 0; i < std::min(n_cache, k_max); ++i) {
        const auto & e = cache[i];
        if (e.type_k == type_k && e.type_v == type_v && e.Dk == Dk && e.Dv == Dv) return e.ok;
    }
    bool ok = false;
    const auto tk = ggml_type(type_k), tv = ggml_type(type_v);
    if (Dk > 0 && Dv > 0 && Dk % ggml_blck_size(tk) == 0 && Dv % ggml_blck_size(tv) == 0) {
        constexpr int nk = 32;
        const size_t row_k = ggml_row_size(tk, Dk), row_v = ggml_row_size(tv, Dv);
        // rows of zeros are valid data for every cache type
        std::vector<char> k(nk*row_k + 128, 0), v(nk*row_v + 128, 0), q(Dk*sizeof(float) + 128, 0), qkv(Dv*sizeof(float) + 128, 0);
        std::vector<uint16_t> mask(nk, 0);
        auto aligned = [] (std::vector<char> & buf) { return buf.data() + (64 - (uintptr_t(buf.data()) & 63)); };
        ok = iqk_flash_attn_impl(type_k, type_v, Dk, Dv, 1, nk, Dk*sizeof(float), row_k, row_v, 0, Dv,
                (const float *)aligned(q), aligned(k), aligned(v), mask.data(), nullptr, 0, 1.0f, 0.0f,
                (float *)aligned(qkv), nullptr, nullptr);
    }
    cache[n_cache++ % k_max] = { type_k, type_v, Dk, Dv, ok };
    return ok;
}

// TODO: get the ggml_type enum here without pollution
//
extern "C" IQK_API bool iqk_flash_attn_noalibi(int type_q, int type_mask, float max_bias,
                            int neq3, int neq2, long nbq3, long nbq2,
                            int nek3, int nek2, long nbk3, long nbk2,
                            int nev3, int nev2, long nbv3, long nbv2,
                            int ne2,  int ne1,  long nb1,
                            int int_type_k_in,      // type of k
                            int int_type_v,         // type of v
                            int Dk,                 // K head size
                            int Dv,                 // V head size
                            int neq1,               // number of columns in q
                            int nek1,               // number of rows in k
                            int stride_q,           // distance between q columns in bytes
                            int stride_k,           // distance between k rows in bytes
                            int stride_v,           // distance between v rows in bytes
                            int stride_m,           // distance between mask rows (in bytes
                            const void  * q,        // q matrix.
                            const void  * k,        // k matrix. Assumed to be fp16, nq x nk elements
                            const void  * v,        // v matrix. Assumed to be fp16, nq x nk elements
                            const void  * mask,     // mask. If not null, assumed to be fp16. nq x nk elements
                            const void  * sinks,    // mask. If not null, assumed to be fp16. nq x nk elements
                            float         scale,    // scale applied before softmax
                            float         softcap,  // if > 0, a "soft-cap" operation is applied before softmax
                            float       * qkv,      // v*softmax(scale*(k*q))
                            [[maybe_unused]] void * work_buffer_in, [[maybe_unused]] barrier_t barrier, [[maybe_unused]] void * barrier_data,
                            int ith, int nth, int n_swa, [[maybe_unused]] ggml_tensor * indexer,
                            const iqk_fa_two_parts * two_parts) {

    if (type_q != 0 || type_mask != 1 || max_bias > 0) return false;

    // All threads have to give the same answer. When some return false, the caller runs the generic implementation
    // in those threads only, each for its own share of the rows, and a thread that had nothing to compute here and
    // returned true leaves its share uncomputed (or the others waiting at a barrier). So decide what the kernels
    // cannot do before the work is split. An unsupported K/V type combination is reported further down.
    if (are_kv_types_supported(ggml_type(int_type_k_in), ggml_type(int_type_v)) &&
        !fa_kernel_available(int_type_k_in, int_type_v, Dk, Dv)) return false;

    // One K/V head and few query rows: the rows in view, on a grid of threads (see fa_grid_run). Every thread
    // finds the same rows and the same grid, so they all take this way or none does. It is also the only way that
    // reads K in two parts that were not put together: when it cannot, the caller is told so.
    const bool grid = neq3 == 1 && nek3 == 1 && nev3 == 1 && nek2 == 1 && nev2 == 1 && neq1 < k_fa_seen_max_q &&
        fa_skip_masked_enabled() && are_kv_types_supported(ggml_type(int_type_k_in), ggml_type(int_type_v));
    if (two_parts && !grid) return false;
    if (grid) {
        fa_grid_args a = { int_type_k_in, int_type_v, Dk, Dv, neq1, neq2, 0, nbq2, stride_q, ne1, nb1,
            (const char *)q, (const float *)sinks, scale, softcap, qkv, (char *)work_buffer_in,
            fa_seen_thread_size(Dv, neq1, neq2, nth), barrier, barrier_data, ith, nth };
        const fa_rows rows = { two_parts ? nullptr : (const char *)k, two_parts ? nullptr : (const char *)v, stride_k, stride_v, two_parts };
        const size_t row_size_k = ggml_row_size(ggml_type(int_type_k_in), Dk);
        const size_t row_size_v = ggml_row_size(ggml_type(int_type_v), Dv);
        // A mask, and no index list or one that only repeats the mask: the blocks of the cache with a row in
        // view are read where they are. When the rows in view are scattered, so that their blocks hold more
        // than twice as many rows, the list is used. In two parts, the first one has to end at a whole block.
        int n_seen = 0;
        if (mask && nek1%32 == 0 && nek1 <= k_fa_seen_max_rows && (!two_parts || two_parts->n0%32 == 0) &&
            (!indexer || fa_index_is_mask(indexer, (const char *)mask, neq1, stride_m, nek1, n_seen))) {
            int first[k_fa_seen_max_runs], last[k_fa_seen_max_runs];
            const int n_runs = fa_seen_runs((const char *)mask, neq1, stride_m, nek1, first, last);
            int n_blocks = 0;
            for (int r = 0; r < n_runs; ++r) n_blocks += last[r] - first[r];
            const bool scattered = indexer && 32*n_blocks > 2*n_seen;
            const uint32_t draw = fa_draw(q);
            a.R = n_runs > 0 && !scattered ? fa_seen_row_groups(32*n_blocks, 32*n_blocks, row_size_k, false, Dk, Dv, neq1, neq2, nth,
                    draw) : 0;
            if (a.R > 0) {
                a.parts_ahead = fa_parts_ahead(a.R, draw);
                const uint64_t t0 = ith == 0 ? fa_now_ns() : 0;
                const bool ok = iqk_fa_seen_rows(a, n_runs, first, last, rows, (const char *)mask, stride_m);
                if (ith == 0) {
                    fa_stats[a.parts_ahead ? k_fa_in_place : k_fa_in_place_not_ahead][a.R].add(fa_now_ns() - t0);
                    if (two_parts) fa_n_two_parts.fetch_add(1, std::memory_order_relaxed);
                }
                return ok;
            }
        }
        // An index list per query row: every query row on the rows of its list, gathered by the threads of the grid.
        if (indexer && indexer->type == GGML_TYPE_I32 && indexer->ne[2] == 1 && indexer->ne[1] >= neq1) {
            const int n_first = fa_listed_count(indexer, (const char *)mask, nek1);
            // the lists of the other query rows name some rows that the first one does not: a quarter, as a guess
            const uint32_t draw = fa_draw(q);
            a.R = fa_seen_row_groups(n_first, n_first*(1 + 0.25*(neq1 - 1)), row_size_k, true, Dk, Dv, neq1, neq2, nth, draw);
            if (a.R > 0) {
                a.parts_ahead = fa_parts_ahead(a.R, draw);
                // drawn together with the parts when R is drawn: the profile then compares asking in advance with not
                a.rows_ahead  = fa_env_row_groups() < 0 ? a.parts_ahead : fa_env_rows_ahead();
                a.size_thread += fa_listed_gather_size(int(indexer->ne[0]), row_size_k, row_size_v);
                const uint64_t t0 = ith == 0 ? fa_now_ns() : 0;
                const bool ok = iqk_fa_listed_rows(a, indexer, n_first, nek1, rows, row_size_k, row_size_v, (const char *)mask, stride_m);
                if (ith == 0) {
                    fa_stats[a.parts_ahead ? k_fa_listed : k_fa_listed_not_ahead][a.R].add(fa_now_ns() - t0);
                    if (two_parts) fa_n_two_parts.fetch_add(1, std::memory_order_relaxed);
                }
                return ok;
            }
        }
        if (two_parts) return false;
        if (ith == 0) fa_n_neither.fetch_add(1, std::memory_order_relaxed);
    }

    if (indexer && indexer->type == GGML_TYPE_I32) {
        // an unsupported K/V type must reach the abort below, as without the indexer, not the generic indexer path
        if (indexer->ne[0] < nek1 && neq3 == 1 && nek3 == 1 && nev3 == 1 && nek2 > 1 && indexer->ne[1] == nek2 &&
            indexer->ne[2] == neq1 && neq2 % nek2 == 0 && indexer->ne[0] % 32 == 0 && mask) {
            if (are_kv_types_supported(ggml_type(int_type_k_in), ggml_type(int_type_v))) {
                if (neq1 == 1) {
                    return iqk_fa_indexed_kv_heads(int_type_k_in, int_type_v, Dk, Dv, neq2, nek2, indexer->ne[0],
                            nbq2, nbk2, nbv2, nb1, stride_k, stride_v,
                            (const char *)q, (const char *)k, (const char *)v, (const ggml_fp16_t *)mask,
                            (const char *)indexer->data, indexer->nb[1], (const float *)sinks, scale, softcap, qkv,
                            (char *)work_buffer_in, barrier, barrier_data, ith, nth);
                }
                auto row_size_k = ggml_row_size(ggml_type(int_type_k_in), Dk);
                auto row_size_v = ggml_row_size(ggml_type(int_type_v   ), Dv);
                return iqk_fa_indexed_kv_heads_rows(int_type_k_in, int_type_v, Dk, Dv, neq1, neq2, nek2, indexer->ne[0],
                        nbq2, nbk2, nbv2, nb1, ne1, stride_q, stride_k, stride_v, stride_m, row_size_k, row_size_v,
                        (const char *)q, (const char *)k, (const char *)v, (const ggml_fp16_t *)mask,
                        (const char *)indexer->data, indexer->nb[1], indexer->nb[2], (const float *)sinks, scale, softcap, qkv,
                        (char *)work_buffer_in, ith, nth);
            }
        } else if (indexer->ne[0] < nek1 && neq3 == 1 && nek3 == 1 && nev3 == 1 && (nek2 == 1 || (neq1 > 1 &&
            nbk2 == (long)ggml_row_size(ggml_type(int_type_k_in), Dk) && nbv2 == (long)ggml_row_size(ggml_type(int_type_v), Dv)))) {
            // Workbuffer: we need
            // * indexer->ne[0] * sizeof(ggml_half) to extract the mask for a row
            // * indexer->ne[0] * ggml_row_size(int_type_k_in, Dk) to extract the selected K cache entries
            // * indexer->ne[0] * ggml_row_size(int_type_v, Dv) to extract the selected V cache entries
            auto row_size_k = ggml_row_size(ggml_type(int_type_k_in), Dk)*nek2;
            auto row_size_v = ggml_row_size(ggml_type(int_type_v   ), Dv)*nek2;
            auto work_size  = (row_size_k + row_size_v + 64) * indexer->ne[0];
            ggml_fp16_t h_inf = ggml_fp32_to_fp16(-INFINITY);
            ggml_fp16_t h_zero = ggml_fp32_to_fp16(0.0f);
            int  nkv = indexer->ne[0];
            if (neq1 == 1) {
                GGML_ASSERT(neq2 <= 256);
                int npt = (neq2 + nth - 1)/nth;
                int ith_mid = nth;
                int neq2_this_thread = npt;
                int first = ith*npt;
                if (npt*nth > neq2) {
                    ith_mid = neq2 - nth*(npt - 1);
                    if (ith >= ith_mid) {
                        --neq2_this_thread;
                        //if (neq2_this_thread < 1) return true;
                        first = ith_mid*npt + (ith - ith_mid)*neq2_this_thread;
                    }
                }
                auto idx = (const int *)indexer->data;
                auto M = (const ggml_fp16_t *)mask;
                auto work_k = (char *)work_buffer_in;
                auto work_v = work_k + row_size_k*indexer->ne[0];
                auto work_m = (ggml_fp16_t *)(work_v + row_size_v*indexer->ne[0]) + indexer->ne[0]*ith;
                int last_found = -1;
                for (int j = 0; j < nkv; ++j) {
                    if (idx[j] >= 0) {
                        last_found = j;
                        work_m[j] = M ? M[idx[j]] : h_zero;
                        if (j % nth == ith) {
                            std::memcpy(work_k + row_size_k*j, ((const char *)k + idx[j]*stride_k), row_size_k);
                            if (k != v) {
                                std::memcpy(work_v + row_size_v*j, ((const char *)v + idx[j]*stride_v), row_size_v);
                            }
                        }
                    } else {
                        work_m[j] = h_inf;
                        if (j % nth == ith) {
                            std::memset(work_k + row_size_k*j, 0, row_size_k);
                            if (k != v) {
                                std::memset(work_v + row_size_v*j, 0, row_size_v);
                            }
                        }
                    }
                }
                barrier(barrier_data);
                if (last_found < 0 || neq2_this_thread < 1) return true;
                ++last_found;
                int this_nkv = 32*((last_found + 31)/32);
                auto this_q = (const char *)q + first*nbq2;
                auto this_qkv = qkv + first*nb1/sizeof(float);
                if (!iqk_flash_attn_impl(int_type_k_in, int_type_v,
                         Dk, Dv, neq2_this_thread, this_nkv, nbq2, row_size_k, row_size_v, 0, Dv,
                         (const float *)this_q, work_k, k == v ? work_k : work_v, work_m, sinks ? (const float *)sinks + first : nullptr, 1,
                         scale, softcap,
                         this_qkv, nullptr, nullptr)) return false;
                return true;
            }
            int npt = (neq1 + nth - 1)/nth;
            int ith_mid = nth;
            int neq1_this_thread = npt;
            int first = ith*npt;
            // the heads this thread computes for its query rows: all of them, unless the rows are shared (below)
            int h_first = 0, nh = neq2/nek2;
            if (nek2 == 1 && nth > neq1) {
                // More threads than query rows, as with the few tokens of a draft: one row per thread would leave
                // the other threads without work. The threads of a row share its heads, and each one gathers the
                // cache rows of the query for itself.
                const int t  = int(int64_t(ith)*neq1/nth);
                const int i0 = int((int64_t(t)*nth + neq1 - 1)/neq1);
                const int i1 = int((int64_t(t + 1)*nth + neq1 - 1)/neq1);
                const int g  = ith - i0, ng = i1 - i0;
                h_first = int(int64_t(neq2)*g/ng);
                nh      = int(int64_t(neq2)*(g + 1)/ng) - h_first;
                if (nh <= 0) return true;
                first = t;
                neq1_this_thread = 1;
            } else if (npt*nth > neq1) {
                ith_mid = neq1 - nth*(npt - 1);
                if (ith >= ith_mid) {
                    --neq1_this_thread;
                    if (neq1_this_thread < 1) return true;
                    first = ith_mid*npt + (ith - ith_mid)*neq1_this_thread;
                }
            }
            auto work_k = (char *)work_buffer_in + ith*work_size;
            auto work_v = work_k + row_size_k*indexer->ne[0];
            auto work_m = (ggml_fp16_t *)(work_v + row_size_v*indexer->ne[0]);
            for (int iq = first; iq < first + neq1_this_thread; ++iq) {
                auto idx = (const int *)((const char *)indexer->data + iq*indexer->nb[1]);
                auto M = mask ? (const ggml_fp16_t *)((const char *)mask + iq*size_t(stride_m)) : nullptr;
                int last_found = -1;
                for (int j = 0; j < nkv; ++j) {
                    if (idx[j] >= 0) {
                        std::memcpy(work_k + row_size_k*j, ((const char *)k + idx[j]*stride_k), row_size_k);
                        if (k != v) {
                            std::memcpy(work_v + row_size_v*j, ((const char *)v + idx[j]*stride_v), row_size_v);
                        }
                        work_m[j] = M ? M[idx[j]] : h_zero;
                        last_found = j;
                    } else {
                        std::memset(work_k + row_size_k*j, 0, row_size_k);
                        if (k != v) {
                            std::memset(work_v + row_size_v*j, 0, row_size_v);
                        }
                        work_m[j] = h_inf;
                    }
                }
                if (last_found < 0) continue;
                ++last_found;
                int this_nkv = 32*((last_found + 31)/32);
                auto this_q = (const char *)q + iq*stride_q + h_first*nbq2;
                auto this_qkv = qkv + (size_t(iq)*ne1 + h_first)*nb1/sizeof(float);
                auto this_sinks = sinks ? (const float *)sinks + h_first : nullptr;
                for (int ik2 = 0; ik2 < nek2; ++ik2) {
                    if (!iqk_flash_attn_impl(int_type_k_in, int_type_v,
                                Dk, Dv, nh, this_nkv, nbq2, row_size_k, row_size_v, 0, Dv,
                                (const float *)(this_q + ik2*(neq2/nek2)*nbq2),
                                work_k + ik2*row_size_k/nek2,
                                k == v ? work_k + ik2*row_size_k/nek2: work_v + ik2*row_size_v/nek2,
                                work_m, this_sinks, 1, scale, softcap,
                                this_qkv + ik2*(neq2/nek2)*nb1/sizeof(float), nullptr, nullptr)) return false;
                }
            }
            return true;
        }
    }

    if (auto type_k = ggml_type(int_type_k_in), type_v = ggml_type(int_type_v); !are_kv_types_supported(type_k, type_v)) {
        if (ith == 0) {
            fprintf(stderr, "\n==================== K cache %s coupled with V cache %s is not a supported combination on the CPU backend.\n",
                    ggml_type_name(type_k), ggml_type_name(type_v));
            auto & supported = supported_kv_types();
            fprintf(stderr, "Supported types are:\n");
            for (auto type : supported) {
                fprintf(stderr, "    %s\n", ggml_type_name(type));
            }
            fprintf(stderr, "    Warning: ik_llama.cpp does not support Q5_0 or Q5_1 KV cache on the CPU.\n");
#ifdef __AVX512BF16__
            fprintf(stderr, "    %s, but only if K and V are both %s\n", ggml_type_name(GGML_TYPE_BF16), ggml_type_name(GGML_TYPE_BF16));
#endif
#ifndef GGML_IQK_FA_ALL_QUANTS
            fprintf(stderr, "    To enable q4_0, q4_1, and iq4_nl KV cache types, recompile with -DGGML_IQK_FA_ALL_QUANTS=ON\n");
#endif
        }
        barrier(barrier_data);
        GGML_ABORT("Fatal error");
    }

    if (n_swa > 0 && mask) {
        constexpr int kMinBatch = 256;
        int ntokens = std::max(kMinBatch, neq1);
        int nblock  = (ntokens + n_swa + kMinBatch - 1)/kMinBatch;
        int first   = nek1 - nblock*kMinBatch;
        if (first > 0) {
            k = (const char *)k + int64_t(first)*stride_k;
            v = (const char *)v + int64_t(first)*stride_v;
            mask = (const uint16_t *)mask + first;
            nek1 -= first;
        }
    }

    int rk2 = neq2/nek2;
    int rv2 = neq2/nev2;
    int rk3 = neq3/nek3;
    int rv3 = neq3/nev3;

    int first_k = 0, last_k = nek1;
    if (neq3 == 1 && rk2 > 1 && neq1 == 1 && nek1 > 256 && mask) {
        // This is a quick hack for SWA models.
        // Given that the mask is the same for all layers, ideally we should determine the
        // cache bounds once, and reuse for the whole graph. But even with this simple hack
        // we get non-negligible performance gains for SWA models and long context.
        auto umask = (const uint16_t *)mask;
        for (; first_k < last_k; ++first_k) {
            if (umask[first_k] == 0) break;
        }
        if (first_k == last_k) {
            fprintf(stderr, "============================== %s: found empty attention mask: nek1 = %d, first_k = %d\n", __func__, nek1, first_k);
            GGML_ABORT("Fatal error");
        }
        for (; last_k > first_k; --last_k) {
            if (umask[last_k-1] == 0) break;
        }
        int non = 32*((last_k - first_k + 31)/32);
        first_k = std::max(0, last_k - non);
        last_k = std::min(first_k + non, nek1);
        //printf("nek1 = %d, first = %d, last = %d\n", nek1, first, last);
        if (last_k - first_k <= 3*nek1/4 && (last_k - first_k)%32 == 0) {
            //printf("Reducing from %d to %d\n", nek1, last_k - first_k);
            k = (const void *)((const char *)k + first_k*stride_k);
            v = (const void *)((const char *)v + first_k*stride_v);
            mask = (const void *)((const uint16_t *)mask + first_k);
            nek1 = last_k - first_k;
        }
    }

    // the kernels need a mask and whole blocks of 32 cache rows: also decided here, the same for every thread
    if (!mask || nek1%32 != 0) return false;

    int int_type_k = int_type_k_in;
    auto work_buffer = work_buffer_in;
    if (neq1 >= 8) {
        uint64_t row_size = 0;
        work_buffer = iqk_repack_k(int_type_k, Dk, nek1, nek2, nek3, stride_k, nbk2, nbk3, k, work_buffer_in, ith, nth, int_type_k, row_size);
        if (int_type_k != int_type_k_in) {
            stride_k = row_size;
            nbk2 = stride_k*nek1;
            nbk3 = nbk2*nek2;
            k = work_buffer_in;
            barrier(barrier_data);
        }
    }
    //uint64_t row_size = 0;
    //auto work_buffer = iqk_repack_k(int_type_k, Dk, nek1, nek2, nek3, stride_k, nbk2, nbk3, k, work_buffer_in, ith, nth, int_type_k, row_size);
    //if (int_type_k != int_type_k_in) {
    //    stride_k = row_size;
    //    nbk2 = stride_k*nek1;
    //    nbk3 = nbk2*nek2;
    //    k = work_buffer_in;
    //    barrier(barrier_data);
    //}

    // Getting confused all the time about where to load data from and store the results to
    // (especially when combining the results from the threads).
    // So, for now, making it work just for MLA (nek2 = 1).
    // I think it would also speed up things for GQA, but I'm leaving this for another day.
    if (neq3 == 1 && rk2 > 1 && neq1 == 1 && nth >= 1 && nek1/32 > 1 && nek2 == 1) {
        int nstep_k = nek1/32;
        if (nstep_k >= 4*nth) {
            int nstep_k_per_thread = (nstep_k + nth - 1)/nth;
            int ith_mid = nth;
            int nstep_k_this_thread = nstep_k_per_thread;
            if (nstep_k_per_thread*nth > nstep_k) {
                ith_mid = nstep_k - nth*(nstep_k_per_thread - 1);
                if (ith >= ith_mid) --nstep_k_this_thread;
            }
            //if (ith == 0) fprintf(stderr, "nstep_k = %d, nstep_k_per_thread = %d, ith_mid = %d\n", nstep_k, nstep_k_per_thread, ith_mid);
            nstep_k_per_thread *= 32;
            nstep_k_this_thread *= 32;

            auto kv_offset = ith <= ith_mid ? ith*nstep_k_per_thread
                                           : ith_mid*nstep_k_per_thread + (ith - ith_mid)*nstep_k_this_thread;
            auto kth = (const char *)k + kv_offset*stride_k;
            auto vth = (const char *)v + kv_offset*stride_v;
            auto qth = (const char *)q;
            auto mth = mask ? (const char *)mask + kv_offset*sizeof(uint16_t) : nullptr; // we don't have ggml_half available here

            auto work = (char *)work_buffer;
            auto size_thread = (Dv + 16)*rk2*sizeof(float);
            auto result_buffer = work;
            auto work_this_thread = (float *)(result_buffer + ith*size_thread);
            if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                     Dk, Dv, rk2, nstep_k_this_thread, nbq2, stride_k, stride_v, 0, Dv, //Dk*sizeof(uint16_t), Dv,
                     (const float *)qth, (const void *)kth, (const void *)vth, (const void *)mth, nullptr, 0,
                     scale, softcap,
                     work_this_thread, work_this_thread + (Dv+0)*rk2, work_this_thread + (Dv+1)*rk2)) return false;

            barrier(barrier_data);

            for (int j = ith; j < rk2; j += nth) {
                auto Racc = qkv + j*nb1/sizeof(float);
                float M = -INFINITY, S = 0;
                for (int jth = 0; jth < nth; ++jth) {
                    auto R = (const float *)(result_buffer + jth*size_thread);
                    auto Mj = R + Dv*rk2;
                    auto Sj = Mj + rk2;
                    R += j*Dv;
                    accumulate_qkv(Dv, M, S, Mj[j], Sj[j], Racc, R);
                }
                // the kernel was given no sinks, as its results are parts of a sum
                finish_qkv(Dv, M, S, sinks ? (const float *)sinks + j : nullptr, Racc);
            }
            return true;
        }
        int gcd_k   = simple_gcd(nstep_k, nth);
        if (gcd_k >= 1) {
            int nth_k = nth/gcd_k;
            int ith_k = ith%gcd_k;
            int ith_q = ith/gcd_k;
            int nq_per_thread = (rk2 + nth_k - 1)/nth_k;
            if (nq_per_thread > 1) {
                int ith_mid = nth_k;
                int nq_this_thread = nq_per_thread;
                if (nq_per_thread*nth_k > rk2) {
                    ith_mid = rk2 - nth_k*(nq_per_thread - 1);
                    if (ith_q >= ith_mid) --nq_this_thread;
                }
                int j_mid = ith_mid*nq_per_thread;
                auto work = (char *)work_buffer;
                auto size_thread = (Dv + 16)*nq_per_thread*sizeof(float);
                auto result_buffer = work;

                auto kth = (const char *)k + ith_k*(nek1/gcd_k)*stride_k;
                auto vth = (const char *)v + ith_k*(nek1/gcd_k)*stride_v;
                auto q_offset = ith_q < ith_mid ? ith_q*nq_per_thread*nbq2 : (ith_mid*nq_per_thread + (ith_q - ith_mid)*nq_this_thread)*nbq2;
                auto qth = (const char *)q + q_offset;
                auto mth = mask ? (const char *)mask + ith_k*(nek1/gcd_k)*sizeof(uint16_t) : nullptr; // we don't have ggml_half available here

                // Each thread will produce a result of size Dv*nq_this_thread*sizeof(float)
                // In addition, we need M, S for the nq_this_thread rows the thread is processing
                // => (Dv + 2)*nq_per_thread*sizeof(float). We use (Dv + 16) instead to make sure threads are not
                // writing onto the same cache line.
                auto work_this_thread = (float *)(result_buffer + ith*size_thread);
                if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                            Dk, Dv, nq_this_thread, nek1/gcd_k, nbq2, stride_k, stride_v, 0, Dv, //Dk*sizeof(uint16_t), Dv,
                            (const float *)qth, (const void *)kth, (const void *)vth, (const void *)mth, nullptr, 0,
                            scale, softcap,
                            work_this_thread, work_this_thread + (Dv+0)*nq_this_thread, work_this_thread + (Dv+1)*nq_this_thread)) return false;

                barrier(barrier_data);

                // There are nek1/gcd_k contributions for each j that we need to sum up
                // Thread i computed k/v (i%gcd_k)*(nek1/gcd_k) for j (i/gcd_k)*(rk2/nth_k)...((i/gcd_k)+1)*(rk2/nth_k) and results at offset i*size_thread

                // TODO: simdify this
                // TODO: if nth > rk2, have threads process portions of the rows instead of entire rows as it is now
                for (int j = ith; j < rk2; j += nth) {
                    auto Racc = qkv + j*nb1/sizeof(float);
                    float M = -INFINITY, S = 0;
                    int jth_first, jj, nq_this_j;
                    if (j < j_mid) {
                        jth_first = j/nq_per_thread;
                        jj = j%nq_per_thread;
                        nq_this_j = nq_per_thread;
                    } else {
                        jth_first = ith_mid + (j - j_mid)/(nq_per_thread-1);
                        jj = (j - j_mid)%(nq_per_thread-1);
                        nq_this_j = nq_per_thread - 1;
                    }
                    jth_first *= gcd_k;
                    for (int jth = jth_first; jth < jth_first + gcd_k; ++jth) {
                        auto R = (const float *)(result_buffer + jth*size_thread);
                        auto Mj = R + Dv*nq_this_j;
                        auto Sj = Mj + nq_this_j;
                        R += jj*Dv;
                        accumulate_qkv(Dv, M, S, Mj[jj], Sj[jj], Racc, R);
                    }
                    finish_qkv(Dv, M, S, sinks ? (const float *)sinks + j : nullptr, Racc);
                }
                return true;
            }
        }
    }

                                                                       // not sure why I wanted this to be true
                                                                       // but it does not seem useful
    if (neq3 == 1 && rk2 > 1 && rk2 == rv2 && neq1 == 1 && nth >= 1) { // && nek2*nek1 >= 32*nth) {
        auto result_size = (Dv + 16)*rk2*sizeof(float);
        int gcd = simple_gcd(nek2, nth);
        int nth_k  = nth/gcd;
        int nek2_k = nek2/gcd;
        int nchunk = nek2_k*nek1/32;
        int npt = (nchunk + nth_k - 1)/nth_k;
        int nk;
        if (npt*nth_k == nchunk) {
            nk = 32 * (nek2*nek1/(32*nth));
        } else {
            //int nm = std::max(1, npt/8);
            int nm = 1;
            while (true) {
                if (nm*4 >= npt) break;
                nm *= 2;
            }
            nk = 32*nm;
        }
        //int nk = 32 * (nek2*nek1/(32*nth));
        int nkk = (nek1 + nk - 1)/nk;
        int nstep_k = nek2*nkk;
        //if (ith == 0) {
        //    static int last_nek1 = 0;
        //    if (last_nek1 != nek1) {
        //        printf("rk2 = %d, nek1 = %d, nek2 = %d, nk = %d, nkk = %d, nstep_k = %d\n", (int)rk2, (int)nek1, (int)nek2, nk, nkk, nstep_k);
        //        last_nek1 = nek1;
        //    }
        //}
        for (int istep_k = ith; istep_k < nstep_k; istep_k += nth) {
            int ik02 = istep_k/nkk;
            int ik01 = nk*(istep_k - ik02*nkk);
            int this_nk = ik01 + nk <= nek1 ? nk : nek1 - ik01;
            if (this_nk <= 0) break;
            auto this_result = (float *)((char *)work_buffer + istep_k*result_size);
            auto this_q = (const float *)((const char *)q + ik02*rk2*nbq2);
            auto this_k = (const char *)k + ik01*stride_k + ik02*nbk2;
            auto this_v = (const char *)v + ik01*stride_v + ik02*nbv2;
            auto this_m = mask ? (const char *)mask + ik01*sizeof(uint16_t) : nullptr; // we don't have ggml_half available here
            if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                     Dk, Dv, rk2, this_nk, nbq2, stride_k, stride_v, 0, Dv,
                     this_q, (const void *)this_k, (const void *)this_v, (const void *)this_m, nullptr, 0,
                     scale, softcap, this_result, this_result + (Dv+0)*rk2, this_result + (Dv+1)*rk2)) return false;
        }

        barrier(barrier_data);

        // We have nkk results for each head
        for (int iq2 = ith; iq2 < neq2; iq2 += nth) {
            // ik02*rk2 + il = iq2 (il = 0...rk2-1) => ik02 = iq2/rk2, il = iq2%rk2;
            int ik02 = iq2/rk2;
            int il = iq2 - ik02*rk2;
            auto Racc = qkv + iq2*nb1/sizeof(float);
            //std::memset(Racc, 0, Dv*sizeof(float));
            float M = -INFINITY, S = 0;
            for (int ikk = 0; ikk < nkk; ++ikk) {
                int istep_k = ik02*nkk + ikk;
                auto this_result = (float *)((char *)work_buffer + istep_k*result_size);
                const float * R  = this_result + il*Dv;
                const float * Mj = this_result + Dv*rk2;
                const float * Sj = Mj + rk2;
                accumulate_qkv(Dv, M, S, Mj[il], Sj[il], Racc, R);
            }
            if (sinks) {
                float s = ((const float *)sinks)[iq2];
                if (s > M) {
                    float m = expf(M - s);
                    for (int i = 0; i < Dv; ++i) Racc[i] *= m;
                    S = S*m + 1;
                } else {
                    S += expf(s - M);
                }
            }
            float norm = S > 0 ? 1/S : 1;
            for (int i = 0; i < Dv; ++i) Racc[i] *= norm;
        }
        return true;
    }

    // I keep changing my mind what is the best strategy to split the threads when processing
    // multiple heads. This is my current thinking, the commented out code below was the previous.
    int ntg = nth/simple_gcd(neq2*neq3, nth);
    int neq1g = (neq1 + ntg - 1)/ntg;
    //int64_t work_per_slice = D*nek1*neq1;
    //int ntg = 1;
    //
    // When neq1 is large, it is better to have more than one thread process one (iq2,iq3) matrix
    // But we also want each thread to process the same amount of rows, so neq1 must be a multiple of
    // the number of threads processing the (iq2, iq3) matrix.
    //
    //if (neq1 >= 8*nth) {
    //    if      (nth%8 == 0 && neq1%8 == 0 && work_per_slice >= (1 << 23)) ntg = 8;
    //    else if (nth%4 == 0 && neq1%4 == 0 && work_per_slice >= (1 << 21)) ntg = 4;
    //    else if (nth%2 == 0 && neq1%2 == 0 && work_per_slice >= (1 << 19)) ntg = 2;
    //}
    int counter = 0;
    for (int64_t iq3 = 0; iq3 < neq3; iq3++) {
        for (int64_t iq2 = 0; iq2 < neq2; iq2++) {
            auto sinksf = sinks ? (const float *)sinks + iq2 : nullptr;
            if (counter++ % (nth/ntg) == ith/ntg) {
                int iq1 = (ith%ntg)*neq1g;
                int this_neq1 = std::min(neq1g, neq1-iq1);
                if (this_neq1 > 0) {
                if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                        Dk, Dv, this_neq1, nek1, stride_q, stride_k, stride_v, stride_m, ne1*nb1/sizeof(float),
                        (const float *)((const char *)q + iq2*nbq2 + iq3*nbq3 + iq1*stride_q),
                        (const void  *)((const char *)k + iq2/rk2*nbk2 + iq3/rk3*nbk3),
                        (const void  *)((const char *)v + iq2/rv2*nbv2 + iq3/rv3*nbv3),
                        mask ? (const void  *)((const char *)mask + size_t(iq1)*stride_m) : nullptr, sinksf, 0,
                        scale, softcap,
                        (float *)((char *)qkv + (iq3*ne2*ne1 + iq2 + iq1*ne1)*nb1), nullptr, nullptr)) return false;
                }
            }
        }
    }

    return true;
}

#else

bool iqk_flash_attn_noalibi([[maybe_unused]] int type_q, [[maybe_unused]] int type_mask, [[maybe_unused]] float max_bias,
                            [[maybe_unused]] int neq3, [[maybe_unused]] int neq2, [[maybe_unused]] long nbq3, [[maybe_unused]] long nbq2,
                            [[maybe_unused]] int nek3, [[maybe_unused]] int nek2, [[maybe_unused]] long nbk3, [[maybe_unused]] long nbk2,
                            [[maybe_unused]] int nev3, [[maybe_unused]] int nev2, [[maybe_unused]] long nbv3, [[maybe_unused]] long nbv2,
                            [[maybe_unused]] int ne2,  [[maybe_unused]] int ne1,  [[maybe_unused]] long nb1,
                            [[maybe_unused]] int type_k,             // type of k
                            [[maybe_unused]] int type_v,             // type of v
                            [[maybe_unused]] int Dk,                 // K head size
                            [[maybe_unused]] int Dv,                 // V head size
                            [[maybe_unused]] int nq,                 // number of columns in q
                            [[maybe_unused]] int nk,                 // number of rows in k
                            [[maybe_unused]] int stride_q,           // distance between q columns in bytes
                            [[maybe_unused]] int stride_k,           // distance between k rows in bytes
                            [[maybe_unused]] int stride_v,           // distance between v rows in bytes
                            [[maybe_unused]] int stride_m,           // distance between mask rows (in bytes
                            [[maybe_unused]] const void  * q,        // q matrix.
                            [[maybe_unused]] const void  * k,        // k matrix. Assumed to be fp16, nq x nk elements
                            [[maybe_unused]] const void  * v,        // v matrix. Assumed to be fp16, nq x nk elements
                            [[maybe_unused]] const void  * mask,     // mask. If not null, assumed to be fp16. nq x nk elements
                            [[maybe_unused]] float         scale,    // scale applied before softmax
                            [[maybe_unused]] float         softcap,  // if > 0, a "soft-cap" operation is applied before softmax
                            [[maybe_unused]] float       * qkv,      // v*softmax(scale*(k*q))
                            [[maybe_unused]] void * work_buffer, [[maybe_unused]] barrier_t barrier, [[maybe_unused]] void * barrier_data,
                            [[maybe_unused]] int ith, [[maybe_unused]] int nth, [[maybe_unused]] int n_swa, [[maybe_unused]] ggml_tensor * indexer,
                            [[maybe_unused]] const iqk_fa_two_parts * two_parts) {
    return false;
}

extern "C" IQK_API void iqk_fa_report(void) {}

#endif

