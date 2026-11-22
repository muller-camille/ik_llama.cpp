// INDEXER_TOPK with AVX-512, for index keys in f16.
//
// The scores. A key row is multiplied with all the heads of a token at once: the query of the token is laid out
// with its heads side by side ([dimension][head], 16 heads in a register), a value of the key row is broadcast,
// and one register per 16 heads collects the dot products of the row with them. No shuffle is needed, so the
// two multiply-add units of the CPU are all that limits it. The keys are converted to f32 in units of 32 rows
// that stay in the L1 cache, once for a tile of tokens instead of once per token.
//
// The selection. The key of the n-th highest score is found bit by bit, each bit by counting the scores that
// are at least a value, 16 at a time; the rows over that key are then written out, and as many of those at it
// as are still wanted. That is 34 short passes over the scores whatever their order, where a partial sort
// takes long when the scores grow with the row, and it gives the rows in no particular order: what reads them
// either sorts them by row or sums over them.
//
// A batch with at least as many tokens as threads: a thread takes whole tokens, in tiles. With fewer tokens
// the threads share the rows of every token, wait for each other once, and each token is then finished by one
// thread.

#include "iqk_indexer_512.h"

#if defined IQK_IMPLEMENT && defined __AVX512F__

#include "ggml.h"

#include <immintrin.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int k_unit      = 32;      // key rows converted and multiplied together
constexpr int k_tile_max  = 8;       // tokens a thread computes the scores of together
constexpr int k_tile_rows = 131072;  // rows x tokens of a tile: its scores stay in the L2 cache

inline bool idx512_enabled() {
    static const bool on = [] { const char * e = getenv("GGML_INDEXER_512"); return !(e && e[0] == '0'); }();
    return on;
}

struct idx512_plan {
    int    d, n_head, n_chunk, n_kv, n_tok, n_top;
    int    group_size, n_b, n_top_out;   // group_size > 0: groups of rows are chosen first
    int    tile;
    bool   batch;                         // a thread takes whole tokens
    // the slice of a thread, in bytes from its start
    size_t off_qt, off_w, off_kf, off_keys, off_tmp, off_out, off_group, off_kept, off_score;
    size_t per_thread;
    size_t shared;                        // the scores of every token, when the threads share the rows
};

inline size_t idx512_pad(size_t n) {
    return (n + 63) & ~size_t(63);
}

bool idx512_make_plan(const struct ggml_tensor * dst, int nth, idx512_plan & p) {
    if (!idx512_enabled() || dst->op != GGML_OP_INDEXER_TOPK || dst->op_params[0] != (int32_t) GGML_UNARY_OP_RELU) {
        return false;
    }
    auto k = dst->src[0];
    auto q = dst->src[1];
    auto w = dst->src[2];
    auto m = dst->src[3];
    const bool select_groups = dst->op_params[2] != 0;
    auto top_k_out = dst->src[5];
    if (!select_groups && (dst->src[4] || top_k_out)) return false;   // rows given by a list: the other way
    if (k->type != GGML_TYPE_F16 || q->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32) return false;
    if (m->type != GGML_TYPE_F32 && m->type != GGML_TYPE_F16 && m->type != GGML_TYPE_I32) return false;
    if (k->ne[2] != 1 || k->ne[3] != 1 || k->ne[0] != q->ne[0]) return false;
    if (q->ne[2] != m->ne[1] || q->ne[1] != w->ne[0] || q->ne[2] != w->ne[1]) return false;
    if (m->type != GGML_TYPE_I32 && m->ne[0] != k->ne[1]) return false;
    if (k->ne[0] <= 0 || k->ne[0] % 16 != 0 || k->ne[1] <= 0 || k->ne[1] % k_unit != 0) return false;
    if (q->ne[1] <= 0 || q->ne[2] <= 0 || k->ne[1] > INT32_MAX/2) return false;

    p.d          = (int) k->ne[0];
    p.n_head     = (int) q->ne[1];
    p.n_chunk    = (p.n_head + 31)/32;
    p.n_kv       = (int) k->ne[1];
    p.n_tok      = (int) q->ne[2];
    p.n_top      = (int) dst->ne[0];
    p.group_size = 0;
    p.n_b        = 0;
    p.n_top_out  = 0;
    if (select_groups) {
        const int group_size = (int) dst->op_params[1];
        if (group_size <= 1 || k_unit % group_size != 0) return false;
        p.group_size = group_size;
        p.n_b        = p.n_kv/group_size;
        if (p.n_top <= 0 || p.n_top > p.n_b) return false;
        if (top_k_out) {
            if (top_k_out->type != GGML_TYPE_I32 || top_k_out->ne[0] <= 0 || top_k_out->ne[0] > p.n_kv ||
                    top_k_out->ne[1] != q->ne[2]) return false;
            p.n_top_out = (int) top_k_out->ne[0];
        }
    } else if (p.n_top <= 0 || p.n_kv <= p.n_top) {
        return false;
    }

    p.batch = p.n_tok >= nth;
    p.tile  = p.batch ? std::max(1, std::min(k_tile_max, k_tile_rows/p.n_kv)) : 1;

    size_t off = 0;
    p.off_qt    = off; off += idx512_pad(sizeof(float)*p.tile*p.n_chunk*p.d*32);
    p.off_w     = off; off += idx512_pad(sizeof(float)*p.tile*p.n_chunk*32);
    p.off_kf    = off; off += idx512_pad(sizeof(float)*k_unit*p.d);
    p.off_keys  = off; off += idx512_pad(sizeof(uint32_t)*(p.n_kv + 16));
    p.off_tmp   = off; off += idx512_pad(sizeof(int32_t)*(p.n_kv + 16));
    p.off_out   = off; off += idx512_pad(sizeof(int32_t)*(std::max(p.n_top, p.n_top_out) + 16));
    p.off_group = off; off += idx512_pad(sizeof(float)*(p.n_b + 16));
    p.off_kept  = off; off += idx512_pad(sizeof(uint32_t)*((p.n_b + 31)/32));
    p.off_score = off; off += p.batch ? idx512_pad(sizeof(float)*p.tile*p.n_kv) : 0;
    p.per_thread = off;
    p.shared     = p.batch ? 0 : idx512_pad(sizeof(float)*p.n_tok*p.n_kv);
    return true;
}

// r[j] holds 16 values of row j; after the call it holds the j-th value of the 16 rows
inline void idx512_transpose16(__m512 * r) {
    __m512 t[16], u[16];
    for (int i = 0; i < 16; i += 2) {
        t[i]   = _mm512_unpacklo_ps(r[i], r[i+1]);
        t[i+1] = _mm512_unpackhi_ps(r[i], r[i+1]);
    }
    for (int i = 0; i < 16; i += 4) {
        u[i]   = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t[i]),   _mm512_castps_pd(t[i+2])));
        u[i+1] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t[i]),   _mm512_castps_pd(t[i+2])));
        u[i+2] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t[i+1]), _mm512_castps_pd(t[i+3])));
        u[i+3] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t[i+1]), _mm512_castps_pd(t[i+3])));
    }
    // u[4g+e] holds, in its 128-bit lane L, the values 4L+e of the rows 4g .. 4g+3
    for (int e = 0; e < 4; ++e) {
        const __m512 lo_a = _mm512_shuffle_f32x4(u[e],   u[4+e],  0x88);
        const __m512 hi_a = _mm512_shuffle_f32x4(u[e],   u[4+e],  0xdd);
        const __m512 lo_b = _mm512_shuffle_f32x4(u[8+e], u[12+e], 0x88);
        const __m512 hi_b = _mm512_shuffle_f32x4(u[8+e], u[12+e], 0xdd);
        r[e]      = _mm512_shuffle_f32x4(lo_a, lo_b, 0x88);
        r[8+e]    = _mm512_shuffle_f32x4(lo_a, lo_b, 0xdd);
        r[4+e]    = _mm512_shuffle_f32x4(hi_a, hi_b, 0x88);
        r[12+e]   = _mm512_shuffle_f32x4(hi_a, hi_b, 0xdd);
    }
}

// The query of a token, [head][d] with `stride` bytes from a head to the next, as [chunk of 32 heads][d][32]:
// the heads side by side. The heads a chunk does not have are zeros, and so are their weights.
void idx512_token(const idx512_plan & p, const char * q, size_t stride, const float * w, float * qt, float * wt) {
    for (int c = 0; c < p.n_chunk; ++c) {
        for (int v = 0; v < 2; ++v) {
            const int h0 = 32*c + 16*v;
            const int nh = std::max(0, std::min(16, p.n_head - h0));
            for (int j = 0; j < 16; ++j) {
                wt[32*c + 16*v + j] = j < nh ? w[h0 + j] : 0.0f;
            }
            float * out = qt + (size_t) c*p.d*32 + 16*v;
            for (int i0 = 0; i0 < p.d; i0 += 16) {
                __m512 r[16];
                for (int j = 0; j < 16; ++j) {
                    r[j] = j < nh ? _mm512_loadu_ps((const float *)(q + (size_t)(h0 + j)*stride) + i0) : _mm512_setzero_ps();
                }
                idx512_transpose16(r);
                for (int j = 0; j < 16; ++j) {
                    _mm512_storeu_ps(out + (size_t)(i0 + j)*32, r[j]);
                }
            }
        }
    }
}

inline void idx512_keys_f32(const char * k, size_t nb, int d, float * kf) {
    for (int r = 0; r < k_unit; ++r) {
        const __m256i * x = (const __m256i *)(k + r*nb);
        for (int i = 0; i < d/16; ++i) {
            _mm512_storeu_ps(kf + (size_t) r*d + 16*i, _mm512_cvtph_ps(_mm256_loadu_si256(x + i)));
        }
    }
}

// the sums of 4 registers, in one
inline __m128 idx512_hsum4(__m512 a, __m512 b, __m512 c, __m512 d) {
    const auto half = [] (__m512 x) {
        return _mm256_add_ps(_mm512_castps512_ps256(x), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(x), 1)));
    };
    const __m256 ab = _mm256_hadd_ps(half(a), half(b));
    const __m256 cd = _mm256_hadd_ps(half(c), half(d));
    const __m256 s  = _mm256_hadd_ps(ab, cd);
    return _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
}

// 8 key rows x of d values with the heads of a chunk, 16*nv of them:
//   score[r] += sum over the heads of w[h] * relu(x[r] . q[h])
template <int nv> inline void idx512_rows8(int d, const float * x, const float * qt, const float * w, float * score) {
    __m512 acc[8*nv];
    for (int i = 0; i < 8*nv; ++i) acc[i] = _mm512_setzero_ps();
    const float * xr[8];
    for (int r = 0; r < 8; ++r) xr[r] = x + (size_t) r*d;
    for (int i = 0; i < d; ++i) {
        __m512 qv[nv];
        for (int v = 0; v < nv; ++v) qv[v] = _mm512_loadu_ps(qt + 32*(size_t) i + 16*v);
        for (int r = 0; r < 8; ++r) {
            const __m512 b = _mm512_set1_ps(xr[r][i]);
            for (int v = 0; v < nv; ++v) acc[nv*r + v] = _mm512_fmadd_ps(qv[v], b, acc[nv*r + v]);
        }
    }
    const __m512 zero = _mm512_setzero_ps();
    __m512 wv[nv];
    for (int v = 0; v < nv; ++v) wv[v] = _mm512_loadu_ps(w + 16*v);
    __m512 s[8];
    for (int r = 0; r < 8; ++r) {
        s[r] = _mm512_mul_ps(_mm512_max_ps(acc[nv*r], zero), wv[0]);
        for (int v = 1; v < nv; ++v) s[r] = _mm512_fmadd_ps(_mm512_max_ps(acc[nv*r + v], zero), wv[v], s[r]);
    }
    const __m256 sums = _mm256_set_m128(idx512_hsum4(s[4], s[5], s[6], s[7]), idx512_hsum4(s[0], s[1], s[2], s[3]));
    _mm256_storeu_ps(score, _mm256_add_ps(_mm256_loadu_ps(score), sums));
}

// the scores of the unit of rows in kf for one token, added to what the mask put there
inline void idx512_unit(const idx512_plan & p, const float * kf, const float * qt, const float * wt, float * score) {
    for (int c = 0; c < p.n_chunk; ++c) {
        const float * qc = qt + (size_t) c*p.d*32;
        const float * wc = wt + 32*c;
        if (p.n_head - 32*c > 16) {
            for (int r = 0; r < k_unit; r += 8) idx512_rows8<2>(p.d, kf + (size_t) r*p.d, qc, wc, score + r);
        } else {
            for (int r = 0; r < k_unit; r += 8) idx512_rows8<1>(p.d, kf + (size_t) r*p.d, qc, wc, score + r);
        }
    }
}

// The mask of the rows ik .. ik+31 of token iq into s. False when it hides them all: their scores are -inf
// whatever the keys hold, so nothing has to be multiplied.
inline bool idx512_mask_unit(const struct ggml_tensor * m, int iq, int ik, float * s) {
    const char * row = (const char *) m->data + (size_t) iq*m->nb[1];
    const __m512 ninf = _mm512_set1_ps(-INFINITY);
    if (m->type == GGML_TYPE_I32) {
        const int n_visible = *(const int32_t *) row;
        if (ik >= n_visible) {
            _mm512_storeu_ps(s, ninf);
            _mm512_storeu_ps(s + 16, ninf);
            return false;
        }
        const __m512i first = _mm512_add_epi32(_mm512_set1_epi32(ik), _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
        const __m512i nv    = _mm512_set1_epi32(n_visible);
        _mm512_storeu_ps(s,      _mm512_mask_blend_ps(_mm512_cmplt_epi32_mask(first, nv), ninf, _mm512_setzero_ps()));
        _mm512_storeu_ps(s + 16, _mm512_mask_blend_ps(_mm512_cmplt_epi32_mask(_mm512_add_epi32(first, _mm512_set1_epi32(16)), nv),
                    ninf, _mm512_setzero_ps()));
        return true;
    }
    __m512 a, b;
    if (m->type == GGML_TYPE_F32) {
        a = _mm512_loadu_ps((const float *) row + ik);
        b = _mm512_loadu_ps((const float *) row + ik + 16);
    } else {
        a = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const ggml_fp16_t *) row + ik)));
        b = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const ggml_fp16_t *) row + ik + 16)));
    }
    _mm512_storeu_ps(s, a);
    _mm512_storeu_ps(s + 16, b);
    return (_mm512_cmp_ps_mask(a, ninf, _CMP_EQ_OQ) & _mm512_cmp_ps_mask(b, ninf, _CMP_EQ_OQ)) != 0xffff;
}

// how many of the n keys are at least t; n is a multiple of 16
inline uint32_t idx512_count_ge(const uint32_t * keys, int n, uint32_t t) {
    const __m512i tv  = _mm512_set1_epi32((int) t);
    const __m512i one = _mm512_set1_epi32(1);
    __m512i acc = _mm512_setzero_si512();
    for (int j = 0; j < n; j += 16) {
        const __mmask16 ge = _mm512_cmpge_epu32_mask(_mm512_loadu_si512((const void *)(keys + j)), tv);
        acc = _mm512_mask_add_epi32(acc, ge, acc, one);
    }
    return (uint32_t) _mm512_reduce_add_epi32(acc);
}

// The n_top highest of the n scores: their places, in no particular order, into out, which has room for 16
// more. keys and tmp have room for n + 16. With equal scores at the limit, the first ones are taken.
void idx512_select(int n, int n_top, const float * score, int32_t * out, uint32_t * keys, int32_t * tmp) {
    if (n_top >= n) {
        for (int j = 0; j < n; ++j) out[j] = j;
        return;
    }
    // keys in the order of the scores: negative values have their bits turned, the others get the top bit.
    // What is after the n scores, up to a multiple of 16, gets the key 0, which is under every value asked for.
    const int n16 = (n + 15) & ~15;
    const __m512i top = _mm512_set1_epi32((int) 0x80000000u);
    for (int j = 0; j < n16; j += 16) {
        const __mmask16 valid = n - j >= 16 ? (__mmask16) 0xffff : (__mmask16) ((1u << (n - j)) - 1);
        const __m512i x = _mm512_castps_si512(_mm512_maskz_loadu_ps(valid, score + j));
        const __m512i key = _mm512_xor_si512(x, _mm512_or_si512(_mm512_srai_epi32(x, 31), top));
        _mm512_storeu_si512((void *)(keys + j), _mm512_maskz_mov_epi32(valid, key));
    }
    // the key of the n_top-th highest score: the highest value that n_top keys reach
    uint32_t thr = 0;
    for (uint32_t bit = 0x80000000u; bit != 0; bit >>= 1) {
        if (idx512_count_ge(keys, n16, thr | bit) >= (uint32_t) n_top) {
            thr |= bit;
        }
    }
    const uint32_t n_over = thr == 0xffffffffu ? 0 : idx512_count_ge(keys, n16, thr + 1);
    const int      n_at   = n_top - (int) n_over;   // of the keys equal to thr, the first ones are taken
    // the rows over thr, then those at it. The padding is masked out of the latter: thr is its key, 0, when
    // rows that score a NaN with the sign set have to be taken.
    const __m512i tv = _mm512_set1_epi32((int) thr);
    __m512i idx = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    int n_out = 0, n_eq = 0;
    for (int j = 0; j < n16; j += 16) {
        const __mmask16 valid = n - j >= 16 ? (__mmask16) 0xffff : (__mmask16) ((1u << (n - j)) - 1);
        const __m512i key = _mm512_loadu_si512((const void *)(keys + j));
        const __mmask16 over = _mm512_cmpgt_epu32_mask(key, tv);
        _mm512_storeu_si512((void *)(out + n_out), _mm512_maskz_compress_epi32(over, idx));
        n_out += __builtin_popcount((unsigned) over);
        if (n_eq < n_at) {
            const __mmask16 at = _mm512_cmpeq_epu32_mask(key, tv) & valid;
            _mm512_storeu_si512((void *)(tmp + n_eq), _mm512_maskz_compress_epi32(at, idx));
            n_eq += __builtin_popcount((unsigned) at);
        }
        idx = _mm512_add_epi32(idx, _mm512_set1_epi32(16));
    }
    for (int j = 0; j < n_at; ++j) out[n_out + j] = tmp[j];
}

// the highest score of every group of rows
void idx512_group_max(int n_b, int group_size, const float * score, float * group) {
    if (group_size == 8) {
        for (int b = 0; b < n_b; ++b) {
            const __m256 v = _mm256_loadu_ps(score + 8*b);
            __m128 mx = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
            mx = _mm_max_ps(mx, _mm_movehl_ps(mx, mx));
            mx = _mm_max_ss(mx, _mm_movehdup_ps(mx));
            group[b] = _mm_cvtss_f32(mx);
        }
        return;
    }
    for (int b = 0; b < n_b; ++b) {
        const float * s = score + (size_t) b*group_size;
        float mx = s[0];
        for (int i = 1; i < group_size; ++i) mx = std::max(mx, s[i]);
        group[b] = mx;
    }
}

// What is given for token iq once its scores are known. The scores are changed when rows are chosen inside
// the groups.
void idx512_finish(const idx512_plan & p, struct ggml_tensor * dst, int iq, float * score, char * slice) {
    auto keys  = (uint32_t *)(slice + p.off_keys);
    auto tmp   = (int32_t  *)(slice + p.off_tmp);
    auto out   = (int32_t  *)(slice + p.off_out);
    auto ids   = (char *) dst->data + (size_t) iq*dst->nb[1];
    if (p.group_size == 0) {
        idx512_select(p.n_kv, p.n_top, score, out, keys, tmp);
        std::memcpy(ids, out, sizeof(int32_t)*p.n_top);
        return;
    }
    // the groups with the best row; the last group with a row in view is always among them
    auto group = (float *)(slice + p.off_group);
    idx512_group_max(p.n_b, p.group_size, score, group);
    for (int b = p.n_b - 1; b >= 0; --b) {
        if (group[b] > -INFINITY) {
            group[b] = FLT_MAX;
            break;
        }
    }
    idx512_select(p.n_b, p.n_top, group, out, keys, tmp);
    std::memcpy(ids, out, sizeof(int32_t)*p.n_top);
    auto top_k_out = dst->src[5];
    if (!top_k_out) {
        return;
    }
    // then the best rows of these groups
    if (p.n_top < p.n_b) {
        auto kept = (uint32_t *)(slice + p.off_kept);
        std::memset(kept, 0, sizeof(uint32_t)*((p.n_b + 31)/32));
        for (int j = 0; j < p.n_top; ++j) {
            kept[out[j] >> 5] |= 1u << (out[j] & 31);
        }
        for (int b = 0; b < p.n_b; ++b) {
            if ((kept[b >> 5] & (1u << (b & 31))) == 0) {
                float * s = score + (size_t) b*p.group_size;
                for (int i = 0; i < p.group_size; ++i) s[i] = -INFINITY;
            }
        }
    }
    idx512_select(p.n_kv, p.n_top_out, score, out, keys, tmp);
    std::memcpy((char *) top_k_out->data + (size_t) iq*top_k_out->nb[1], out, sizeof(int32_t)*p.n_top_out);
}

}

bool iqk_indexer512_usable(const struct ggml_tensor * dst) {
    idx512_plan p;
    return idx512_make_plan(dst, 1, p);
}

size_t iqk_indexer512_work_size(const struct ggml_tensor * dst, int nth) {
    idx512_plan p;
    if (!idx512_make_plan(dst, nth, p)) {
        return 0;
    }
    return p.shared + p.per_thread*nth;
}

void iqk_indexer512(struct ggml_tensor * dst, void * work_buffer, barrier_t barrier, void * barrier_data, int ith, int nth) {
    idx512_plan p;
    if (!idx512_make_plan(dst, nth, p)) {
        GGML_ABORT("iqk_indexer512 called for a node it cannot compute");
    }
    auto k = dst->src[0];
    auto q = dst->src[1];
    auto w = dst->src[2];
    auto m = dst->src[3];

    auto slice = (char *) work_buffer + p.shared + p.per_thread*ith;
    auto qt    = (float *)(slice + p.off_qt);
    auto wt    = (float *)(slice + p.off_w);
    auto kf    = (float *)(slice + p.off_kf);
    const size_t qt_size = (size_t) p.n_chunk*p.d*32;
    const size_t wt_size = (size_t) p.n_chunk*32;

    if (p.batch) {
        auto score = (float *)(slice + p.off_score);
        for (int i0 = ith; i0 < p.n_tok; i0 += nth*p.tile) {
            int tok[k_tile_max];
            int nt = 0;
            for (int iq = i0; iq < p.n_tok && nt < p.tile; iq += nth) {
                idx512_token(p, (const char *) q->data + (size_t) iq*q->nb[2], q->nb[1],
                        (const float *)((const char *) w->data + (size_t) iq*w->nb[1]), qt + nt*qt_size, wt + nt*wt_size);
                tok[nt++] = iq;
            }
            for (int ik = 0; ik < p.n_kv; ik += k_unit) {
                bool wanted[k_tile_max];
                bool any = false;
                for (int t = 0; t < nt; ++t) {
                    wanted[t] = idx512_mask_unit(m, tok[t], ik, score + (size_t) t*p.n_kv + ik);
                    any = any || wanted[t];
                }
                if (!any) {
                    continue;
                }
                idx512_keys_f32((const char *) k->data + (size_t) ik*k->nb[1], k->nb[1], p.d, kf);
                for (int t = 0; t < nt; ++t) {
                    if (wanted[t]) {
                        idx512_unit(p, kf, qt + t*qt_size, wt + t*wt_size, score + (size_t) t*p.n_kv + ik);
                    }
                }
            }
            for (int t = 0; t < nt; ++t) {
                idx512_finish(p, dst, tok[t], score + (size_t) t*p.n_kv, slice);
            }
        }
        return;
    }

    // fewer tokens than threads: this thread takes a part of the rows, for every token
    auto scores = (float *) work_buffer;
    const int n_unit = p.n_kv/k_unit;
    const int u0 = (int)((int64_t) n_unit*ith/nth);
    const int u1 = (int)((int64_t) n_unit*(ith + 1)/nth);
    for (int iq = 0; iq < p.n_tok && u1 > u0; ++iq) {
        float * score = scores + (size_t) iq*p.n_kv;
        bool have_token = false;
        for (int u = u0; u < u1; ++u) {
            const int ik = u*k_unit;
            if (!idx512_mask_unit(m, iq, ik, score + ik)) {
                continue;
            }
            if (!have_token) {
                idx512_token(p, (const char *) q->data + (size_t) iq*q->nb[2], q->nb[1],
                        (const float *)((const char *) w->data + (size_t) iq*w->nb[1]), qt, wt);
                have_token = true;
            }
            idx512_keys_f32((const char *) k->data + (size_t) ik*k->nb[1], k->nb[1], p.d, kf);
            idx512_unit(p, kf, qt, wt, score + ik);
        }
    }
    barrier(barrier_data);
    // and the tokens are finished one per thread
    for (int iq = ith; iq < p.n_tok; iq += nth) {
        idx512_finish(p, dst, iq, scores + (size_t) iq*p.n_kv, slice);
    }
}

#else

bool iqk_indexer512_usable(const struct ggml_tensor *) {
    return false;
}

size_t iqk_indexer512_work_size(const struct ggml_tensor *, int) {
    return 0;
}

void iqk_indexer512(struct ggml_tensor *, void *, barrier_t, void *, int, int) {
}

#endif
