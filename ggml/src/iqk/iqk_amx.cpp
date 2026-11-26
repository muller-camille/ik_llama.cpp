//
// Matrix products of a batch of tokens with the tiles of Intel AMX, see iqk_amx.h, and the attention of a batch
// of tokens on the rows of their lists (iqk_amx_flash_attn, at the end)
//

#include "iqk_amx.h"

#include "ggml.h"
#include "ggml-quants.h"   // the layouts of the quantized blocks

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__x86_64__) && defined(__linux__) && defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11
#define IQK_AMX_BUILD 1
#else
#define IQK_AMX_BUILD 0
#endif

#if IQK_AMX_BUILD

#include <cpuid.h>
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {

// ----------------------------------------------------------------------------------------------------------
// What follows up to the "target" pragma is compiled for the processor of the build, like the rest of the
// library: it also runs where there is no AVX-512. Nothing of the standard library is used past that pragma, so
// that no inline function compiled there can take the place of its plain copy at link time.
// ----------------------------------------------------------------------------------------------------------

constexpr int  k_min_tokens = 48;        // fewer tokens than this: the kernels of the type itself
constexpr long k_kc_max     = 2048;      // columns of one pass: an A panel of 32 tokens is 128 kB at most
constexpr long k_w_bytes    = 1 << 20;   // converted weights kept at hand between two passes (L2 is 2 MB)

// Why the tiles cannot be used, or nullptr if they can. The second value tells whether the processor has them.
const char * detect(bool & cpu) {
    cpu = false;
    unsigned a, b, c, d;
    if (__get_cpuid_max(0, nullptr) < 7) return "no AMX";
    __cpuid_count(7, 0, a, b, c, d);
    const bool avx512 = (b >> 16 & 1) && (b >> 17 & 1) && (b >> 30 & 1) && (b >> 31 & 1);   // F, DQ, BW, VL
    const bool amx    = (d >> 22 & 1) && (d >> 24 & 1);                                     // AMX-BF16, AMX-TILE
    if (!avx512 || !amx) return "no AMX";
    __cpuid_count(7, 1, a, b, c, d);
    if (!(a >> 5 & 1)) return "no AVX512-BF16";
    cpu = true;
    __cpuid(1, a, b, c, d);
    if (!(c >> 27 & 1)) return "the system does not save the extended registers";
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    // the system saves the SSE, AVX and AVX-512 registers and the two parts of the tile state
    constexpr unsigned k_need = 1u << 1 | 1u << 2 | 1u << 5 | 1u << 6 | 1u << 7 | 1u << 17 | 1u << 18;
    if ((lo & k_need) != k_need) return "the system does not save the AMX tiles (a kernel without AMX support)";
    // Linux gives the tile data to the processes that ask for it: ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA
    if (syscall(SYS_arch_prctl, 0x1023, 18) != 0) return "the system refused the permission to use the AMX tiles";
    return nullptr;
}

// Once, and only on a processor that has the tiles: one line that says whether they are used
bool detect_and_tell() {
    bool cpu;
    const char * why = detect(cpu);
    const char * env = getenv("GGML_AMX");
    const bool off = env && env[0] == '0' && atoi(env) == 0;
    if (cpu) {
        if (why)      fprintf(stderr, "ggml: this processor has AMX-BF16, but %s: the usual kernels are used\n", why);
        else if (off) fprintf(stderr, "ggml: AMX-BF16 is not used for the batches of tokens (GGML_AMX=0)\n");
        else          fprintf(stderr, "ggml: AMX-BF16 tiles for the matrix products of a batch of tokens (GGML_AMX=0 turns them off)\n");
    }
    return !why && !off;
}

std::atomic<int> g_forced{-1};

bool enabled() {
    static const bool on = detect_and_tell();
    const int forced = g_forced.load(std::memory_order_relaxed);
    return on && forced != 0;
}

// The buffers of a thread: the converted weights, a zero-padded copy of the last tokens when they are not 32,
// and the results of a tile product that does not go straight to its place.
struct Buffers {
    char * w = nullptr; size_t w_size = 0;
    char * a = nullptr; size_t a_size = 0;
    char * f = nullptr; size_t f_size = 0;   // all that the attention of a token needs
    ~Buffers() { free(w); free(a); free(f); }
};

Buffers & buffers() {
    thread_local Buffers b;
    return b;
}

char * grow(char *& p, size_t & have, size_t want) {
    if (have < want) {
        free(p);
        want = (want + 4095) & ~(size_t)4095;
        p = (char *)aligned_alloc(4096, want);
        GGML_ASSERT(p);
        have = want;
    }
    return p;
}

std::atomic<int> g_attn_forced{-1};

// The attention of a batch through the tiles: only when asked for
bool attn_enabled() {
    static const bool on = [] {
        const char * env = getenv("GGML_AMX_ATTN");
        const bool want = env && atoi(env) != 0;
        if (want && enabled()) fprintf(stderr, "ggml: AMX-BF16 tiles for the attention of a batch of tokens (GGML_AMX_ATTN=1)\n");
        return want;
    }();
    const int forced = g_attn_forced.load(std::memory_order_relaxed);
    return enabled() && (forced < 0 ? on : forced != 0);
}

// What the weighted sum of a query row is multiplied by: 1 over the sum of its weights, the sink of the row
// counted as the usual kernels count it. M: the largest score of the row, S: the sum of exp(score - M).
__attribute__((noinline))
void attn_factors(int nq, const float * M, const float * S, const float * sinks, float * f) {
    for (int j = 0; j < nq; ++j) {
        float sum = S[j], m = 1;
        if (sinks) {
            if (sinks[j] > M[j]) {
                m   = expf(M[j] - sinks[j]);
                sum = sum*m + 1;
            } else {
                sum += expf(sinks[j] - M[j]);
            }
        }
        f[j] = sum > 0 ? m/sum : 0;
    }
}

// Units [first, first + n) for thread ith of nth: every unit once, and no thread more than one above another
inline void split(long nunit, int ith, int nth, long & first, long & n) {
    first = nunit*ith/nth;
    n     = nunit*(ith + 1)/nth - first;
}

}

#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512dq,avx512vl,avx512bf16,amx-tile,amx-bf16")

namespace {

struct alignas(64) TileConfig {
    uint8_t  palette;
    uint8_t  start_row;
    uint8_t  reserved[14];
    uint16_t colsb[16];
    uint8_t  rows[16];
};

// The weights as the tiles take them. A panel is 16 rows of the matrix; for each block of 32 columns it holds
// 16 tile rows of 64 bytes, tile row p being the columns 2p and 2p + 1 of the 16 matrix rows.
// Q8_0_R8: rows [0, nr) of the matrix at A (nr a multiple of 8), blocks [ib0, ib0 + nb), to np panels
// (the panels past the rows there are, are zero).
__attribute__((noinline))
void convert_q8_0_r8(const char * A, long strideA, int nr, int np, int ib0, int nb, char * w) {
    const __m512i even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i odd  = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    const __m512i rep[4] = {
        _mm512_setr_epi32( 0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3),
        _mm512_setr_epi32( 4,  4,  4,  4,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7),
        _mm512_setr_epi32( 8,  8,  8,  8,  9,  9,  9,  9, 10, 10, 10, 10, 11, 11, 11, 11),
        _mm512_setr_epi32(12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15),
    };
    for (int p = 0; p < np; ++p) {
        char * out = w + (size_t)p*nb*1024;
        if (16*p >= nr) {
            memset(out, 0, (size_t)nb*1024);
            continue;
        }
        const block_q8_0_r8 * g0 = (const block_q8_0_r8 *)(A + (size_t)(16*p)*strideA) + ib0;
        const block_q8_0_r8 * g1 = 16*p + 8 < nr ? (const block_q8_0_r8 *)(A + (size_t)(16*p + 8)*strideA) + ib0 : nullptr;
        for (int ib = 0; ib < nb; ++ib) {
            const __m256 d0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)g0[ib].d));
            const __m256 d1 = g1 ? _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)g1[ib].d)) : _mm256_setzero_ps();
            const __m512 d  = _mm512_insertf32x8(_mm512_castps256_ps512(d0), d1, 1);
            const __m512 ds[4] = { _mm512_permutexvar_ps(rep[0], d), _mm512_permutexvar_ps(rep[1], d),
                                   _mm512_permutexvar_ps(rep[2], d), _mm512_permutexvar_ps(rep[3], d) };
            const int8_t * q0 = g0[ib].qs;
            const int8_t * q1 = g1 ? g1[ib].qs : nullptr;
            // 4 columns of the 8 rows of a group are 32 bytes in a row: [row][column]
            for (int q = 0; q < 8; ++q) {
                __m512 f0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(q0 + 32*q))));
                __m512 f1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(q0 + 32*q + 16))));
                __m512 f2 = _mm512_setzero_ps(), f3 = _mm512_setzero_ps();
                if (q1) {
                    f2 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(q1 + 32*q))));
                    f3 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(q1 + 32*q + 16))));
                }
                const __m256i h0 = (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(f0, ds[0]));
                const __m256i h1 = (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(f1, ds[1]));
                const __m256i h2 = (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(f2, ds[2]));
                const __m256i h3 = (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(f3, ds[3]));
                // 32 bits = 2 columns of a row: the even ones of the 16 rows are one tile row, the odd ones the next
                const __m512i x0 = _mm512_inserti64x4(_mm512_castsi256_si512(h0), h1, 1);
                const __m512i x1 = _mm512_inserti64x4(_mm512_castsi256_si512(h2), h3, 1);
                _mm512_storeu_si512((__m512i *)(out + 128*q),      _mm512_permutex2var_epi32(x0, even, x1));
                _mm512_storeu_si512((__m512i *)(out + 128*q + 64), _mm512_permutex2var_epi32(x0, odd,  x1));
            }
            out += 1024;
        }
    }
}

// 32 tokens by 32 rows over nb blocks of 32 columns. a: the tokens, as rows of bf16 a_stride bytes apart, from
// the first column of the pass. w0, w1: the two panels of the rows. c: where the 32 x 32 results are,
// c[token*c_stride + row]; they are added to what is there if add is set.
__attribute__((noinline))
void product_32x32(const char * a, long a_stride, const char * w0, const char * w1, int nb,
                   float * c, long c_stride, bool add) {
    const long cs = c_stride*(long)sizeof(float);
    if (add) {
        _tile_loadd(4, c, cs);
        _tile_loadd(5, c + 16, cs);
        _tile_loadd(6, c + 16*c_stride, cs);
        _tile_loadd(7, c + 16*c_stride + 16, cs);
    } else {
        _tile_zero(4);
        _tile_zero(5);
        _tile_zero(6);
        _tile_zero(7);
    }
    const char * a1 = a + 16*a_stride;
    for (int ib = 0; ib < nb; ++ib) {
        _tile_loadd(0, a  + 64*ib, a_stride);
        _tile_loadd(1, a1 + 64*ib, a_stride);
        _tile_loadd(2, w0 + 1024*(size_t)ib, 64);
        _tile_loadd(3, w1 + 1024*(size_t)ib, 64);
        _tile_dpbf16ps(4, 0, 2);
        _tile_dpbf16ps(5, 0, 3);
        _tile_dpbf16ps(6, 1, 2);
        _tile_dpbf16ps(7, 1, 3);
    }
    _tile_stored(4, c, cs);
    _tile_stored(5, c + 16, cs);
    _tile_stored(6, c + 16*c_stride, cs);
    _tile_stored(7, c + 16*c_stride + 16, cs);
}

// n activations to bf16, rounded to the nearest
__attribute__((noinline))
void to_bf16(const float * x, uint16_t * y, int64_t n) {
    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        _mm512_storeu_si512((__m512i *)(y + i), (__m512i)_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16), _mm512_loadu_ps(x + i)));
    }
    if (i < n) {
        const __mmask16 m0 = (__mmask16)((1u << (n - i < 16 ? n - i : 16)) - 1);
        const __mmask16 m1 = n - i > 16 ? (__mmask16)((1u << (n - i - 16)) - 1) : 0;
        const __m512i h = (__m512i)_mm512_cvtne2ps_pbh(_mm512_maskz_loadu_ps(m1, x + i + 16), _mm512_maskz_loadu_ps(m0, x + i));
        _mm512_mask_storeu_epi16(y + i, (__mmask32)m1 << 16 | m0, h);
    }
}

// Rows [x0, x1) of the matrix by tokens [t0, t1). x0 is a multiple of 32, and so is t0.
void product(long x0, long x1, long t0, long t1, long ne00, const char * A, long strideA,
             const char * B, long strideB, float * C, long stride_C) {
    if (x0 >= x1 || t0 >= t1) return;
    // the columns in passes of kc, the rows in blocks whose converted weights stay in the L2 cache
    const long npass = (ne00 + k_kc_max - 1)/k_kc_max;
    const long kc    = ((ne00 + npass - 1)/npass + 31) & ~31L;
    const long rb    = (k_w_bytes/(2*kc)) & ~31L;
    // blocks of the same size: the activations go by once per block, whatever its number of rows
    const long nblk  = (x1 - x0 + (rb < 32 ? 32 : rb) - 1)/(rb < 32 ? 32 : rb);
    const long rows  = ((x1 - x0 + nblk - 1)/nblk + 31) & ~31L;

    auto & buf = buffers();
    char * w   = grow(buf.w, buf.w_size, (size_t)rows*kc*2);
    char * pad = (t1 - t0)%32 ? grow(buf.a, buf.a_size, (size_t)32*kc*2) : nullptr;
    alignas(64) float tmp[32*32];

    for (long xb = x0; xb < x1; xb += rows) {
        const int nr = (int)(x1 - xb < rows ? x1 - xb : rows);
        const int nu = (nr + 31)/32;
        for (long k0 = 0; k0 < ne00; k0 += kc) {
            const long kn  = ne00 - k0 < kc ? ne00 - k0 : kc;
            const int  nb  = (int)(kn/32);
            const bool add = k0 > 0;
            convert_q8_0_r8(A + (size_t)xb*strideA, strideA, nr, 2*nu, (int)(k0/32), nb, w);
            for (long t = t0; t < t1; t += 32) {
                const int    nt = (int)(t1 - t < 32 ? t1 - t : 32);
                const char * a  = B + (size_t)t*strideB + 2*k0;
                long a_stride   = strideB;
                if (nt < 32) {
                    for (int i = 0; i < nt; ++i) memcpy(pad + (size_t)i*2*kn, a + (size_t)i*strideB, 2*kn);
                    memset(pad + (size_t)nt*2*kn, 0, (size_t)(32 - nt)*2*kn);
                    a = pad; a_stride = 2*kn;
                }
                for (int u = 0; u < nu; ++u) {
                    const int    nx = nr - 32*u < 32 ? nr - 32*u : 32;
                    const char * w0 = w + (size_t)(2*u)*nb*1024;
                    const char * w1 = w0 + (size_t)nb*1024;
                    float * c = C + (size_t)t*stride_C + xb + 32*u;
                    if (nt == 32 && nx == 32) {
                        product_32x32(a, a_stride, w0, w1, nb, c, stride_C, add);
                    } else {
                        // the results a thread may write are fewer than 32 x 32: through a copy
                        if (add) {
                            memset(tmp, 0, sizeof(tmp));
                            for (int i = 0; i < nt; ++i) memcpy(tmp + 32*i, c + (size_t)i*stride_C, nx*sizeof(float));
                        }
                        product_32x32(a, a_stride, w0, w1, nb, tmp, 32, add);
                        for (int i = 0; i < nt; ++i) memcpy(c + (size_t)i*stride_C, tmp + 32*i, nx*sizeof(float));
                    }
                }
            }
        }
    }
}

}

bool iqk_amx_mul_mat(long Nx, long Ny, long ne00, int typeA, const void * A, long strideA,
                     const void * B, long strideB, float * C, long stride_C, int ith, int nth) {
    if (!iqk_amx_wanted(typeA, Nx, Ny, ne00)) return false;

    // The work is cut in units of 32 rows, and when there are few of them for the threads, the tokens are cut
    // too: a thread then has whole units, and at each end of its range a unit for some of the token slices only.
    const long nux    = (Nx + 31)/32;
    const long nchunk = (Ny + 31)/32;
    long nty = nux >= 4L*nth ? 1 : (4L*nth + nux - 1)/nux;
    if (nty > nchunk) nty = nchunk;
    long first, n;
    split(nux*nty, ith, nth, first, n);
    if (n <= 0) return true;
    const long last = first + n;

    TileConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.palette = 1;
    for (int i = 0; i < 8; ++i) { cfg.rows[i] = 16; cfg.colsb[i] = 64; }
    _tile_loadconfig(&cfg);

    auto rows_of   = [Nx] (long u) { return u*32 < Nx ? u*32 : Nx; };
    auto tokens_of = [Ny, nchunk, nty] (long s) { const long t = 32*(nchunk*s/nty); return t < Ny ? t : Ny; };
    auto run = [&] (long u0, long u1, long s0, long s1) {
        product(rows_of(u0), rows_of(u1), tokens_of(s0), tokens_of(s1), ne00,
                (const char *)A, strideA, (const char *)B, strideB, C, stride_C);
    };
    const long u_first = first/nty, s_first = first%nty;
    const long u_last  = last/nty,  s_last  = last%nty;
    if (u_first == u_last) {
        run(u_first, u_first + 1, s_first, s_last);
    } else {
        long u = u_first;
        if (s_first > 0) { run(u, u + 1, s_first, nty); ++u; }
        run(u, u_last, 0, nty);
        if (s_last > 0) run(u_last, u_last + 1, 0, s_last);
    }

    _tile_release();
    return true;
}

void iqk_amx_from_float(const float * x, void * y, int64_t n) {
    to_bf16(x, (uint16_t *)y, n);
}

// ----------------------------------------------------------------------------------------------------------
// The attention of a token of a batch on the rows of its list
//
// A tile product is C[m][n] += sum over k of A[m][2k]*B[k][2n] + A[m][2k+1]*B[k][2n+1], 16 x 32 bf16 values by
// 32 x 16 into 16 x 16 floats. Two of them make the attention:
//   scores[row][query]  = K[row] . Q[query]         A: 16 rows of K, 32 values each
//                                                   B: the same 32 values of 16 query rows, two values per query
//   sum[query][value]   = weight[query][row] * V    A: the weights of 16 query rows for 32 rows of V
//                                                   B: 16 values of these 32 rows, two rows per value
// K and V are the same rows, so one pass over the rows of the list writes both forms; the softmax in between runs
// on 16 query rows at a time.
// ----------------------------------------------------------------------------------------------------------

namespace {

alignas(64) const uint16_t k_pair_lo[32] = {  0, 32,  1, 33,  2, 34,  3, 35,  4, 36,  5, 37,  6, 38,  7, 39,
                                              8, 40,  9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47 };
alignas(64) const uint16_t k_pair_hi[32] = { 16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55,
                                             24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63 };

// exp(x) for x <= 0, the polynomial of v_expf in iqk_utils.h. Below -87 the result would be a denormal: -87 is
// taken, a weight of 1.6e-38 next to the 1 of the row with the largest score.
inline __m512 attn_exp(__m512 x) {
    x = _mm512_max_ps(_mm512_set1_ps(-87.f), x);
    const __m512 r = _mm512_set1_ps(0x1.8p23f);
    const __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
    const __m512 n = _mm512_sub_ps(z, r);
    const __m512 b = _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f),
                                      _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
    const __m512 u = _mm512_mul_ps(b, b);
    const __m512 j = _mm512_fmadd_ps(
        _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b, _mm512_set1_ps(0x1.573e2ep-5f)), u,
                        _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b, _mm512_set1_ps(0x1.fffdb6p-2f))), u,
        _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b, _mm512_set1_ps(1.0f)));
    return _mm512_scalef_ps(j, n);
}

// 32 rows of the cache, nb blocks of 32 f16 values each (nullptr past the first: a row of zeros), as the two
// products take them:
//   kt: 2*nb tiles, the tile of rows 0-15 for each block, then that of rows 16-31; a tile row is 32 values of a row
//   vt: 2*nb tiles, one per 16 values; tile row p is these 16 values of rows 2p and 2p + 1, two rows per value
__attribute__((noinline))
void attn_convert(const uint16_t * const * rows, int nb, char * kt, char * vt) {
    const __m512i lo = _mm512_load_si512((const __m512i *)k_pair_lo);
    const __m512i hi = _mm512_load_si512((const __m512i *)k_pair_hi);
    for (int p = 0; p < 16; ++p) {
        if (p + 1 < 16) {
            for (int i = 2; i < 4; ++i) {
                const char * r = (const char *)rows[2*p + i];
                if (r) for (int o = 0; o < 64*nb; o += 64) _mm_prefetch(r + o, _MM_HINT_T0);
            }
        }
        // a masked load reads nothing where its mask is 0: for a row that is not there, the address of the first
        // one, which always is
        const __mmask16   m0 = rows[2*p]     ? 0xffff : 0, m1 = rows[2*p + 1] ? 0xffff : 0;
        const uint16_t  * r0 = rows[2*p]     ? rows[2*p]     : rows[0];
        const uint16_t  * r1 = rows[2*p + 1] ? rows[2*p + 1] : rows[0];
        char * k = kt + (size_t)(p/8)*nb*1024 + (2*p%16)*64;
        char * v = vt + p*64;
        for (int c = 0; c < nb; ++c) {
            const __m512i x0 = (__m512i)_mm512_cvtne2ps_pbh(
                    _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(m0, r0 + 32*c + 16)),
                    _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(m0, r0 + 32*c)));
            const __m512i x1 = (__m512i)_mm512_cvtne2ps_pbh(
                    _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(m1, r1 + 32*c + 16)),
                    _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(m1, r1 + 32*c)));
            _mm512_storeu_si512((__m512i *)(k + (size_t)c*1024),      x0);
            _mm512_storeu_si512((__m512i *)(k + (size_t)c*1024 + 64), x1);
            _mm512_storeu_si512((__m512i *)(v + (size_t)(2*c)*1024),     _mm512_permutex2var_epi16(x0, lo, x1));
            _mm512_storeu_si512((__m512i *)(v + (size_t)(2*c + 1)*1024), _mm512_permutex2var_epi16(x0, hi, x1));
        }
    }
}

// The query rows as tiles: for 16 query rows and each block of 32 values, tile row k is the values 2k and 2k + 1
// of the 16 rows. qb: the rows as bf16, D values each, their number a multiple of 16.
__attribute__((noinline))
void attn_query_tiles(const uint16_t * qb, int D, int nqb, char * qt) {
    const int nb = D/32;
    const __m512i vi = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                                          _mm512_set1_epi32(D/2));
    for (int b = 0; b < nqb; ++b) {
        const int * base = (const int *)(qb + (size_t)16*b*D);
        for (int c = 0; c < nb; ++c) for (int k = 0; k < 16; ++k) {
            _mm512_storeu_si512((__m512i *)(qt + (((size_t)b*nb + c)*16 + k)*64), _mm512_i32gather_epi32(vi, base + 16*c + k, 4));
        }
    }
}

// The products of 32 rows with all the query rows: st[row*stride + query], stride in floats
__attribute__((noinline))
void attn_scores(const char * kt, const char * qt, int nb, int nqb, float * st, long stride) {
    const long sb = stride*(long)sizeof(float);
    for (int b = 0; b < nqb; b += 2) {
        const bool   two = b + 1 < nqb;
        const char * q0  = qt + (size_t)b*nb*1024;
        const char * q1  = q0 + (size_t)nb*1024;
        _tile_zero(4);
        _tile_zero(5);
        _tile_zero(6);
        _tile_zero(7);
        for (int c = 0; c < nb; ++c) {
            _tile_loadd(0, kt + (size_t)c*1024, 64);
            _tile_loadd(1, kt + (size_t)(nb + c)*1024, 64);
            _tile_loadd(2, q0 + (size_t)c*1024, 64);
            _tile_dpbf16ps(4, 0, 2);
            _tile_dpbf16ps(6, 1, 2);
            if (two) {
                _tile_loadd(3, q1 + (size_t)c*1024, 64);
                _tile_dpbf16ps(5, 0, 3);
                _tile_dpbf16ps(7, 1, 3);
            }
        }
        float * s = st + 16*b;
        _tile_stored(4, s, sb);
        _tile_stored(6, s + 16*stride, sb);
        if (two) {
            _tile_stored(5, s + 16, sb);
            _tile_stored(7, s + 16*stride + 16, sb);
        }
    }
}

// The weights exp(score - largest score of the query row) of the first nk rows, rounded to bf16, the other way
// round: pb[query*stride + 2*row], stride in bytes, zero for the rows from nk to 16*ng. S gets their sums.
__attribute__((noinline))
void attn_weights(const float * st, long stride, int nk, int ng, int nqb, const float * M, float * S, char * pb, long pb_stride) {
    for (int g = 0; g < ng; ++g) for (int b = 0; b < nqb; ++b) {
        const __m512 vm  = _mm512_loadu_ps(M + 16*b);
        __m512       sum = _mm512_loadu_ps(S + 16*b);
        __m256i r[16], a[16];
        for (int i = 0; i < 16; ++i) {
            const int row = 16*g + i;
            if (row < nk) {
                const __m512  e = attn_exp(_mm512_sub_ps(_mm512_loadu_ps(st + (size_t)row*stride + 16*b), vm));
                const __m256i h = (__m256i)_mm512_cvtneps_pbh(e);
                // the sum is that of the weights the tiles see
                sum  = _mm512_add_ps(sum, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16)));
                r[i] = h;
            } else {
                r[i] = _mm256_setzero_si256();
            }
        }
        _mm512_storeu_ps(S + 16*b, sum);
        // r[row] holds 16 query rows: turned into 16 query rows of 16 rows each
        for (int i = 0; i < 8; ++i) {
            a[2*i]     = _mm256_unpacklo_epi16(r[2*i], r[2*i + 1]);
            a[2*i + 1] = _mm256_unpackhi_epi16(r[2*i], r[2*i + 1]);
        }
        for (int i = 0; i < 4; ++i) {
            r[4*i]     = _mm256_unpacklo_epi32(a[4*i],     a[4*i + 2]);
            r[4*i + 1] = _mm256_unpackhi_epi32(a[4*i],     a[4*i + 2]);
            r[4*i + 2] = _mm256_unpacklo_epi32(a[4*i + 1], a[4*i + 3]);
            r[4*i + 3] = _mm256_unpackhi_epi32(a[4*i + 1], a[4*i + 3]);
        }
        for (int i = 0; i < 2; ++i) for (int k = 0; k < 4; ++k) {
            a[8*i + 2*k]     = _mm256_unpacklo_epi64(r[8*i + k], r[8*i + 4 + k]);
            a[8*i + 2*k + 1] = _mm256_unpackhi_epi64(r[8*i + k], r[8*i + 4 + k]);
        }
        char * p = pb + (size_t)16*b*pb_stride + 32*g;
        for (int j = 0; j < 8; ++j) {
            _mm256_storeu_si256((__m256i *)(p + (size_t)j*pb_stride),       _mm256_permute2x128_si256(a[j], a[8 + j], 0x20));
            _mm256_storeu_si256((__m256i *)(p + (size_t)(j + 8)*pb_stride), _mm256_permute2x128_si256(a[j], a[8 + j], 0x31));
        }
    }
}

// The sums weight * V over the nkb blocks of 32 rows, each query row multiplied by its factor on the way out
__attribute__((noinline))
void attn_values(const char * pb, long pb_stride, const char * vt, int nkb, int nb16, int nq, const float * f,
                 float * out, long stride_out) {
    alignas(64) float tmp[16*16];
    const int nqb = (nq + 15)/16;
    auto put = [&] (int q0, int d) {
        const int nr = nq - q0 < 16 ? nq - q0 : 16;
        for (int i = 0; i < nr; ++i) {
            float * o = (float *)((char *)out + (size_t)(q0 + i)*stride_out) + 16*d;
            _mm512_storeu_ps(o, _mm512_mul_ps(_mm512_set1_ps(f[q0 + i]), _mm512_load_ps(tmp + 16*i)));
        }
    };
    for (int b = 0; b < nqb; b += 2) {
        const bool   two = b + 1 < nqb;
        const char * p0  = pb + (size_t)16*b*pb_stride;
        const char * p1  = p0 + (size_t)16*pb_stride;
        for (int d = 0; d < nb16; d += 2) {
            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);
            for (int kb = 0; kb < nkb; ++kb) {
                const char * v = vt + ((size_t)kb*nb16 + d)*1024;
                _tile_loadd(0, p0 + 64*kb, pb_stride);
                _tile_loadd(2, v, 64);
                _tile_loadd(3, v + 1024, 64);
                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(5, 0, 3);
                if (two) {
                    _tile_loadd(1, p1 + 64*kb, pb_stride);
                    _tile_dpbf16ps(6, 1, 2);
                    _tile_dpbf16ps(7, 1, 3);
                }
            }
            _tile_stored(4, tmp, 64); put(16*b, d);
            _tile_stored(5, tmp, 64); put(16*b, d + 1);
            if (two) {
                _tile_stored(6, tmp, 64); put(16*b + 16, d);
                _tile_stored(7, tmp, 64); put(16*b + 16, d + 1);
            }
        }
    }
}

}

void iqk_amx_flash_attn(int D, int nq, int n, const float * q, long stride_q, const void * kv, long stride_kv,
                        const int * idx, const uint16_t * mask, const float * sinks, float scale,
                        float * out, long stride_out) {
    const int nb   = D/32;
    const int nqb  = (nq + 15)/16, nq16 = 16*nqb;
    const int n32  = (n + 31) & ~31;

    // the buffers of this thread, all in one piece
    const size_t size_rows = ((size_t)n32*sizeof(void *) + 63) & ~(size_t)63;
    const size_t size_q    = (size_t)nq16*D*2;          // the query rows as bf16, then as tiles
    const size_t size_kt   = (size_t)2*nb*1024;         // K of the 32 rows at hand
    const size_t size_vt   = (size_t)(n32/32)*2*nb*1024;
    const size_t size_st   = (size_t)n32*nq16*sizeof(float);
    const size_t size_pb   = (size_t)nq16*n32*2;
    const size_t size_msf  = (size_t)3*nq16*sizeof(float);
    auto & buf = buffers();
    char * mem = grow(buf.f, buf.f_size, size_rows + 2*size_q + size_kt + size_vt + size_st + size_pb + size_msf);
    auto rows = (const uint16_t **)mem;  mem += size_rows;
    auto qb   = (uint16_t *)mem;         mem += size_q;
    char * qt = mem;                     mem += size_q;
    char * kt = mem;                     mem += size_kt;
    char * vt = mem;                     mem += size_vt;
    auto st   = (float *)mem;            mem += size_st;
    char * pb = mem;                     mem += size_pb;
    auto M    = (float *)mem;
    float * S = M + nq16, * f = S + nq16;

    // the rows of the list that are seen, one after the other
    int  nk = 0;
    bool listed = false;
    for (int j = 0; j < n; ++j) {
        if (idx[j] < 0) continue;
        listed = true;
        if (mask && mask[idx[j]] != 0) continue;
        rows[nk++] = (const uint16_t *)((const char *)kv + (size_t)idx[j]*stride_kv);
    }
    if (!listed) return;
    if (nk == 0) {
        for (int j = 0; j < nq; ++j) memset((char *)out + (size_t)j*stride_out, 0, D*sizeof(float));
        return;
    }
    const int nkb = (nk + 31)/32;
    for (int j = nk; j < 32*nkb; ++j) rows[j] = nullptr;

    for (int j = 0; j < nq; ++j) to_bf16((const float *)((const char *)q + (size_t)j*stride_q), qb + (size_t)j*D, D);
    if (nq16 > nq) memset(qb + (size_t)nq*D, 0, (size_t)(nq16 - nq)*D*2);
    attn_query_tiles(qb, D, nqb, qt);

    TileConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.palette = 1;
    for (int i = 0; i < 8; ++i) { cfg.rows[i] = 16; cfg.colsb[i] = 64; }
    _tile_loadconfig(&cfg);

    const __m512 vscale = _mm512_set1_ps(scale);
    for (int b = 0; b < nqb; ++b) {
        _mm512_storeu_ps(M + 16*b, _mm512_set1_ps(-INFINITY));
        _mm512_storeu_ps(S + 16*b, _mm512_setzero_ps());
    }
    for (int kb = 0; kb < nkb; ++kb) {
        float * s = st + (size_t)32*kb*nq16;
        attn_convert(rows + 32*kb, nb, kt, vt + (size_t)kb*2*nb*1024);
        attn_scores(kt, qt, nb, nqb, s, nq16);
        // the scores of the rows that are there, and the largest one of each query row
        const int nr = nk - 32*kb < 32 ? nk - 32*kb : 32;
        for (int b = 0; b < nqb; ++b) {
            __m512 vm = _mm512_loadu_ps(M + 16*b);
            for (int i = 0; i < nr; ++i) {
                float * si = s + (size_t)i*nq16 + 16*b;
                const __m512 v = _mm512_mul_ps(vscale, _mm512_loadu_ps(si));
                _mm512_storeu_ps(si, v);
                vm = _mm512_max_ps(vm, v);
            }
            _mm512_storeu_ps(M + 16*b, vm);
        }
    }
    attn_weights(st, nq16, nk, 2*nkb, nqb, M, S, pb, 64L*nkb);
    attn_factors(nq, M, S, sinks, f);
    attn_values(pb, 64L*nkb, vt, nkb, 2*nb, nq, f, out, stride_out);

    _tile_release();
}

#pragma GCC pop_options

int iqk_amx_bf16(void) {
    return enabled();
}

void iqk_amx_enable(int on) {
    g_forced.store(on ? 1 : 0, std::memory_order_relaxed);
}

int iqk_amx_attn(void) {
    return attn_enabled();
}

void iqk_amx_attn_enable(int on) {
    g_attn_forced.store(on < 0 ? -1 : on ? 1 : 0, std::memory_order_relaxed);
}

int iqk_amx_attn_fits(int type_k, int type_v, int same_rows, int Dk, int Dv, float softcap) {
    return type_k == GGML_TYPE_F16 && type_v == GGML_TYPE_F16 && same_rows && Dk == Dv && Dk >= 32 && Dk%32 == 0 && !(softcap > 0);
}

int iqk_amx_wanted(int typeA, long nx, long ny, long ne00) {
    if (typeA != GGML_TYPE_Q8_0_R8) return 0;
    if (ny < k_min_tokens || nx < 32 || nx%8 != 0 || ne00 < 64 || ne00%32 != 0) return 0;
    return enabled();
}

#else

int  iqk_amx_bf16(void) { return 0; }
int  iqk_amx_wanted(int, long, long, long) { return 0; }
void iqk_amx_enable(int) {}
void iqk_amx_from_float(const float *, void *, int64_t) {}
bool iqk_amx_mul_mat(long, long, long, int, const void *, long, const void *, long, float *, long, int, int) { return false; }
int  iqk_amx_attn(void) { return 0; }
void iqk_amx_attn_enable(int) {}
int  iqk_amx_attn_fits(int, int, int, int, int, float) { return 0; }
void iqk_amx_flash_attn(int, int, int, const float *, long, const void *, long, const int *, const uint16_t *, const float *,
                        float, float *, long) {}

#endif
