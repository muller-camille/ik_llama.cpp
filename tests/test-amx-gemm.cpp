// The matrix products of a batch of tokens through the tiles of Intel AMX (ggml/src/iqk/iqk_amx.cpp) against
// the same products written out in double precision, and against what the usual kernels give: the tiles must
// not be further from the exact result than they are.
//
//   test-amx-gemm --bench [tokens [threads [repeats [values rows [groups]]]]]
//       times both ways the products of one layer of DeepSeek-V4.1 Flash, or of the matrix given

#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" {
int  iqk_amx_bf16(void);
void iqk_amx_enable(int on);
}

static int n_fail = 0;

static std::mt19937 rng(97);

// 8 rows of Q8_0, interleaved 4 columns at a time
struct q8_0_r8 {
    ggml_fp16_t d[8];
    int8_t      qs[256];
};

struct gemm_case {
    int ne00;   // values per row
    int nx;     // rows of the matrix
    int ny;     // tokens
    int ng;     // matrices side by side (the third dimension), each with its own activations
    int n_threads;
};

struct gemm_data {
    ggml_context * ctx = nullptr;
    ggml_tensor  * w = nullptr, * x = nullptr;
    std::vector<float> wf;   // the weights as floats, [group][row][column]
};

static gemm_data build(const gemm_case & c, bool exact) {
    gemm_data s;
    const size_t w_bytes = (size_t) c.ne00/32*34*c.nx*c.ng;
    ggml_init_params ip = { (size_t) 64 << 20, nullptr, false };
    ip.mem_size += w_bytes + (size_t) c.ne00*c.ny*c.ng*4*4 + (size_t) c.nx*c.ny*c.ng*4*2;
    s.ctx = ggml_init(ip);
    s.w = c.ng > 1 ? ggml_new_tensor_3d(s.ctx, GGML_TYPE_Q8_0_R8, c.ne00, c.nx, c.ng)
                   : ggml_new_tensor_2d(s.ctx, GGML_TYPE_Q8_0_R8, c.ne00, c.nx);
    s.x = c.ng > 1 ? ggml_new_tensor_3d(s.ctx, GGML_TYPE_F32, c.ne00, c.ny, c.ng)
                   : ggml_new_tensor_2d(s.ctx, GGML_TYPE_F32, c.ne00, c.ny);
    if (ggml_nbytes(s.w) != w_bytes) { printf("unexpected size of a Q8_0_R8 tensor\n"); exit(1); }

    std::uniform_int_distribution<int> quant(-127, 127);
    std::uniform_real_distribution<float> scale(0.0005f, 0.004f);
    const int nb = c.ne00/32;
    if (exact) s.wf.resize((size_t) c.ne00*c.nx*c.ng);
    auto blocks = (q8_0_r8 *) s.w->data;
    for (int g = 0; g < c.ng; ++g) for (int r8 = 0; r8 < c.nx/8; ++r8) for (int b = 0; b < nb; ++b) {
        auto & blk = blocks[((size_t) g*(c.nx/8) + r8)*nb + b];
        float d[8];
        for (int k = 0; k < 8; ++k) {
            blk.d[k] = ggml_fp32_to_fp16(scale(rng));
            d[k] = ggml_fp16_to_fp32(blk.d[k]);
        }
        for (int i = 0; i < 256; ++i) blk.qs[i] = (int8_t) quant(rng);
        if (exact) for (int k = 0; k < 8; ++k) for (int q = 0; q < 8; ++q) for (int i = 0; i < 4; ++i) {
            s.wf[((size_t) g*c.nx + 8*r8 + k)*c.ne00 + 32*b + 4*q + i] = d[k]*blk.qs[32*q + 4*k + i];
        }
    }
    // activations like those of a model: a spread of magnitudes inside a row, and a few large values
    std::normal_distribution<float> gauss(0.f, 1.f);
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    auto xf = (float *) s.x->data;
    for (size_t i = 0; i < (size_t) c.ne00*c.ny*c.ng; ++i) {
        float v = gauss(rng);
        if (u01(rng) < 0.01f) v *= 8;
        xf[i] = v*0.3f;
    }
    return s;
}

static std::vector<float> run(const gemm_case & c, gemm_data & s, int amx, double * seconds = nullptr, int repeats = 1) {
    iqk_amx_enable(amx);
    ggml_tensor * y  = ggml_mul_mat(s.ctx, s.w, s.x);
    ggml_cgraph * gf = ggml_new_graph(s.ctx);
    ggml_build_forward_expand(gf, y);
    double best = 1e30;
    for (int r = 0; r < repeats; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        ggml_graph_compute_with_ctx(s.ctx, gf, c.n_threads);
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    if (seconds) *seconds = best;
    return std::vector<float>((const float *) y->data, (const float *) y->data + ggml_nelements(y));
}

// the distance to the exact result, relative to its size: root of the mean square, and the largest one
static void distance(const gemm_case & c, const gemm_data & s, const std::vector<float> & y, double & rms, double & worst) {
    const float * xf = (const float *) s.x->data;
    double e2 = 0, r2 = 0, emax = 0;
    for (int g = 0; g < c.ng; ++g) for (int t = 0; t < c.ny; ++t) for (int r = 0; r < c.nx; ++r) {
        const float * w = s.wf.data() + ((size_t) g*c.nx + r)*c.ne00;
        const float * a = xf + ((size_t) g*c.ny + t)*c.ne00;
        double sum = 0;
        for (int j = 0; j < c.ne00; ++j) sum += (double) w[j]*a[j];
        const double e = y[((size_t) g*c.ny + t)*c.nx + r] - sum;
        e2 += e*e; r2 += sum*sum;
        emax = std::max(emax, std::fabs(e));
    }
    const double size = std::sqrt(r2/((double) c.ng*c.ny*c.nx));
    rms   = std::sqrt(e2/((double) c.ng*c.ny*c.nx))/size;
    worst = emax/size;
}

static void check(const gemm_case & c) {
    gemm_data s = build(c, true);
    // the usual kernels take a batch 16 rows at a time
    const bool usual = c.nx%16 == 0;
    double rms0 = 0, worst0 = 0, rms1, worst1;
    if (usual) {
        const std::vector<float> y_usual = run(c, s, 0);
        distance(c, s, y_usual, rms0, worst0);
    }
    const std::vector<float> y_tiles = run(c, s, 1);
    distance(c, s, y_tiles, rms1, worst1);
    bool finite = true;
    for (float v : y_tiles) finite = finite && std::isfinite(v);
    // the usual kernels give each block of 32 activations one 8-bit scale; the tiles keep 8 bits of each value
    const bool ok = finite && rms1 < 4e-3 && worst1 < 3e-2 && (!usual || rms1 <= 1.25*rms0 + 1e-4);
    if (!ok) ++n_fail;
    printf("%s %5d x %5d, %4d tokens, %d group(s), %d thread(s): distance to the exact product %.2e (largest %.2e) with the tiles",
           ok ? "ok  " : "FAIL", c.nx, c.ne00, c.ny, c.ng, c.n_threads, rms1, worst1);
    if (usual) printf(", %.2e (largest %.2e) with the usual kernels", rms0, worst0);
    printf("\n");
    ggml_free(s.ctx);
}

static void bench(int ny, int n_threads, int repeats, int ne00 = 0, int nx = 0, int ng = 1) {
    struct mat { const char * name; int ne00, nx, ng; };
    std::vector<mat> mats = {
        { "attn_q_a      5120 ->  1280", 5120,  1280, 1 },
        { "attn_kv       5120 ->   512", 5120,   512, 1 },
        { "attn_q_b      1280 -> 32768", 1280, 32768, 1 },
        { "attn_output_a 8 x (4096 -> 1024)", 4096, 1024, 8 },
        { "attn_output_b 8192 ->  5120", 8192,  5120, 1 },
        { "ffn_down_shexp 2304 -> 5120", 2304,  5120, 1 },
    };
    if (ne00 > 0) mats = { { "the matrix asked for", ne00, nx, ng } };
    printf("%d tokens, %d threads, best of %d\n", ny, n_threads, repeats);
    double total[2] = {0, 0};
    for (auto & m : mats) {
        gemm_case c = { m.ne00, m.nx, ny, m.ng, n_threads };
        gemm_data s = build(c, false);
        double t[2];
        run(c, s, 0, &t[0], repeats);
        run(c, s, 1, &t[1], repeats);
        const double macs = (double) m.ne00*m.nx*m.ng*ny;
        printf("  %-34s usual %8.2f ms (%5.0f G MAC/s per thread)   tiles %8.2f ms (%5.0f G MAC/s per thread)   x %.1f\n", m.name,
               1e3*t[0], macs/t[0]/n_threads/1e9, 1e3*t[1], macs/t[1]/n_threads/1e9, t[0]/t[1]);
        total[0] += t[0]; total[1] += t[1];
        ggml_free(s.ctx);
    }
    printf("  all: usual %.1f ms, tiles %.1f ms, x %.2f\n", 1e3*total[0], 1e3*total[1], total[0]/total[1]);
}

int main(int argc, char ** argv) {
    if (!iqk_amx_bf16()) {
        printf("no AMX-BF16 on this machine, or not for this process: nothing to compare\n");
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        bench(argc > 2 ? atoi(argv[2]) : 512, argc > 3 ? atoi(argv[3]) : 4, argc > 4 ? atoi(argv[4]) : 3,
              argc > 6 ? atoi(argv[5]) : 0, argc > 6 ? atoi(argv[6]) : 0, argc > 7 ? atoi(argv[7]) : 1);
        return 0;
    }
    const gemm_case cases[] = {
        // one pass over the columns, rows and tokens in whole units of 32
        { 1280, 1024,  96, 1, 1 }, { 1280, 1024,  96, 1, 4 },
        // tokens that are not a multiple of 32, of 16
        { 1280,  512,  75, 1, 3 }, { 1280,  512, 100, 1, 4 }, {  512,  256,  49, 1, 2 },
        // rows that are not: 8, 16 and 24 left over
        {  640,   40,  64, 1, 1 }, {  640,  112,  80, 1, 3 }, {  640,  216,  67, 1, 4 },
        // several passes over the columns, the last one shorter
        { 5120,  256,  64, 1, 4 }, { 4096,  128,  70, 1, 2 }, { 2112,   96,  96, 1, 1 },
        // more threads than units of rows: the tokens are shared too
        { 1024,   64, 256, 1, 8 }, { 1024,   96, 200, 1, 7 }, {  512,   32, 131, 1, 4 },
        // matrices side by side, as attn_output_a
        { 2048,  256,  96, 4, 4 }, { 1024,  128,  77, 8, 8 }, {  512,   64,  64, 3, 5 },
        // the smallest product that goes through the tiles
        {   64,   32,  48, 1, 1 },
    };
    for (const auto & c : cases) check(c);
    printf(n_fail ? "%d case(s) failed\n" : "all cases passed\n", n_fail);
    return n_fail ? 1 : 0;
}
