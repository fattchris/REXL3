// gb-nvcc: --fmad=false
// The full GLM-5.3 (`glm_moe_dsa`) DSA indexer: a per-token key cache (no pools), the key's LayerNorm with bias, interleaved rope on
// the first 64 of the 128 dims, the query rope, the row-invariant score and the deterministic top-k selection.
//
//   key   k[r][t][d] = LN(wk(h)[t], k_norm_w, k_norm_b, 1e-6)[d], then rope on d < 64 (theta 8e6, interleaved pairs)
//   q     qi[r][h][d] = wq_b(q_a_norm)[r][h][d], rope on d < 64
//   score s[r][t]     = sum_h w[r][h] * relu(qi[r][h] . k[t] * 128^-0.5) * 32^-0.5
//   select            top-2048 tokens by (score, then lower position), ascending positions
//
// Row-invariance: a row's key, score and selection depend only on its own inputs; each output element is one fixed-order fma chain.
// The planes are f16 `[blocks][256][128]`; scores f32 `[rows][stride]`; the selection matches `dsa::Selection` (i32 indices, ascending,
// -1 after the count; meta[2r] = 1 for a sparse row).
#include "common.cuh"

namespace {

constexpr int D = 128;
constexpr int H = 32;
constexpr int ROPE = 64;
constexpr int RH = ROPE / 2;
constexpr int PAGE = 256;
constexpr float LN_EPS = 1e-6f;
constexpr float QSCALE = 0.08838834764831845f;  // 128^-0.5
constexpr float WSCALE = 0.17677669529663687f;  // 32^-0.5

typedef __half f16;

__device__ __forceinline__ float to_f32(f16 v) { return __half2float(v); }
__device__ __forceinline__ f16 from_f32(float v) { return __float2half_rn(v); }

// ---------------------------------------------------------------- key: LayerNorm (fixed ascending order) + rope, into the plane

__global__ void __launch_bounds__(128) idx_key_kernel(const f16* __restrict__ kraw, int64_t kraw_stride,
                                                      const f16* __restrict__ lnw, const f16* __restrict__ lnb,
                                                      const float* __restrict__ ct, const float* __restrict__ st,
                                                      int table_stride, const int* __restrict__ row_slot,
                                                      const int* __restrict__ row_pos, const int* __restrict__ table,
                                                      int max_blocks, f16* __restrict__ keys) {
    const int r = blockIdx.x, t = threadIdx.x;
    const int pos = row_pos[r];
    if (pos < 0) return;
    __shared__ float sh[D];
    __shared__ float mean, var;
    sh[t] = to_f32(kraw[(int64_t)r * kraw_stride + t]);
    __syncthreads();
    // the two reductions are one fixed ascending chain each (D = 128, thread 0), so every row's bits are the same wherever it runs
    if (t == 0) {
        float s = 0.0f;
        for (int i = 0; i < D; ++i) s = __fadd_rn(s, sh[i]);
        mean = __fdiv_rn(s, (float)D);
        float v = 0.0f;
        for (int i = 0; i < D; ++i) {
            const float d = __fsub_rn(sh[i], mean);
            v = __fadd_rn(v, __fmul_rn(d, d));
        }
        var = __fdiv_rn(v, (float)D);
    }
    __syncthreads();
    float y = __fadd_rn(__fmul_rn(__fsub_rn(sh[t], mean), rsqrtf(__fadd_rn(var, LN_EPS))), 0.0f);
    y = __fadd_rn(__fmul_rn(y, to_f32(lnw[t])), to_f32(lnb[t]));
    sh[t] = y;
    __syncthreads();
    const int blk = table[(int64_t)row_slot[r] * max_blocks + pos / PAGE];
    f16* dst = keys + ((int64_t)blk * PAGE + pos % PAGE) * D;
    if (t < RH) {
        const float c = __ldg(ct + (int64_t)pos * table_stride + t), s = __ldg(st + (int64_t)pos * table_stride + t);
        const float x0 = sh[2 * t], x1 = sh[2 * t + 1];
        dst[t] = from_f32(__fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s)));
        dst[t + RH] = from_f32(__fadd_rn(__fmul_rn(x0, s), __fmul_rn(x1, c)));
    }
    if (t >= ROPE) dst[t] = from_f32(sh[t]);
}

// ---------------------------------------------------------------- query rope: first ROPE dims of every head

__global__ void __launch_bounds__(256) idx_q_kernel(const f16* __restrict__ q, int64_t q_stride, const float* __restrict__ ct,
                                                    const float* __restrict__ st, int table_stride,
                                                    const int* __restrict__ row_pos, f16* __restrict__ qout) {
    const int r = blockIdx.x;
    const int pos = row_pos[r];
    for (int idx = threadIdx.x; idx < H * RH; idx += blockDim.x) {
        const int h = idx / RH, i = idx % RH;
        f16* dst = qout + (int64_t)r * q_stride + (int64_t)h * D;
        if (pos < 0) {
            dst[i] = from_f32(0.0f);
            dst[i + RH] = from_f32(0.0f);
            continue;
        }
        const float c = __ldg(ct + (int64_t)pos * table_stride + i), s = __ldg(st + (int64_t)pos * table_stride + i);
        const float x0 = to_f32(q[(int64_t)r * q_stride + (int64_t)h * D + 2 * i]);
        const float x1 = to_f32(q[(int64_t)r * q_stride + (int64_t)h * D + 2 * i + 1]);
        dst[i] = from_f32(__fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s)));
        dst[i + RH] = from_f32(__fadd_rn(__fmul_rn(x0, s), __fmul_rn(x1, c)));
    }
    // the un-roped tail of every head is copied as stored
    for (int idx = threadIdx.x; idx < H * (D - ROPE); idx += blockDim.x) {
        const int h = idx / (D - ROPE), d = ROPE + idx % (D - ROPE);
        qout[(int64_t)r * q_stride + (int64_t)h * D + d] = q[(int64_t)r * q_stride + (int64_t)h * D + d];
    }
}

// ---------------------------------------------------------------- score: one warp per key, lane h owns head h's whole 128-dim dot

__global__ void __launch_bounds__(256) idx_score_kernel(const f16* __restrict__ q, int64_t q_stride,
                                                        const f16* __restrict__ w, const f16* __restrict__ keys,
                                                        const int* __restrict__ table, int max_blocks,
                                                        const int* __restrict__ row_slot, const int* __restrict__ row_pos,
                                                        int n_keys_max, int stride, float* __restrict__ scores) {
    const int r = blockIdx.x, tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
    const int pos = row_pos[r];
    float* row = scores + (int64_t)r * stride;
    for (int t = wid; t < n_keys_max; t += blockDim.x / 32) {
        if (t > pos) {
            row[t] = -INFINITY;
            continue;
        }
        const int blk = table[(int64_t)row_slot[r] * max_blocks + t / PAGE];
        const f16* kt = keys + ((int64_t)blk * PAGE + t % PAGE) * D;
        float dot = 0.0f;
        if (lane < H) {
            const f16* qh = q + (int64_t)r * q_stride + (int64_t)lane * D;
#pragma unroll 4
            for (int d = 0; d < D; ++d) dot = __fmaf_rn(to_f32(qh[d]), to_f32(kt[d]), dot);
        }
        // relu and the per-head weight; the sum over heads is a warp reduction in fixed lane order
        float v = (lane < H) ? __fmul_rn(fmaxf(__fmul_rn(dot, QSCALE), 0.0f), to_f32(w[(int64_t)r * H + lane])) : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, off));
        if (lane == 0) row[t] = __fmul_rn(v, WSCALE);
    }
}

// ---------------------------------------------------------------- selection: (score, lower position) top-k, ascending positions
//
// One block per row. Scores are mapped to monotonic i64 keys (IEEE bits ordered, position in the low bits); the k-th key is found by a
// 32-step bisection on the ordered value, then every key above the threshold is written, and ties are filled in position order.
__device__ __forceinline__ unsigned long long order_u(float s, int t) {
    const int b = __float_as_int(s);
    const long long o = (b < 0) ? (long long)(b ^ 0x7FFFFFFF) : (long long)b;
    const long long key = (o << 32) | (long long)(0xFFFFFFFFu - (unsigned)t);
    return (unsigned long long)key ^ (1ULL << 63);
}

__global__ void __launch_bounds__(256) idx_select_kernel(const float* __restrict__ scores, int stride,
                                                         const int* __restrict__ row_pos, int k, int* __restrict__ out,
                                                         int* __restrict__ meta) {
    const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int pos = row_pos[r];
    if (pos < 0) {
        if (tid == 0) meta[2 * r] = 0;
        return;
    }
    const float* row = scores + (int64_t)r * stride;
    const int n = pos + 1;
    const int kk = min(k, n);
    if (tid == 0) meta[2 * r] = n > k ? 1 : 0;
    __shared__ int sw[8];
    __shared__ int stot;
    __shared__ unsigned long long slo, shi;
    if (tid == 0) {
        slo = 0;
        shi = ~0ULL;
    }
    __syncthreads();
    // the k-th largest ordered key: the smallest T with count(key >= T) <= kk (keys are unique, so count(>= T) == kk at the end)
    while (shi - slo > 1) {
        const unsigned long long mid = slo + (shi - slo) / 2;
        int c = 0;
        for (int t = tid; t < n; t += blockDim.x) {
            if (row[t] == -INFINITY) continue;
            c += order_u(row[t], t) >= mid;
        }
        for (int off = 16; off > 0; off >>= 1) c += __shfl_down_sync(0xffffffffu, c, off);
        if (lane == 0) sw[wid] = c;
        __syncthreads();
        if (tid == 0) {
            int sum = 0;
            for (int w = 0; w < blockDim.x / 32; ++w) sum += sw[w];
            stot = sum;
        }
        __syncthreads();
        if (tid == 0) {
            if (stot > kk) slo = mid; else shi = mid;
        }
        __syncthreads();
    }
    const unsigned long long thr = shi;
    // the selected keys are exactly the kk largest: write their positions in ascending order
    if (tid == 0) {
        int cnt = 0;
        for (int t = 0; t < n; ++t) {
            if (row[t] == -INFINITY) continue;
            if (order_u(row[t], t) >= thr) out[(int64_t)r * 2056 + cnt++] = t;
        }
        for (int i = cnt; i < 2056; ++i) out[(int64_t)r * 2056 + i] = -1;
        meta[2 * r + 1] = cnt;
    }
}

}  // namespace

#define GB_EXPORT extern "C" __attribute__((visibility("default")))

GB_EXPORT int gb_glm_idx_key(const void* kraw, long long kraw_stride, const void* lnw, const void* lnb, const void* ct,
                             const void* st, int table_stride, const int* row_slot, const int* row_pos,
                             const int* block_table, int max_blocks, void* keys, int rows, void* stream) {
    if (rows < 0 || max_blocks <= 0 || kraw_stride < D) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    idx_key_kernel<<<rows, 128, 0, gb_stream(stream)>>>((const f16*)kraw, kraw_stride, (const f16*)lnw, (const f16*)lnb,
                                                         (const float*)ct, (const float*)st, table_stride, row_slot, row_pos,
                                                         block_table, max_blocks, (f16*)keys);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_glm_idx_q(const void* q, long long q_stride, const void* ct, const void* st, int table_stride,
                           const int* row_pos, void* qout, int rows, void* stream) {
    if (rows < 0 || q_stride < (long long)H * D) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    idx_q_kernel<<<rows, 256, 0, gb_stream(stream)>>>((const f16*)q, q_stride, (const float*)ct, (const float*)st, table_stride,
                                                       row_pos, (f16*)qout);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_glm_idx_score(const void* q, long long q_stride, const void* w, const void* keys, const int* block_table,
                               int max_blocks, const int* row_slot, const int* row_pos, int n_keys_max, int stride,
                               void* scores, int rows, void* stream) {
    if (rows < 0 || n_keys_max < 1 || stride < n_keys_max || max_blocks <= 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    idx_score_kernel<<<rows, 256, 0, gb_stream(stream)>>>((const f16*)q, q_stride, (const f16*)w, (const f16*)keys,
                                                           block_table, max_blocks, row_slot, row_pos, n_keys_max, stride,
                                                           (float*)scores);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_glm_idx_select(const void* scores, int stride, const int* row_pos, int k, void* out, void* meta, int rows,
                                void* stream) {
    if (rows < 0 || k < 1 || k > 2048 || stride < 1) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    idx_select_kernel<<<rows, 256, 0, gb_stream(stream)>>>((const float*)scores, stride, row_pos, k, (int*)out, (int*)meta);
    return GB_CHECK_LAUNCH();
}
