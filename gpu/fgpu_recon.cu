// CUDA reconstruction stage. Must be built with --fmad=false: contracting the predictor or
// the dequantization into FMA changes results relative to the CPU decoder.
#include "fgpu_recon.h"

#include <cuda_runtime.h>
#include <zstd.h>
#define FSE_STATIC_LINKING_ONLY
#include "Falcons/encoder/zfse/fse.h"
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <chrono>
#include <map>
#include <mutex>
#include <unordered_map>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <algorithm>

namespace fgpu {

#define CK(x)                                                                          \
    do {                                                                               \
        cudaError_t e_ = (x);                                                          \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x " : ") + cudaGetErrorString(e_)); \
    } while (0)

template <class T> struct Bits;
template <> struct Bits<float> { using U = uint32_t; };
template <> struct Bits<double> { using U = uint64_t; };

template <class T> __device__ __forceinline__ T lin(T a, T b) { return (a + b) * T(0.5); }
template <class T> __device__ __forceinline__ T lin1(T a, T b) { return fma(-T(0.5), a, T(1.5) * b); }
template <class T> __device__ __forceinline__ T quad1(T a, T b, T c) {
    T t = fma(T(6), b, -c);
    t = fma(T(3), a, t);
    return t * T(0.125);
}
template <class T> __device__ __forceinline__ T quad2(T a, T b, T c) {
    T t = fma(T(6), b, -a);
    t = fma(T(3), c, t);
    return t * T(0.125);
}
template <class T> __device__ __forceinline__ T cubic(T a, T b, T c, T d) {
    T t = (b + c) * T(9);
    t = t - a;
    t = t - d;
    return t * T(0.0625);
}

__device__ __forceinline__ void resolve(const FgNest &n, uint32_t off, uint64_t &d, uint64_t &p) {
    uint32_t r = off;
    const uint32_t i3 = r % n.cnt[3];
    r /= n.cnt[3];
    const uint32_t i2 = r % n.cnt[2];
    r /= n.cnt[2];
    const uint32_t i1 = r % n.cnt[1];
    const uint32_t i0 = r / n.cnt[1];
    d = n.data0 + i0 * n.dstr[0] + i1 * n.dstr[1] + i2 * n.dstr[2] + i3 * n.dstr[3];
    p = n.pos0 + i0 * n.pstr[0] + i1 * n.pstr[1] + i2 * n.pstr[2] + i3 * n.pstr[3];
}

template <class T>
__global__ void recon_kernel(T *__restrict__ data, const int16_t *__restrict__ stream, const FgNest *__restrict__ nests,
                             const FgWork *__restrict__ work, const uint64_t *__restrict__ escPos,
                             const typename Bits<T>::U *__restrict__ escVal, uint64_t numEsc,
                             const typename Bits<T>::U *__restrict__ anchors, double ebx2) {
    using U = typename Bits<T>::U;
    const FgWork w = work[blockIdx.x];
    const uint32_t off = w.start + threadIdx.x;
    const FgNest &n = nests[w.nest];
    if (off >= n.points) return;
    uint64_t d, p;
    resolve(n, off, d, p);

    if (n.op == FG_ANCHOR) {
        reinterpret_cast<U *>(data)[d] = anchors[p];
        return;
    }

    const int16_t q = stream[p];
    if (q == FG_ESCAPE) {
        uint64_t lo = 0, hi = numEsc;
        while (lo < hi) {
            uint64_t mid = (lo + hi) >> 1;
            if (escPos[mid] < p) lo = mid + 1;
            else hi = mid;
        }
        reinterpret_cast<U *>(data)[d] = escVal[lo];
        return;
    }

    const T *x = data + d;
    T pred;
    switch (n.op) {
        case FG_EQUAL: pred = x[n.opnd[0]]; break;
        case FG_LINEAR: pred = lin(x[n.opnd[0]], x[n.opnd[1]]); break;
        case FG_LINEAR1: pred = lin1(x[n.opnd[0]], x[n.opnd[1]]); break;
        case FG_QUAD1: pred = quad1(x[n.opnd[0]], x[n.opnd[1]], x[n.opnd[2]]); break;
        case FG_QUAD2: pred = quad2(x[n.opnd[0]], x[n.opnd[1]], x[n.opnd[2]]); break;
        case FG_CUBIC: pred = cubic(x[n.opnd[0]], x[n.opnd[1]], x[n.opnd[2]], x[n.opnd[3]]); break;
        default: pred = T(0); break;
    }
    const double qd = q;
    data[d] = n.fused ? static_cast<T>(fma(qd, ebx2, static_cast<double>(pred)))
                        : static_cast<T>(static_cast<double>(pred) + qd * ebx2);
}

template <class T>
__global__ void esc_val_kernel(const uint64_t *__restrict__ escPos, typename Bits<T>::U *__restrict__ escVal,
                               const typename Bits<T>::U *__restrict__ unpred, uint64_t numEsc,
                               const uint64_t *__restrict__ streamBase, const uint64_t *__restrict__ escFirst,
                               const uint64_t *__restrict__ unpredBase, int numStreams) {
    const uint64_t r = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= numEsc) return;
    const uint64_t p = escPos[r];
    int lo = 0, hi = numStreams - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (streamBase[mid] <= p) lo = mid;
        else hi = mid - 1;
    }
    escVal[r] = unpred[unpredBase[lo] + (r - escFirst[lo])];
}

namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

// Freed device blocks are kept for reuse: cudaMalloc costs milliseconds at these sizes.
class Pool {
   public:
    void *get(size_t bytes) {
        std::lock_guard<std::mutex> lk(m_);
        auto it = free_.lower_bound(bytes);
        if (it != free_.end() && it->first <= 2 * bytes + 4096) {
            void *p = it->second;
            size_[p] = it->first;
            free_.erase(it);
            return p;
        }
        void *p;
        cudaError_t e = cudaMalloc(&p, bytes);
        if (e == cudaErrorMemoryAllocation) {   // make room and retry
            trimLocked();
            e = cudaMalloc(&p, bytes);
        }
        CK(e);
        size_[p] = bytes;
        return p;
    }
    void put(void *p) {
        std::lock_guard<std::mutex> lk(m_);
        free_.emplace(size_[p], p);
    }
    void forget(void *p) {
        std::lock_guard<std::mutex> lk(m_);
        size_.erase(p);
    }
    void trim() {
        std::lock_guard<std::mutex> lk(m_);
        trimLocked();
    }

   private:
    void trimLocked() {
        for (auto &kv : free_) {
            size_.erase(kv.second);
            cudaFree(kv.second);
        }
        free_.clear();
    }
    std::mutex m_;
    std::multimap<size_t, void *> free_;
    std::unordered_map<void *, size_t> size_;
};

Pool &pool() {
    static Pool *p = new Pool;
    return *p;
}

template <class D> struct Dev {
    D *p = nullptr;
    explicit Dev(size_t n) { p = static_cast<D *>(pool().get(std::max<size_t>(n, 1) * sizeof(D))); }
    ~Dev() {
        if (p) pool().put(p);
    }
    void reset() {
        if (p) pool().put(p);
        p = nullptr;
    }
    D *release() {
        pool().forget(p);
        D *r = p;
        p = nullptr;
        return r;
    }
    Dev(const Dev &) = delete;
    Dev &operator=(const Dev &) = delete;
};

struct IsEsc {
    __host__ __device__ bool operator()(int16_t q) const { return q == FG_ESCAPE; }
};

template <class T>
void locate_escapes(const Plan &P, const int16_t *stream, uint64_t numUnpred, const std::vector<uint64_t> &perTid,
                    const typename Bits<T>::U *unpred, Dev<uint64_t> &escPos, Dev<typename Bits<T>::U> &escVal) {
    uint64_t nstream = 0;
    for (auto l : P.streamLen) nstream += l;
    uint64_t numEsc = 0;
    if (numUnpred) {
        auto first = thrust::counting_iterator<uint64_t>(0);
        auto outPos = thrust::device_pointer_cast(escPos.p);
        auto end = thrust::copy_if(first, first + nstream, thrust::device_pointer_cast(const_cast<int16_t *>(stream)), outPos, IsEsc());
        numEsc = end - outPos;
    }
    if (numEsc != numUnpred) throw std::runtime_error("escape count does not match the outlier count");
    if (!numEsc) return;
    const int K = static_cast<int>(P.streamLen.size());
    std::vector<uint64_t> base(K, 0), first(K, 0);
    uint64_t b = 0, f = 0;
    for (int i = 0; i < K; ++i) {
        base[i] = b;
        first[i] = f;
        b += P.streamLen[i];
        f += perTid[i];
    }
    Dev<uint64_t> dbase(K), dfirst(K);
    CK(cudaMemcpy(dbase.p, base.data(), K * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dfirst.p, first.data(), K * sizeof(uint64_t), cudaMemcpyHostToDevice));
    const int blocks = static_cast<int>((numEsc + FG_BLOCK - 1) / FG_BLOCK);
    esc_val_kernel<T><<<blocks, FG_BLOCK>>>(escPos.p, escVal.p, unpred, numEsc, dbase.p, dfirst.p, dfirst.p, K);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
}

template <class T>
void run_phases(const Plan &P, T *data, const int16_t *stream, const FgNest *nests, const FgWork *work, const uint64_t *escPos,
                const typename Bits<T>::U *escVal, uint64_t numEsc, const typename Bits<T>::U *anchors) {
    const bool timed = getenv("FGPU_TIMES") != nullptr;
    for (const auto &ph : P.phases) {
        auto a = Clock::now();
        recon_kernel<T><<<static_cast<int>(ph.workN), FG_BLOCK>>>(data, stream, nests, work + ph.work0, escPos, escVal, numEsc,
                                                                  anchors, ph.ebx2);
        CK(cudaGetLastError());
        if (timed) {
            CK(cudaDeviceSynchronize());
            const double t = ms(a, Clock::now());
            printf("  phase ops %.2f M, %u nests, %.3f ms, %.0f M/s\n", ph.points / 1e6, ph.nestN, t, ph.points / 1e3 / t);
        }
    }
    CK(cudaDeviceSynchronize());
}

template <class T>
void run(const ReconInput &in, void *out, Timing *tm, void **device) {
    using U = typename Bits<T>::U;
    const Plan &P = *in.plan;
    uint64_t nstream = 0;
    for (auto l : P.streamLen) nstream += l;
    uint64_t nanchor = 0;
    for (auto l : P.anchorLen) nanchor += l;

    auto t0 = Clock::now();
    Dev<T> data(P.num);
    Dev<int16_t> stream(nstream);
    Dev<FgNest> nests(P.nests.size());
    Dev<FgWork> work(P.work.size());
    Dev<U> anchors(nanchor), unpred(in.numUnpred);
    uint64_t b = 0;
    for (size_t i = 0; i < in.streams.size(); ++i) {
        CK(cudaMemcpy(stream.p + b, in.streams[i], P.streamLen[i] * sizeof(int16_t), cudaMemcpyHostToDevice));
        b += P.streamLen[i];
    }
    CK(cudaMemcpy(nests.p, P.nests.data(), P.nests.size() * sizeof(FgNest), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(work.p, P.work.data(), P.work.size() * sizeof(FgWork), cudaMemcpyHostToDevice));
    if (nanchor) CK(cudaMemcpy(anchors.p, in.anchors, nanchor * sizeof(U), cudaMemcpyHostToDevice));
    if (in.numUnpred) CK(cudaMemcpy(unpred.p, in.unpred, in.numUnpred * sizeof(U), cudaMemcpyHostToDevice));
    auto t1 = Clock::now();

    Dev<uint64_t> escPos(in.numUnpred);
    Dev<U> escVal(in.numUnpred);
    locate_escapes<T>(P, stream.p, in.numUnpred, in.unpredPerTid, unpred.p, escPos, escVal);
    auto t2 = Clock::now();

    run_phases<T>(P, data.p, stream.p, nests.p, work.p, escPos.p, escVal.p, in.numUnpred, anchors.p);
    auto t3 = Clock::now();
    if (out) CK(cudaMemcpy(out, data.p, P.num * sizeof(T), cudaMemcpyDeviceToHost));
    auto t4 = Clock::now();
    if (device) *device = data.release();
    if (tm) {
        tm->upload = ms(t0, t1);
        tm->escape = ms(t1, t2);
        tm->recon = ms(t2, t3);
        tm->download = ms(t3, t4);
    }
}
}  // namespace

void reconstruct(const ReconInput &in, void *out, Timing *t, void **device) {
    if (in.sizeT == 4) run<float>(in, out, t, device);
    else run<double>(in, out, t, device);
}

void free_device(void *p) { cudaFree(p); }

void trim_pool() { pool().trim(); }

void download(void *dst, const void *device, size_t bytes) { CK(cudaMemcpy(dst, device, bytes, cudaMemcpyDeviceToHost)); }

#include "fgpu_pure.inl"
#include "fgpu_compress.inl"

}  // namespace fgpu
