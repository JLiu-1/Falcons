// GPU compression: the same sweeps as decoding, running the quantizer instead of the dequantizer.
// Included by fgpu_recon.cu.

namespace {

#include "fgpu_encode.inl"

template <class T>
struct EscEntry {
    uint64_t pos;
    typename Bits<T>::U val;
};

template <class T>
struct Extrema {
    T lo, hi;
};

template <class T>
struct ToExtrema {
    __host__ __device__ Extrema<T> operator()(T v) const {
        const T inf = std::numeric_limits<T>::infinity();
        return v == v ? Extrema<T>{v, v} : Extrema<T>{inf, -inf};
    }
};

template <class T>
struct MergeExtrema {
    __host__ __device__ Extrema<T> operator()(const Extrema<T> &a, const Extrema<T> &b) const {
        return {a.lo < b.lo ? a.lo : b.lo, a.hi > b.hi ? a.hi : b.hi};
    }
};

template <class T>
double range_of(const void *device, size_t count) {
    const T inf = std::numeric_limits<T>::infinity();
    auto first = thrust::device_pointer_cast(const_cast<T *>(static_cast<const T *>(device)));
    const Extrema<T> e = thrust::transform_reduce(first, first + count, ToExtrema<T>(), Extrema<T>{inf, -inf}, MergeExtrema<T>());
    return e.lo > e.hi ? 0 : static_cast<T>(e.hi - e.lo);
}

// cudaMemcpy into untouched memory faults every page in on one thread, which costs far more than the copy.
template <class V>
void touch_pages(V &v) {
    auto *p = reinterpret_cast<volatile char *>(v.data());
    const size_t bytes = v.size() * sizeof(v[0]);
#pragma omp parallel for
    for (size_t i = 0; i < bytes; i += 4096) p[i] = 0;
}

template <class T>
struct EscLess {
    __host__ __device__ bool operator()(const EscEntry<T> &a, const EscEntry<T> &b) const { return a.pos < b.pos; }
};

template <class T>
__global__ void quant_kernel(T *__restrict__ data, int16_t *__restrict__ stream, const FgNest *__restrict__ nests,
                             const FgWork *__restrict__ work, EscEntry<T> *__restrict__ esc, uint64_t *escCount, uint64_t escCap,
                             typename Bits<T>::U *__restrict__ anchors, double ebx2, double ebx2r, T eb) {
    using U = typename Bits<T>::U;
    const FgWork w = work[blockIdx.x];
    const uint32_t off = w.start + threadIdx.x;
    const FgNest &n = nests[w.nest];
    if (off >= n.points) return;
    uint64_t d, p;
    resolve(n, off, d, p);

    if (n.op == FG_ANCHOR) {
        anchors[p] = reinterpret_cast<const U *>(data)[d];
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
    const T ori = data[d];
    const T diff = ori - pred;
    const double qd = rint(static_cast<double>(diff) * ebx2r);
    bool ok = qd > -32768.0 && qd < 32768.0;   // false for NaN
    if (ok) {
        const T rec = static_cast<T>(fma(qd, ebx2, static_cast<double>(pred)));
        const T err = rec - ori;
        ok = err >= -eb && err <= eb;
        if (ok) {
            stream[p] = static_cast<int16_t>(qd);
            data[d] = rec;
        }
    }
    if (!ok) {
        stream[p] = FG_ESCAPE;
        const uint64_t slot = atomicAdd(reinterpret_cast<unsigned long long *>(escCount), 1ull);
        if (slot < escCap) esc[slot] = {p, reinterpret_cast<const U *>(&ori)[0]};
    }
}

template <class T>
struct QuantState {
    using U = typename Bits<T>::U;
    Dev<T> data;
    Dev<int16_t> stream;
    Dev<FgNest> nests;
    Dev<FgWork> work;
    Dev<U> anchors;
    uint64_t nstream, nanchor, numEsc = 0;
    std::vector<EscEntry<T>> esc;
    bool ok = false;
    double upload = 0, quant = 0, gather = 0;
    QuantState(const Plan &P, uint64_t ns, uint64_t na) : data(P.num), stream(ns), nests(P.nests.size()), work(P.work.size()), anchors(na), nstream(ns), nanchor(na) {}
};

template <class T>
void run_quant(const Plan &P, const T *src, const void *deviceSrc, QuantState<T> &q) {
    auto t0 = Clock::now();
    const uint64_t escCap = std::min<uint64_t>(q.nstream + 1, std::max<uint64_t>(1u << 20, q.nstream / 64));
    Dev<EscEntry<T>> esc(escCap);
    Dev<uint64_t> escCount(1);
    CK(cudaMemset(escCount.p, 0, 8));
    if (deviceSrc) CK(cudaMemcpy(q.data.p, deviceSrc, P.num * sizeof(T), cudaMemcpyDeviceToDevice));
    else CK(cudaMemcpy(q.data.p, src, P.num * sizeof(T), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(q.nests.p, P.nests.data(), P.nests.size() * sizeof(FgNest), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(q.work.p, P.work.data(), P.work.size() * sizeof(FgWork), cudaMemcpyHostToDevice));
    auto t1 = Clock::now();
    for (const auto &ph : P.phases) {
        quant_kernel<T><<<static_cast<int>(ph.workN), FG_BLOCK>>>(q.data.p, q.stream.p, q.nests.p, q.work.p + ph.work0, esc.p, escCount.p, escCap,
                                                                 q.anchors.p, ph.ebx2, ph.ebx2r, static_cast<T>(ph.ebT));
        CK(cudaGetLastError());
    }
    CK(cudaDeviceSynchronize());
    auto t2 = Clock::now();
    CK(cudaMemcpy(&q.numEsc, escCount.p, 8, cudaMemcpyDeviceToHost));
    if (q.numEsc > escCap) return;   // almost incompressible: the CPU path handles it
    if (q.numEsc) {
        auto first = thrust::device_pointer_cast(esc.p);
        thrust::sort(first, first + q.numEsc, EscLess<T>());
        q.esc.resize(q.numEsc);
        CK(cudaMemcpy(q.esc.data(), esc.p, q.numEsc * sizeof(EscEntry<T>), cudaMemcpyDeviceToHost));
    }
    q.ok = true;
    auto t3 = Clock::now();
    q.upload = ms(t0, t1);
    q.quant = ms(t1, t2);
    q.gather = ms(t2, t3);
}

template <class T>
void compress_codes(const CompressInput &in, const T *src, CompressOutput &co) {
    const Plan &P = *in.plan;
    uint64_t nstream = 0, nanchor = 0;
    for (auto l : P.streamLen) nstream += l;
    for (auto l : P.anchorLen) nanchor += l;
    QuantState<T> q(P, nstream, nanchor);
    run_quant<T>(P, src, in.deviceData, q);
    co.ok = q.ok;
    co.upload = q.upload;
    co.quant = q.quant;
    if (!q.ok) return;
    auto t0 = Clock::now();
    if (co.codes.size() != nstream) {
        co.codes.resize(nstream);
        touch_pages(co.codes);
    }
    co.anchors.resize(nanchor * sizeof(T));
    co.escPos.resize(q.numEsc);
    co.escVal.resize(q.numEsc * sizeof(T));
    CK(cudaMemcpy(co.codes.data(), q.stream.p, nstream * 2, cudaMemcpyDeviceToHost));
    if (nanchor) CK(cudaMemcpy(co.anchors.data(), q.anchors.p, nanchor * sizeof(T), cudaMemcpyDeviceToHost));
    for (uint64_t i = 0; i < q.numEsc; ++i) {
        co.escPos[i] = q.esc[i].pos;
        std::memcpy(co.escVal.data() + i * sizeof(T), &q.esc[i].val, sizeof(T));
    }
    co.gather = q.gather + ms(t0, Clock::now());
}

template <class T>
void compress_pure_impl(const PureCompressInput &in, const T *src, PureCompressOutput &co) {
    const Plan &P = *in.plan;
    const int K = static_cast<int>(P.streamLen.size());
    uint64_t nstream = 0, nanchor = 0;
    std::vector<uint64_t> base(K + 1, 0), abase(K + 1, 0);
    for (int t = 0; t < K; ++t) {
        base[t + 1] = base[t] + P.streamLen[t];
        abase[t + 1] = abase[t] + P.anchorLen[t];
    }
    nstream = base[K];
    nanchor = abase[K];
    QuantState<T> q(P, nstream, nanchor);
    run_quant<T>(P, src, in.deviceData, q);
    co.upload = q.upload;
    co.quant = q.quant;
    co.ok = q.ok;
    if (!q.ok) return;
    q.data.reset();   // the reconstruction is done with, and entropy coding needs the room
    auto t0 = Clock::now();

    // Method choice, as in RleFseEncoder::preprocess_encode.
    Dev<uint64_t> dbase(K), dlen(K);
    Dev<StreamStat> dstat(K);
    CK(cudaMemcpy(dbase.p, base.data(), K * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dlen.p, P.streamLen.data(), K * 8, cudaMemcpyHostToDevice));
    stat_kernel<<<K, 1>>>(q.stream.p, dbase.p, dlen.p, dstat.p);
    CK(cudaGetLastError());
    std::vector<StreamStat> stat(K);
    CK(cudaMemcpy(stat.data(), dstat.p, K * sizeof(StreamStat), cudaMemcpyDeviceToHost));
    std::vector<uint8_t> method(K, 0), delta(K, 0);
    for (int t = 0; t < K; ++t) {
        const double p0 = stat[t].cnt ? 1.0 * stat[t].z / stat[t].cnt : 1.0;
        const double dz = (stat[t].cnt && stat[t].cntd) ? (1.0 * stat[t].zd / stat[t].cntd - p0) : 0.0;
        delta[t] = dz > 0.005;
        method[t] = p0 >= 0.80 ? 0 : 1;
        if (P.streamLen[t] < static_cast<uint64_t>(20) * O1_NCTX * O1_NSYM) method[t] = 0;
    }

    if (in.codesIfTans && std::any_of(method.begin(), method.end(), [](uint8_t m) { return m != 0; })) {
        CompressOutput &c = *in.codesIfTans;
        c.ok = true;
        c.upload = q.upload;
        c.quant = q.quant;
        c.codes.resize(nstream);
        touch_pages(c.codes);
        c.anchors.resize(nanchor * sizeof(T));
        c.escPos.resize(q.numEsc);
        c.escVal.resize(q.numEsc * sizeof(T));
        CK(cudaMemcpy(c.codes.data(), q.stream.p, nstream * 2, cudaMemcpyDeviceToHost));
        if (nanchor) CK(cudaMemcpy(c.anchors.data(), q.anchors.p, nanchor * sizeof(T), cudaMemcpyDeviceToHost));
        for (uint64_t i = 0; i < q.numEsc; ++i) {
            c.escPos[i] = q.esc[i].pos;
            std::memcpy(c.escVal.data() + i * sizeof(T), &q.esc[i].val, sizeof(T));
        }
        co.ok = false;
        return;
    }

    // Working codes: a forward difference for streams that asked for it.
    bool anyDelta = false;
    for (int t = 0; t < K; ++t) anyDelta = anyDelta || delta[t];
    Dev<int16_t> workCopy(anyDelta ? nstream : 1);
    int16_t *wp = q.stream.p;
    if (anyDelta) {
        wp = workCopy.p;
        CK(cudaMemcpy(wp, q.stream.p, nstream * 2, cudaMemcpyDeviceToDevice));
        for (int t = 0; t < K; ++t) {
            const uint64_t n = P.streamLen[t];
            if (delta[t] && n) {
                delta_kernel<<<static_cast<int>((n + FG_BLOCK - 1) / FG_BLOCK), FG_BLOCK>>>(q.stream.p + base[t], wp + base[t], n);
                CK(cudaGetLastError());
            }
        }
    }

    // Method 0: runs and values. Everything here scales with the non-zero codes, not with the data.
    uint64_t nzMax = 1, nzTotal = 0;
    std::vector<uint64_t> nz(K, 0);
    auto wd = thrust::device_pointer_cast(wp);
    for (int t = 0; t < K; ++t) {
        if (method[t]) continue;
        nz[t] = thrust::count_if(wd + base[t], wd + base[t + 1], NonZero());
        nzMax = std::max(nzMax, nz[t]);
        nzTotal += nz[t];
    }
    const uint64_t entries = 2 * nzTotal + 2 * K + 2;
    Dev<uint64_t> pos(nzMax);
    Dev<uint8_t> cls(entries);
    Dev<uint64_t> val(entries), bits(entries);
    const uint64_t manWords = entries + 1024 * K;
    Dev<uint32_t> manW(manWords);
    CK(cudaMemset(manW.p, 0, manWords * 4));
    std::vector<uint64_t> nr(K, 0), nv(K, 0), rOff(K, 0), vOff(K, 0), rbp(K, 0), vbp(K, 0), rman(K, 0), vman(K, 0);
    uint64_t clsTop = 0, manTop = 0;
    std::vector<EncBlk> eb;
    std::vector<size_t> rBlk(K, 0), vBlk(K, 0);
    auto clsp = thrust::device_pointer_cast(cls.p);
    auto bitp = thrust::device_pointer_cast(bits.p);
    for (int t = 0; t < K; ++t) {
        if (method[t]) continue;
        const uint64_t n = P.streamLen[t];
        auto first = thrust::counting_iterator<uint64_t>(0);
        auto outPos = thrust::device_pointer_cast(pos.p);
        nv[t] = thrust::copy_if(first, first + n, wd + base[t], outPos, NonZero()) - outPos;
        uint64_t lastPos = 0;
        if (nv[t]) CK(cudaMemcpy(&lastPos, pos.p + nv[t] - 1, 8, cudaMemcpyDeviceToHost));
        nr[t] = nv[t] + ((nv[t] == 0 || lastPos != n - 1) ? 1 : 0);
        if (n == 0) nr[t] = 0;
        rOff[t] = clsTop;
        vOff[t] = clsTop + nr[t];
        clsTop += nr[t] + nv[t];
        for (auto [off, cnt, isRun] : {std::tuple<uint64_t, uint64_t, int>{rOff[t], nr[t], 1}, std::tuple<uint64_t, uint64_t, int>{vOff[t], nv[t], 0}}) {
            uint64_t &bytes = isRun ? rbp[t] : vbp[t];
            uint64_t &manPos = isRun ? rman[t] : vman[t];
            manPos = manTop;
            if (!cnt) continue;
            const int blocks = static_cast<int>((cnt + FG_BLOCK - 1) / FG_BLOCK);
            if (isRun) run_class_kernel<<<blocks, FG_BLOCK>>>(pos.p, cls.p + off, val.p + off, nv[t], cnt, n);
            else value_class_kernel<<<blocks, FG_BLOCK>>>(pos.p, wp + base[t], cls.p + off, val.p + off, cnt);
            CK(cudaGetLastError());
            thrust::exclusive_scan(clsp + off, clsp + off + cnt, bitp + off, uint64_t(0));
            uint64_t lastBit = 0;
            uint8_t lastCl = 0;
            CK(cudaMemcpy(&lastBit, bits.p + off + cnt - 1, 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(&lastCl, cls.p + off + cnt - 1, 1, cudaMemcpyDeviceToHost));
            bytes = (lastBit + lastCl + 7) / 8;
            pack_kernel<<<blocks, FG_BLOCK>>>(cls.p + off, val.p + off, bits.p + off, manW.p + manPos / 4, cnt);
            CK(cudaGetLastError());
            manTop = (manPos + bytes + 3) & ~uint64_t(3);
        }
        rBlk[t] = eb.size();
        for (uint64_t o = 0; o < nr[t]; o += FSE_FBLK) eb.push_back({rOff[t] + o, static_cast<uint32_t>(std::min<uint64_t>(nr[t] - o, FSE_FBLK)), 0});
        vBlk[t] = eb.size();
        for (uint64_t o = 0; o < nv[t]; o += FSE_FBLK) eb.push_back({vOff[t] + o, static_cast<uint32_t>(std::min<uint64_t>(nv[t] - o, FSE_FBLK)), 0});
    }
    auto t1 = Clock::now();

    std::vector<uint8_t> foHost, clsHost;
    if (!eb.empty()) {
        Dev<EncBlk> deb(eb.size());
        Dev<uint8_t> fo(eb.size() * FO_CAP);
        CK(cudaMemcpy(deb.p, eb.data(), eb.size() * sizeof(EncBlk), cudaMemcpyHostToDevice));
        fse_enc_kernel<<<static_cast<int>(eb.size()), 1>>>(cls.p, deb.p, fo.p);
        CK(cudaGetLastError());
        CK(cudaMemcpy(eb.data(), deb.p, eb.size() * sizeof(EncBlk), cudaMemcpyDeviceToHost));
        foHost.resize(eb.size() * FO_CAP);
        clsHost.resize(clsTop);
        CK(cudaMemcpy(foHost.data(), fo.p, foHost.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(clsHost.data(), cls.p, clsTop, cudaMemcpyDeviceToHost));
    }
    std::vector<uint8_t> manHost(manTop);
    if (manTop) CK(cudaMemcpy(manHost.data(), manW.p, manTop, cudaMemcpyDeviceToHost));
    auto t2 = Clock::now();

    // Method 1: segments, tables, chains.
    std::vector<EncSeg> segs;
    std::vector<uint32_t> segFirst(K, 0), segCount(K, 0);
    for (int t = 0; t < K; ++t) {
        segFirst[t] = static_cast<uint32_t>(segs.size());
        if (!method[t]) continue;
        const uint64_t n = P.streamLen[t];
        uint64_t S = n / O1_SEG_TARGET;
        S = std::max<uint64_t>(S, 1);
        S = std::min<uint64_t>(S, 16);
        const uint64_t seg = (n + S - 1) / S;
        segCount[t] = static_cast<uint32_t>(S);
        for (uint64_t g = 0; g < S; ++g) {
            const uint64_t lo = g * seg, hi = std::min(lo + seg, n);
            segs.push_back({base[t] + lo, hi - lo, static_cast<uint32_t>(in.o1Chains), 0});
        }
    }
    const size_t G = segs.size();
    std::vector<EncChain> ch;
    std::vector<ChainResult> cres;
    std::vector<uint16_t> nfHost;
    std::vector<uint8_t> chainBytes;
    std::vector<uint64_t> chainStart;
    std::vector<int16_t> ovfHost;
    if (G) {
        Dev<EncSeg> dseg(G);
        Dev<uint32_t> cnt(G * O1_NCTX * O1_NSYM);
        Dev<uint16_t> nf(G * O1_NCTX * O1_NSYM);
        Dev<uint8_t> ct(G * O1_NCTX * CT_BYTES), ctx(65536);
        auto lut = o1_ctx_lut();
        CK(cudaMemcpy(ctx.p, lut.data(), 65536, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dseg.p, segs.data(), G * sizeof(EncSeg), cudaMemcpyHostToDevice));
        o1_hist_kernel<<<static_cast<int>(G), 1>>>(wp, dseg.p, cnt.p, ctx.p);
        CK(cudaGetLastError());
        o1_ctable_kernel<<<static_cast<int>(G) * O1_NCTX, 1>>>(cnt.p, nf.p, ct.p);
        CK(cudaGetLastError());
        uint64_t out = 0;
        for (size_t g = 0; g < G; ++g) {
            const EncSeg &s = segs[g];
            const uint64_t cs = (s.N + s.K - 1) / s.K;
            for (uint32_t k = 0; k < s.K; ++k) {
                const uint64_t lo = s.code + k * cs, hi = std::min(lo + cs, s.code + s.N);
                ch.push_back({std::min(lo, s.code + s.N), hi, out, g * O1_NCTX * CT_BYTES, lo});
                out += cs * 2 + 4096;
            }
        }
        Dev<EncChain> dch(ch.size());
        Dev<ChainResult> dres(ch.size());
        Dev<uint8_t> cb(out);
        Dev<int16_t> ovf(nstream);
        CK(cudaMemcpy(dch.p, ch.data(), ch.size() * sizeof(EncChain), cudaMemcpyHostToDevice));
        o1_enc_kernel<<<static_cast<int>(ch.size()), 1>>>(wp, dch.p, ct.p, ctx.p, cb.p, ovf.p, dres.p);
        CK(cudaGetLastError());
        cres.resize(ch.size());
        CK(cudaMemcpy(cres.data(), dres.p, ch.size() * sizeof(ChainResult), cudaMemcpyDeviceToHost));
        nfHost.resize(G * O1_NCTX * O1_NSYM);
        CK(cudaMemcpy(nfHost.data(), nf.p, nfHost.size() * 2, cudaMemcpyDeviceToHost));
        uint64_t total = 0, totalOv = 0;
        chainStart.resize(ch.size());
        for (size_t i = 0; i < ch.size(); ++i) {
            chainStart[i] = total;
            total += cres[i].blen;
            totalOv += cres[i].novf;
        }
        chainBytes.resize(total);
        ovfHost.resize(totalOv);
        uint64_t ov = 0;
        for (size_t i = 0; i < ch.size(); ++i) {
            if (cres[i].blen) CK(cudaMemcpy(chainBytes.data() + chainStart[i], cb.p + ch[i].out, cres[i].blen, cudaMemcpyDeviceToHost));
            if (cres[i].novf) CK(cudaMemcpy(ovfHost.data() + ov, ovf.p + ch[i].ovf, cres[i].novf * 2, cudaMemcpyDeviceToHost));
            ov += cres[i].novf;
        }
    }
    std::vector<uint8_t> anchorHost(nanchor * sizeof(T));
    if (nanchor) CK(cudaMemcpy(anchorHost.data(), q.anchors.p, nanchor * sizeof(T), cudaMemcpyDeviceToHost));
    auto t3 = Clock::now();

    // Assemble each stream's payload, then the raw zstd frame around it.
    std::vector<uint64_t> ebase(K + 1, 0);
    {
        uint64_t e = 0;
        for (int t = 0; t < K; ++t) {
            while (e < q.numEsc && q.esc[e].pos < base[t + 1]) ++e;
            ebase[t + 1] = e;
        }
    }
    std::vector<uint64_t> chainFirst(K, 0);
    {
        size_t c = 0;
        for (int t = 0; t < K; ++t) {
            chainFirst[t] = c;
            for (uint32_t g = segFirst[t]; g < segFirst[t] + segCount[t]; ++g) c += segs[g].K;
        }
    }
    std::vector<uint64_t> ovfFirst(ch.size() + 1, 0);
    for (size_t i = 0; i < ch.size(); ++i) ovfFirst[i + 1] = ovfFirst[i] + cres[i].novf;
    co.blocks.assign(K, {});
#pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < K; ++t) {
        std::vector<uint8_t> b;
        const uint64_t n = P.streamLen[t], nun = P.anchorLen[t] + (ebase[t + 1] - ebase[t]);
        if (t == K - 1) b.insert(b.end(), in.header.begin(), in.header.end());
        const int nunI = static_cast<int>(nun);
        const auto *pi = reinterpret_cast<const uint8_t *>(&nunI);
        b.insert(b.end(), pi, pi + 4);
        const auto *ap = anchorHost.data() + abase[t] * sizeof(T);
        b.insert(b.end(), ap, ap + P.anchorLen[t] * sizeof(T));
        for (uint64_t e = ebase[t]; e < ebase[t + 1]; ++e) {
            const auto *v = reinterpret_cast<const uint8_t *>(&q.esc[e].val);
            b.insert(b.end(), v, v + sizeof(T));
        }
        if (K == 1) {
            put_u64(b, n);
        } else {
            const int nI = static_cast<int>(n);
            const auto *p = reinterpret_cast<const uint8_t *>(&nI);
            b.insert(b.end(), p, p + 4);
        }
        b.push_back(static_cast<uint8_t>(method[t] | (delta[t] ? 0x80 : 0)));
        b.insert(b.end(), 4, 0);   // lo, mode
        if (method[t] == 0) {
            const size_t rb = (nr[t] + FSE_FBLK - 1) / FSE_FBLK, vb = (nv[t] + FSE_FBLK - 1) / FSE_FBLK;
            std::vector<uint8_t> orcl, ovcl;
            append_fse(orcl, eb, rBlk[t], rb, foHost.data(), clsHost.data(), 0);
            append_fse(ovcl, eb, vBlk[t], vb, foHost.data(), clsHost.data(), 0);
            for (uint64_t x : {nr[t], nv[t], uint64_t(orcl.size()), uint64_t(ovcl.size()), rbp[t], vbp[t]}) put_u64(b, x);
            b.insert(b.end(), orcl.begin(), orcl.end());
            b.insert(b.end(), ovcl.begin(), ovcl.end());
            if (rbp[t]) b.insert(b.end(), manHost.begin() + rman[t], manHost.begin() + rman[t] + rbp[t]);
            if (vbp[t]) b.insert(b.end(), manHost.begin() + vman[t], manHost.begin() + vman[t] + vbp[t]);
        } else {
            put_u64(b, segCount[t]);
            size_t c = chainFirst[t];
            for (uint32_t g = segFirst[t]; g < segFirst[t] + segCount[t]; ++g) {
                const EncSeg &s = segs[g];
                put_u64(b, s.N);
                put_u64(b, s.K);
                for (uint32_t k = 0; k < s.K; ++k) put_u64(b, cres[c + k].blen);
                for (uint32_t k = 0; k < s.K; ++k) put_u64(b, cres[c + k].novf);
                const auto *nfp = reinterpret_cast<const uint8_t *>(nfHost.data() + uint64_t(g) * O1_NCTX * O1_NSYM);
                b.insert(b.end(), nfp, nfp + O1_NFBYTES);
                uint64_t totOv = 0;
                for (uint32_t k = 0; k < s.K; ++k) {
                    b.insert(b.end(), chainBytes.begin() + chainStart[c + k], chainBytes.begin() + chainStart[c + k] + cres[c + k].blen);
                    totOv += cres[c + k].novf;
                }
                append_ovf(b, ovfHost.data() + ovfFirst[c], totOv);
                c += s.K;
            }
        }
        wrap_zstd_raw(co.blocks[t], b.data(), b.size());
    }
    auto t4 = Clock::now();
    co.rle = ms(t0, t1);
    co.fse = ms(t1, t2);
    co.tans = ms(t2, t3);
    co.assemble = ms(t3, t4);
}

}  // namespace

double device_range(const void *device, size_t count, size_t sizeT) {
    return sizeT == 4 ? range_of<float>(device, count) : range_of<double>(device, count);
}

void compress_gpu(const CompressInput &in, const void *data, CompressOutput &out) {
    if (in.sizeT == 4) compress_codes<float>(in, static_cast<const float *>(data), out);
    else compress_codes<double>(in, static_cast<const double *>(data), out);
}

void compress_pure(const PureCompressInput &in, const void *data, PureCompressOutput &out) {
    if (in.sizeT == 4) compress_pure_impl<float>(in, static_cast<const float *>(data), out);
    else compress_pure_impl<double>(in, static_cast<const double *>(data), out);
}
