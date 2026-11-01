// Parses a Falcons stream on the host and runs the CPU side of the hybrid decoder.
#pragma once
#include <cstring>
#include <string>
#include <vector>

#include "Falcons/api/falcons.hpp"
#include "fgpu_recon.h"

namespace fgpu {

struct CompressTimes {
    double upload = 0, quant = 0, gather = 0;
};

template <class T>
struct HostStreams {
    std::vector<int16_t *> codes;   // one new[]'d array per stream, owned by this struct
    std::vector<T> anchors, unpred;
    std::vector<uint64_t> unpredPerTid;
    int numStreams = 0;
    PlanParams params;
    Plan plan;
    HostStreams() = default;
    HostStreams(HostStreams &&o) noexcept = default;
    ~HostStreams() {
        for (auto p : codes) delete[] p;
    }
};

inline InterpHeader read_interp_header(const unsigned char *&c, int N) {
    using namespace Falcons;
    InterpHeader h{};
    for (int i = 0; i < N; ++i) read(h.dims[i], c);
    read(h.blocksize, c);
    read(h.interpId, c);
    read(h.direction, c);
    read(h.anchorStride, c);
    read(h.alpha, c);
    read(h.beta, c);
    unsigned char uid;
    read(uid, c);
    read(h.eb, c);
    read(h.radius, c);
    return h;
}

// CPU half of the hybrid decoder: zstd + entropy decoding per stream.
template <class T>
HostStreams<T> decode_streams(const Falcons::Config &conf, const unsigned char *cmp, size_t cmpSize, bool avx = true) {
    using namespace Falcons;
    HostStreams<T> S;
    const unsigned char *pos = cmp;
    const bool serial = stream_is_serial(cmp);
    int K = 1;
    std::vector<size_t> offs(1, 0);
    if (!serial) {
        read(K, pos);
        offs.resize(K);
        read(offs.data(), K, pos);
    }
    S.numStreams = K;
    const size_t remaining = cmpSize - (pos - cmp);

    std::vector<std::vector<T>> un(K);
    std::vector<int16_t *> codes(K);
    std::vector<uint64_t> ncodes(K);
    InterpHeader hdr{};
#pragma omp parallel for schedule(static, 1) num_threads(K)
    for (int t = 0; t < K; ++t) {
        size_t len = (t != K - 1) ? offs[t + 1] - offs[t] : remaining - offs[t];
        uchar *buf = nullptr;
        size_t bufSize = 0;
        Lossless_zstd zstd;
        zstd.decompress(pos + offs[t], len, buf, bufSize);
        const uchar *p = buf;
        if (t == K - 1) hdr = read_interp_header(p, conf.N);
        int cnt;
        read(cnt, p);
        un[t].resize(cnt);
        if (cnt > 0) read(un[t].data(), cnt, p);
        if (serial || K == 1) {
            read(ncodes[t], p);
        } else {
            int n;
            read(n, p);
            ncodes[t] = n;
        }
        RleFseEncoder<int16_t> enc;
        size_t rem = bufSize - (p - buf);
        enc.load(p, rem);
        codes[t] = enc.decode(p, ncodes[t]);
        enc.postprocess_decode();
        free(buf);
    }

    PlanParams &pp = S.params;
    pp.N = static_cast<int>(conf.N);
    std::copy_n(hdr.dims, 4, pp.dims.begin());
    pp.sizeT = sizeof(T);
    size_t tc = K;
    for (int i = 1; i < pp.N; ++i) tc = std::min(tc, hdr.dims[i]);
    pp.threads = static_cast<int>(tc);
    pp.interp = hdr.interpId;
    pp.direction = hdr.direction;
    pp.anchorStride = hdr.anchorStride;
    pp.eb = hdr.eb;
    pp.alpha = hdr.alpha;
    pp.beta = hdr.beta;
    pp.avx = avx;
    pp.serial = serial;
    pp.fusedAll = serial || conf.falconsDataVer == versionInt(Falcons_DATA_VER);
    S.plan = build_plan(pp);

    uint64_t nun = 0, nanch = 0;
    for (int t = 0; t < K; ++t) {
        uint64_t expect = t < pp.threads ? S.plan.streamLen[t] : 0;
        uint64_t anch = t < pp.threads ? S.plan.anchorLen[t] : 0;
        if (expect != ncodes[t])
            throw std::runtime_error("plan disagrees with stream " + std::to_string(t) + ": " + std::to_string(expect) +
                                     " vs " + std::to_string(ncodes[t]));
        if (anch > un[t].size()) throw std::runtime_error("stream has fewer outliers than anchors");
        nanch += anch;
        nun += un[t].size() - anch;
    }
    S.anchors.resize(nanch);
    S.unpred.resize(nun);
    S.unpredPerTid.assign(pp.threads, 0);
    uint64_t a = 0, u = 0;
    for (int t = 0; t < K; ++t) {
        if (t >= pp.threads) {
            delete[] codes[t];
            continue;
        }
        uint64_t anch = S.plan.anchorLen[t];
        std::copy_n(un[t].begin(), anch, S.anchors.begin() + a);
        std::copy(un[t].begin() + anch, un[t].end(), S.unpred.begin() + u);
        S.unpredPerTid[t] = un[t].size() - anch;
        a += anch;
        u += S.unpredPerTid[t];
        S.codes.push_back(codes[t]);
    }
    return S;
}

// Shared by the GPU compressors: resolves the config and builds the plan.
template <class T>
Falcons::Config prepare_compress(const Falcons::Config &config, const T *data, int streams, PlanParams &pp) {
    using namespace Falcons;
    Config conf(config);
    conf.openmp = conf.openmp || streams > 1;
    calAbsErrorBound(conf, data);
    if (conf.interpAnchorStride < 0) {
        std::array<size_t, 4> strides = {4096, 128, 32, 16};
        conf.interpAnchorStride = strides[conf.N - 1];
    }
    const int N = static_cast<int>(conf.N);
    pp.N = N;
    std::copy(conf.dims.begin(), conf.dims.end(), pp.dims.begin());
    pp.sizeT = sizeof(T);
    size_t tc = streams;
    for (int i = 1; i < N; ++i) tc = std::min(tc, conf.dims[i]);
    pp.threads = static_cast<int>(tc);
    pp.serial = streams == 1;
    pp.interp = conf.interpAlgo;
    pp.direction = conf.interpDirection;
    pp.anchorStride = conf.interpAnchorStride;
    pp.eb = conf.absErrorBound;
    pp.alpha = conf.interpAlpha;
    pp.beta = conf.interpBeta;
    return conf;
}

inline std::vector<uint8_t> interp_header(const Falcons::Config &conf, int streams, int radius) {
    using namespace Falcons;
    const int N = static_cast<int>(conf.N);
    bool useAnchor = false;
    size_t maxDim = 1;
    for (int i = 0; i < N; ++i) {
        useAnchor = useAnchor || conf.dims[i] > static_cast<size_t>(conf.interpAnchorStride);
        maxDim = std::max(maxDim, conf.dims[i]);
    }
    std::vector<uint8_t> h(256);
    uchar *p = h.data();
    for (int i = 0; i < N; ++i) write(conf.dims[i], p);
    write(static_cast<uint32_t>(maxDim), p);
    write(static_cast<int>(conf.interpAlgo), p);
    write(static_cast<int>(conf.interpDirection), p);
    write(static_cast<size_t>(useAnchor ? conf.interpAnchorStride : 0), p);
    write(conf.interpAlpha, p);
    write(conf.interpBeta, p);
    write(static_cast<uchar>(streams == 1 ? 0b10 : 0b11), p);
    write(conf.absErrorBound, p);
    write(radius, p);
    h.resize(p - h.data());
    return h;
}

inline std::vector<char> join_blocks(const Falcons::Config &config, const PlanParams &pp, const PureCompressOutput &co, int streams) {
    using namespace Falcons;
    Config conf(config);
    const int K = pp.threads;
    const size_t confEst = conf.size_est();

    size_t total = confEst;
    if (streams > 1) total += 4 + 8 * size_t(K);
    for (auto &b : co.blocks) total += b.size();
    std::vector<char> out(total, 0);
    auto *q = reinterpret_cast<uchar *>(out.data()) + confEst;
    if (streams > 1) {
        write(K, q);
        size_t off = 0;
        for (auto &b : co.blocks) {
            write(off, q);
            off += b.size();
        }
    }
    for (auto &b : co.blocks) {
        std::memcpy(q, b.data(), b.size());
        q += b.size();
    }
    auto *c = reinterpret_cast<uchar *>(out.data());
    conf.save(c);
    return out;
}

// Pure-GPU compression. A valid stream for every Falcons decoder, not byte-equal to the CPU's because
// the zstd layer is stored raw. Empty when the data is nearly incompressible; use the CPU then.
template <class T>
std::vector<char> compress_pure_stream(const Falcons::Config &config, const T *data, int streams, int o1Chains = 4, PureCompressOutput *times = nullptr,
                                       const void *deviceData = nullptr) {
    using namespace Falcons;
    PlanParams pp;
    Config conf = prepare_compress<T>(config, data, streams, pp);
    Plan plan = build_plan(pp);
    PureCompressInput in;
    in.plan = &plan;
    in.sizeT = sizeof(T);
    in.serial = streams == 1;
    in.o1Chains = o1Chains;
    in.header = interp_header(conf, streams, conf.quantbinCnt / 2);
    in.deviceData = deviceData;
    PureCompressOutput co;
    compress_pure(in, data, co);
    if (!co.ok) return {};
    if (times) {
        times->upload = co.upload;
        times->quant = co.quant;
        times->rle = co.rle;
        times->fse = co.fse;
        times->tans = co.tans;
        times->assemble = co.assemble;
    }
    return join_blocks(conf, pp, co, streams);
}

// Hybrid compression: prediction and quantization on the GPU, encoding on the CPU with the very same
// encoder, so the result equals the CPU compressor's stream byte for byte.
// streams > 1 writes the -P layout, streams == 1 the serial layout. conf.cmprAlgo must be ALGO_INTERP.
template <class T>
std::vector<char> encode_on_cpu(const Falcons::Config &conf, const PlanParams &pp, const Plan &plan, CompressOutput &co, int streams) {
    using namespace Falcons;
    const int K = pp.threads;
    std::vector<uint64_t> base(K + 1, 0), abase(K + 1, 0), ebase(K + 1, 0);
    for (int t = 0; t < K; ++t) {
        base[t + 1] = base[t] + plan.streamLen[t];
        abase[t + 1] = abase[t] + plan.anchorLen[t];
        ebase[t + 1] = std::lower_bound(co.escPos.begin(), co.escPos.end(), base[t + 1]) - co.escPos.begin();
    }
    std::vector<std::vector<uchar>> blocks(K);
    std::vector<size_t> sizes(K);
    const int radius = conf.quantbinCnt / 2;
    const auto header = interp_header(conf, streams, radius);
#pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < K; ++t) {
        const size_t n = plan.streamLen[t], nun = plan.anchorLen[t] + (ebase[t + 1] - ebase[t]);
        std::vector<uchar> buf(4096 + 8 * n + (2 * sizeof(T) + 8) * nun + 2 * RleFseEncoder<int16_t>().size_est_without_init());
        uchar *p = buf.data();
        if (t == K - 1) {
            std::memcpy(p, header.data(), header.size());
            p += header.size();
        }
        write(static_cast<int>(nun), p);
        std::memcpy(p, co.anchors.data() + abase[t] * sizeof(T), plan.anchorLen[t] * sizeof(T));
        p += plan.anchorLen[t] * sizeof(T);
        std::memcpy(p, co.escVal.data() + ebase[t] * sizeof(T), (ebase[t + 1] - ebase[t]) * sizeof(T));
        p += (ebase[t + 1] - ebase[t]) * sizeof(T);
        if (streams == 1) write(static_cast<uint64_t>(n), p);
        else write(static_cast<int>(n), p);
        RleFseEncoder<int16_t> enc;
        enc.preprocess_encode(co.codes.data() + base[t], n, radius * 2);
        enc.save(p);
        enc.encode(co.codes.data() + base[t], n, p);
        const size_t size = p - buf.data();
        blocks[t].resize(ZSTD_compressBound(size) + 64);
        Lossless_zstd zstd;
        sizes[t] = zstd.compress(buf.data(), size, blocks[t].data(), blocks[t].size());
    }

    const size_t confEst = conf.size_est();
    size_t total = confEst;
    if (streams > 1) total += 4 + 8 * size_t(K);
    for (auto s : sizes) total += s;
    std::vector<char> out(total, 0);
    auto *q = reinterpret_cast<uchar *>(out.data()) + confEst;
    if (streams > 1) {
        write(K, q);
        size_t off = 0;
        for (int t = 0; t < K; ++t) {
            write(off, q);
            off += sizes[t];
        }
    }
    for (int t = 0; t < K; ++t) {
        std::memcpy(q, blocks[t].data(), sizes[t]);
        q += sizes[t];
    }
    auto *c = reinterpret_cast<uchar *>(out.data());
    Config saved(conf);
    saved.save(c);
    return out;
}

template <class T>
std::vector<char> compress_hybrid(const Falcons::Config &config, const T *data, int streams, CompressTimes *times = nullptr) {
    using namespace Falcons;
    PlanParams pp;
    Config conf = prepare_compress<T>(config, data, streams, pp);
    Plan plan = build_plan(pp);
    CompressInput ci;
    ci.plan = &plan;
    ci.sizeT = sizeof(T);
    CompressOutput co;
    compress_gpu(ci, data, co);
    if (!co.ok) return {};
    if (times) {
        times->upload = co.upload;
        times->quant = co.quant;
        times->gather = co.gather;
    }
    return encode_on_cpu<T>(conf, pp, plan, co, streams);
}

// Pure when the streams allow it. Tight bounds need order-1 tANS, whose chains the CPU encodes faster:
// quantization stays on the GPU and the codes go through the CPU encoders.
template <class T>
std::vector<char> compress_auto(const Falcons::Config &config, const T *data, int streams, bool *usedCpuEncoder = nullptr) {
    using namespace Falcons;
    PlanParams pp;
    Config conf = prepare_compress<T>(config, data, streams, pp);
    Plan plan = build_plan(pp);
    PureCompressInput in;
    in.plan = &plan;
    in.sizeT = sizeof(T);
    in.serial = streams == 1;
    in.header = interp_header(conf, streams, conf.quantbinCnt / 2);
    CompressOutput codes;
    in.codesIfTans = &codes;
    PureCompressOutput co;
    compress_pure(in, data, co);
    if (usedCpuEncoder) *usedCpuEncoder = codes.ok;
    if (codes.ok) return encode_on_cpu<T>(conf, pp, plan, codes, streams);
    if (!co.ok) return {};
    return join_blocks(conf, pp, co, streams);
}

}  // namespace fgpu
