// Pure-GPU decoder for the OMP stream layout. The host only walks container metadata;
// every payload byte is decoded in device memory. Included by fgpu_recon.cu.

namespace {

constexpr int FSE_FBLK = 1 << 16;

struct CopyChunk {
    uint64_t src, dst;
    uint32_t len;
    uint32_t pad;
};

struct FseChain {
    uint64_t src;     // offset into raw
    uint64_t dst;     // offset into cls
    uint64_t n;       // decoded symbols
    uint32_t first;   // first FseBlk entry
    uint32_t pad;
};

struct FseBlk {
    uint64_t src, dst;
    uint32_t len, clen;
    uint8_t flag, rle, pad[6];
};

// Backward bit reader over a 4-aligned byte buffer, cached in a 64-bit word.
struct BackCache {
    const uint32_t *w;
    int64_t low;     // lowest bit already pulled into cont
    uint64_t cont;
    int have;
    __device__ __forceinline__ void init(const uint8_t *base, int64_t bitEnd) {
        w = reinterpret_cast<const uint32_t *>(base);
        low = bitEnd;
        cont = 0;
        have = 0;
    }
    __device__ __forceinline__ uint32_t read(uint32_t nb) {
        while (have < static_cast<int>(nb)) {
            int n = 32;
            uint32_t v = 0;
            if (low > 0) {
                const int64_t wl = (low - 1) >> 5;
                n = static_cast<int>(low - (wl << 5));
                v = w[wl];
                if (n < 32) v &= (1u << n) - 1;
                low -= n;
            }
            cont = (cont << n) | v;
            have += n;
        }
        have -= nb;
        return static_cast<uint32_t>(cont >> have) & ((1u << nb) - 1);
    }
};

__device__ __forceinline__ uint32_t rd32le(const uint8_t *p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

// FSE_readNCount. Returns the header length.
__device__ inline uint32_t fse_read_ncount(int16_t *norm, uint32_t &maxSV, uint32_t &tableLog, const uint8_t *src, uint32_t size) {
    const uint8_t *ip = src;
    const uint8_t *iend = src + (size < 4 ? 4 : size);
    uint32_t bitStream = rd32le(ip);
    int nbBits = (bitStream & 0xF) + 5;
    bitStream >>= 4;
    int bitCount = 4;
    tableLog = nbBits;
    int remaining = (1 << nbBits) + 1;
    int threshold = 1 << nbBits;
    nbBits++;
    uint32_t charnum = 0;
    int previous0 = 0;
    for (uint32_t i = 0; i <= maxSV; ++i) norm[i] = 0;
    while ((remaining > 1) & (charnum <= maxSV)) {
        if (previous0) {
            uint32_t n0 = charnum;
            while ((bitStream & 0xFFFF) == 0xFFFF) {
                n0 += 24;
                if (ip < iend - 5) {
                    ip += 2;
                    bitStream = rd32le(ip) >> bitCount;
                } else {
                    bitStream >>= 16;
                    bitCount += 16;
                }
            }
            while ((bitStream & 3) == 3) {
                n0 += 3;
                bitStream >>= 2;
                bitCount += 2;
            }
            n0 += bitStream & 3;
            bitCount += 2;
            while (charnum < n0) norm[charnum++] = 0;
            if ((ip <= iend - 7) || (ip + (bitCount >> 3) <= iend - 4)) {
                ip += bitCount >> 3;
                bitCount &= 7;
                bitStream = rd32le(ip) >> bitCount;
            } else {
                bitStream >>= 2;
            }
        }
        int max = (2 * threshold - 1) - remaining;
        int count;
        if ((bitStream & (threshold - 1)) < uint32_t(max)) {
            count = bitStream & (threshold - 1);
            bitCount += nbBits - 1;
        } else {
            count = bitStream & (2 * threshold - 1);
            if (count >= threshold) count -= max;
            bitCount += nbBits;
        }
        count--;
        remaining -= count < 0 ? -count : count;
        norm[charnum++] = static_cast<int16_t>(count);
        previous0 = !count;
        while (remaining < threshold) {
            nbBits--;
            threshold >>= 1;
        }
        if ((ip <= iend - 7) || (ip + (bitCount >> 3) <= iend - 4)) {
            ip += bitCount >> 3;
            bitCount &= 7;
        } else {
            bitCount -= static_cast<int>(8 * (iend - 4 - ip));
            ip = iend - 4;
        }
        bitStream = rd32le(ip) >> (bitCount & 31);
    }
    maxSV = charnum - 1;
    ip += (bitCount + 7) >> 3;
    return static_cast<uint32_t>(ip - src);
}

// Cell layout: newState | symbol << 16 | nbBits << 24.
__device__ inline void fse_build_table(uint32_t *table, const int16_t *norm, uint32_t maxSV, uint32_t tableLog) {
    uint16_t symbolNext[256];
    const uint32_t tableSize = 1u << tableLog;
    uint32_t high = tableSize - 1;
    for (uint32_t s = 0; s <= maxSV; ++s) {
        if (norm[s] == -1) {
            table[high--] = s << 16;
            symbolNext[s] = 1;
        } else {
            symbolNext[s] = norm[s];
        }
    }
    const uint32_t mask = tableSize - 1, step = (tableSize >> 1) + (tableSize >> 3) + 3;
    uint32_t pos = 0;
    for (uint32_t s = 0; s <= maxSV; ++s) {
        for (int i = 0; i < norm[s]; ++i) {
            table[pos] = s << 16;
            pos = (pos + step) & mask;
            while (pos > high) pos = (pos + step) & mask;
        }
    }
    for (uint32_t u = 0; u < tableSize; ++u) {
        const uint32_t sym = table[u] >> 16;
        const uint32_t next = symbolNext[sym]++;
        const uint32_t nb = tableLog - (31 - __clz(next));
        table[u] = ((next << nb) - tableSize) | (sym << 16) | (nb << 24);
    }
}

__global__ void copy_bytes_kernel(uint8_t *__restrict__ dst, const uint8_t *__restrict__ src, const CopyChunk *__restrict__ work) {
    const CopyChunk c = work[blockIdx.x];
    if (threadIdx.x < c.len) dst[c.dst + threadIdx.x] = src[c.src + threadIdx.x];
}

__global__ void gather_kernel(uint8_t *__restrict__ dst, const uint8_t *__restrict__ src, const uint64_t *__restrict__ offs, int bytesEach) {
    const uint32_t i = blockIdx.x, b = threadIdx.x;
    dst[uint64_t(i) * bytesEach + b] = src[offs[i] + b];
}

// Walks the [flag][u32 clen?][data] framing of one chain.
__global__ void fse_walk_kernel(const uint8_t *__restrict__ raw, const FseChain *__restrict__ chains, FseBlk *__restrict__ blks, int numChains) {
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= uint32_t(numChains)) return;
    const FseChain c = chains[id];
    uint64_t rp = c.src, dp = 0;
    uint32_t b = c.first;
    while (dp < c.n) {
        const uint32_t len = c.n - dp < uint64_t(FSE_FBLK) ? static_cast<uint32_t>(c.n - dp) : FSE_FBLK;
        FseBlk k{};
        k.flag = raw[rp++];
        k.dst = c.dst + dp;
        k.len = len;
        if (k.flag == 0) {
            k.src = rp;
            rp += len;
        } else if (k.flag == 2) {
            k.rle = raw[rp++];
        } else {
            k.clen = rd32le(raw + rp);
            rp += 4;
            k.src = rp;
            rp += k.clen;
        }
        blks[b++] = k;
        dp += len;
    }
}

__global__ void fse_block_kernel(const uint8_t *__restrict__ raw, uint8_t *__restrict__ cls, const FseBlk *__restrict__ blks) {
    __shared__ uint32_t table[1u << 13];   // on-chip: the symbol loop is one table load per symbol
    const uint32_t id = blockIdx.x;
    const FseBlk k = blks[id];
    uint8_t *dst = cls + k.dst;
    if (k.flag == 0) {
        for (uint32_t i = 0; i < k.len; ++i) dst[i] = raw[k.src + i];
        return;
    }
    if (k.flag == 2) {
        for (uint32_t i = 0; i < k.len; ++i) dst[i] = k.rle;
        return;
    }
    const uint8_t *src = raw + k.src;
    int16_t norm[256];
    uint32_t maxSV = 255, tableLog;
    const uint32_t hl = fse_read_ncount(norm, maxSV, tableLog, src, k.clen);
    fse_build_table(table, norm, maxSV, tableLog);
    src += hl;
    const uint32_t size = k.clen - hl;
    BackCache br;
    br.init(raw, int64_t(k.src + hl + size - 1) * 8 + (31 - __clz(uint32_t(src[size - 1]))));
    uint32_t st[2];
    st[0] = br.read(tableLog);
    st[1] = br.read(tableLog);
    uint32_t a = st[0], b = st[1], i = 0;
    uint32_t *out = reinterpret_cast<uint32_t *>(dst);   // chains start 4-aligned
    for (; i + 4 <= k.len; i += 4) {
        const uint32_t c0 = table[a];
        a = (c0 & 0xFFFF) + br.read(c0 >> 24);
        const uint32_t c1 = table[b];
        b = (c1 & 0xFFFF) + br.read(c1 >> 24);
        const uint32_t c2 = table[a];
        a = (c2 & 0xFFFF) + br.read(c2 >> 24);
        const uint32_t c3 = table[b];
        b = (c3 & 0xFFFF) + br.read(c3 >> 24);
        out[i >> 2] = ((c0 >> 16) & 0xFF) | ((c1 >> 8) & 0xFF00) | (c2 & 0xFF0000) | ((c3 << 8) & 0xFF000000);
    }
    for (; i < k.len; ++i) {
        uint32_t &s = (i & 1) ? b : a;
        const uint32_t cell = table[s];
        dst[i] = (cell >> 16) & 0xFF;
        s = (cell & 0xFFFF) + br.read(cell >> 24);
    }
}

// MSB-first mantissa bits at bit offset b.
__device__ __forceinline__ uint64_t read_msb(const uint8_t *m, uint64_t b, uint32_t k) {
    if (k == 0) return 0;
    const uint64_t byte = b >> 3;
    unsigned __int128 w = 0;
    const uint32_t need = ((b & 7) + k + 7) >> 3;
    for (uint32_t i = 0; i < need; ++i) w = (w << 8) | m[byte + i];
    const uint32_t drop = need * 8 - (b & 7) - k;
    return static_cast<uint64_t>(w >> drop) & ((k == 64 ? 0 : (1ull << k)) - 1);
}

// One stream's arrays inside the shared class buffer. Positions and bit offsets come from one global
// prefix sum over all streams; the base values turn them back into per-stream numbers.
struct RleChain {
    uint64_t runOff, valOff, nr, nv;
    uint64_t bitBaseR, bitBaseV, lenBase;
    uint64_t rman, vman, code;
};

struct RleWork {
    uint32_t chain;
    uint32_t start;
};

__global__ void run_len_kernel(const uint8_t *__restrict__ cls, const uint64_t *__restrict__ bits, const uint8_t *__restrict__ raw, uint64_t *__restrict__ len,
                               const RleChain *__restrict__ chains, const RleWork *__restrict__ work) {
    const RleWork w = work[blockIdx.x];
    const RleChain &c = chains[w.chain];
    const uint64_t i = uint64_t(w.start) + threadIdx.x;
    if (i >= c.nr) return;
    const uint64_t at = c.runOff + i;
    const uint32_t k = cls[at];
    len[at] = (1ull << k) | read_msb(raw + c.rman, bits[at] - bits[c.runOff], k);   // L + 1, the distance to the next value
}

__global__ void value_kernel(const uint8_t *__restrict__ cls, const uint64_t *__restrict__ bits, const uint8_t *__restrict__ raw, const uint64_t *__restrict__ end,
                             int16_t *__restrict__ out, const RleChain *__restrict__ chains, const RleWork *__restrict__ work) {
    const RleWork w = work[blockIdx.x];
    const RleChain &c = chains[w.chain];
    const uint64_t i = uint64_t(w.start) + threadIdx.x;
    if (i >= c.nv) return;
    const uint64_t at = c.valOff + i;
    const uint32_t k = cls[at];
    const uint32_t vv = (1u << k) | static_cast<uint32_t>(read_msb(raw + c.vman, bits[at] - bits[c.valOff], k));
    const uint32_t zz = vv - 1;
    const int32_t fct = static_cast<int32_t>((zz >> 1) ^ (~(zz & 1) + 1));
    const uint64_t base = c.runOff ? end[c.runOff - 1] : 0;
    out[c.code + (end[c.runOff + i] - base) - 1] = static_cast<int16_t>(fct);
}

#include "fgpu_entropy.inl"

struct StreamMeta {
    uint64_t rawBase, rawLen;
    uint64_t unpredOff;   // offset of the outlier array, relative to rawBase
    int unpredCnt;
    uint64_t quantCnt;
    uint8_t method;
    bool delta;
    uint64_t nr, nv, sRcl, sVcl, rbp, vbp, payload;
};

template <class V>
V get(const uint8_t *&p) {
    V v;
    std::memcpy(&v, p, sizeof(V));
    p += sizeof(V);
    return v;
}

void launch_chunks(uint8_t *dst, const uint8_t *src, const std::vector<CopyChunk> &chunks) {
    if (chunks.empty()) return;
    Dev<CopyChunk> w(chunks.size());
    CK(cudaMemcpy(w.p, chunks.data(), chunks.size() * sizeof(CopyChunk), cudaMemcpyHostToDevice));
    copy_bytes_kernel<<<static_cast<int>(chunks.size()), FG_BLOCK>>>(dst, src, w.p);
    CK(cudaGetLastError());
}

void add_copy(std::vector<CopyChunk> &v, uint64_t src, uint64_t dst, uint64_t len) {
    for (uint64_t o = 0; o < len; o += FG_BLOCK)
        v.push_back({src + o, dst + o, static_cast<uint32_t>(std::min<uint64_t>(len - o, FG_BLOCK)), 0});
}

// Debug aid: compares device buffers against host-side zstd and the reference codes.
inline void check_stages(const PureInput &in, const Plan &P, int K, const std::vector<StreamMeta> &M, const uint8_t *raw,
                         const int16_t *stream, const std::vector<uint64_t> &codeBase) {
    std::vector<size_t> offs(K, 0);
    uint64_t dataBase = 0;
    if (!stream_is_serial(in.cmp)) {
        std::memcpy(offs.data(), in.cmp + 4, K * 8);
        dataBase = 4 + 8 * uint64_t(K);
    }
    for (int t = 0; t < K; ++t) {
        uint64_t len = (t != K - 1 ? offs[t + 1] : in.cmpSize - dataBase) - offs[t];
        std::vector<uint8_t> host(M[t].rawLen), dev(M[t].rawLen);
        ZSTD_decompress(host.data(), host.size(), in.cmp + dataBase + offs[t] + 8, len - 8);
        CK(cudaMemcpy(dev.data(), raw + M[t].rawBase, dev.size(), cudaMemcpyDeviceToHost));
        size_t i = 0;
        while (i < dev.size() && dev[i] == host[i]) ++i;
        if (i != dev.size()) printf("  stream %d: raw differs at %zu of %zu\n", t, i, dev.size());
        if (size_t(t) < P.streamLen.size() && t < int(in.refCodes.size())) {
            std::vector<int16_t> c(P.streamLen[t]);
            CK(cudaMemcpy(c.data(), stream + codeBase[t], c.size() * 2, cudaMemcpyDeviceToHost));
            size_t j = 0;
            while (j < c.size() && c[j] == in.refCodes[t][j]) ++j;
            if (j != c.size())
                printf("  stream %d: codes differ at %zu of %zu (method %d delta %d nr %llu nv %llu): gpu %d ref %d\n", t, j, c.size(),
                       M[t].method, int(M[t].delta), (unsigned long long)M[t].nr, (unsigned long long)M[t].nv, c[j], in.refCodes[t][j]);
        }
    }
}

template <class T>
bool pure(const PureInput &in, void *out, PureTiming *tm, void **device) {
    using U = typename Bits<T>::U;
    auto t0 = Clock::now();
    const uint8_t *cmp = in.cmp;
    const bool serial = stream_is_serial(cmp);
    int K = 1;
    std::vector<size_t> offs(1, 0);
    uint64_t dataBase = 0;
    if (!serial) {
        std::memcpy(&K, cmp, 4);
        offs.resize(K);
        std::memcpy(offs.data(), cmp + 4, K * 8);
        dataBase = 4 + 8 * uint64_t(K);
    }

    // Container walk: zstd frame headers and block headers only.
    std::vector<StreamMeta> M(K);
    std::vector<CopyChunk> rawCopies;
    std::vector<ZBlk> zblks;
    std::vector<ZStream> zstreams;
    uint64_t rawTotal = 0;
    for (int t = 0; t < K; ++t) {
        uint64_t len = (t != K - 1 ? offs[t + 1] : in.cmpSize - dataBase) - offs[t];
        const uint8_t *p = cmp + dataBase + offs[t];
        uint64_t srcLen;
        std::memcpy(&srcLen, p, 8);
        const uint8_t *q = p + 8;
        uint32_t magic;
        std::memcpy(&magic, q, 4);
        if (magic != 0xFD2FB528u) throw std::runtime_error("stream is not a zstd frame");
        const unsigned fhd = q[4];
        const unsigned single = (fhd >> 5) & 1, fcsf = fhd >> 6, did = fhd & 3;
        const unsigned didSize = did == 3 ? 4 : did;
        const unsigned fcs = fcsf == 0 ? (single ? 1 : 0) : (1u << fcsf);
        q += 4 + 1 + (single ? 0 : 1) + didSize + fcs;
        M[t].rawBase = rawTotal;
        M[t].rawLen = srcLen;
        // zstd cuts a frame every 128 KB at this level, so each block's output offset is known
        // up front. Raw blocks go straight to the parallel copy; compressed blocks check the
        // assumption when they finish.
        constexpr uint64_t ZBLOCK = 128 * 1024;
        const uint32_t firstBlk = static_cast<uint32_t>(zblks.size());
        uint64_t dst = rawTotal;
        bool last = false;
        while (!last) {
            uint32_t bh = q[0] | (q[1] << 8) | (uint32_t(q[2]) << 16);
            last = bh & 1;
            unsigned type = (bh >> 1) & 3;
            uint32_t bs = bh >> 3;
            q += 3;
            if (type == 3) throw std::runtime_error("reserved zstd block type");
            const uint64_t size = last ? rawTotal + srcLen - dst : ZBLOCK;
            if (dst + size > rawTotal + srcLen || (type != 2 && bs != size)) return false;
            if (type == 0) add_copy(rawCopies, q - cmp, dst, bs);
            else zblks.push_back({static_cast<uint64_t>(q - cmp), dst, dst + size, bs, type});
            q += type == 1 ? 1 : bs;
            dst += size;
        }
        if (uint64_t(q - p) > len) throw std::runtime_error("zstd frame overruns its stream");
        if (zblks.size() > firstBlk) zstreams.push_back({firstBlk, static_cast<uint32_t>(zblks.size()) - firstBlk});
        rawTotal += (srcLen + 7) & ~uint64_t(7);
    }
    // Compressed blocks hold the tANS tables, so they mark an order-1 stream without decoding anything.
    if (in.rejectMethod1 && !zstreams.empty()) return false;
    auto t1 = Clock::now();

    Dev<uint8_t> dcmp(in.cmpSize), raw(rawTotal + 2 * FG_BLOCK);   // headers are gathered in fixed-size pieces
    CK(cudaMemcpy(dcmp.p, cmp, in.cmpSize, cudaMemcpyHostToDevice));
    auto t2 = Clock::now();
    launch_chunks(raw.p, dcmp.p, rawCopies);
    if (!zstreams.empty()) {
        const size_t Z = zstreams.size();
        Dev<ZStream> dzs(Z);
        Dev<ZBlk> dzb(zblks.size());
        Dev<uint8_t> zsc(Z * ZSCRATCH);
        Dev<uint32_t> zbad(Z);
        CK(cudaMemset(zbad.p, 0, Z * 4));
        CK(cudaMemcpy(dzs.p, zstreams.data(), Z * sizeof(ZStream), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dzb.p, zblks.data(), zblks.size() * sizeof(ZBlk), cudaMemcpyHostToDevice));
        zstd_kernel<<<static_cast<int>(Z), 1>>>(raw.p, dcmp.p, dzs.p, dzb.p, zsc.p, zbad.p);
        CK(cudaGetLastError());
        std::vector<uint32_t> bad(Z);
        CK(cudaMemcpy(bad.data(), zbad.p, Z * 4, cudaMemcpyDeviceToHost));
        for (auto b : bad)
            if (b) return false;
    }
    auto t2z = Clock::now();

    // Stream headers. The interpolation header sits at the front of the last stream.
    constexpr int HDR_BYTES = FG_BLOCK;
    Dev<uint64_t> gatherOff(K);
    Dev<uint8_t> gathered(K * HDR_BYTES);
    std::vector<uint64_t> gOff(K);
    std::vector<uint8_t> hdr(K * HDR_BYTES);
    auto gather = [&]() {
        CK(cudaMemcpy(gatherOff.p, gOff.data(), K * 8, cudaMemcpyHostToDevice));
        gather_kernel<<<K, HDR_BYTES>>>(gathered.p, raw.p, gatherOff.p, HDR_BYTES);
        CK(cudaGetLastError());
        CK(cudaMemcpy(hdr.data(), gathered.p, K * HDR_BYTES, cudaMemcpyDeviceToHost));
    };
    for (int t = 0; t < K; ++t) gOff[t] = M[t].rawBase;
    gather();
    const int N = static_cast<int>(in.conf->N);
    InterpHeader ih{};
    {
        const uint8_t *c = hdr.data() + uint64_t(K - 1) * HDR_BYTES;
        for (int i = 0; i < N; ++i) ih.dims[i] = get<size_t>(c);
        ih.blocksize = get<uint32_t>(c);
        ih.interpId = get<int>(c);
        ih.direction = get<int>(c);
        ih.anchorStride = get<size_t>(c);
        ih.alpha = get<double>(c);
        ih.beta = get<double>(c);
        get<uint8_t>(c);
        ih.eb = get<double>(c);
        ih.radius = get<int>(c);
        const uint64_t hsize = c - (hdr.data() + uint64_t(K - 1) * HDR_BYTES);
        for (int t = 0; t < K; ++t) {
            const uint64_t skip = t == K - 1 ? hsize : 0;
            const uint8_t *u = hdr.data() + uint64_t(t) * HDR_BYTES + skip;
            M[t].unpredCnt = get<int>(u);
            M[t].unpredOff = skip + 4;
            gOff[t] = M[t].rawBase + M[t].unpredOff + uint64_t(M[t].unpredCnt) * sizeof(T);
        }
    }
    gather();
    for (int t = 0; t < K; ++t) {
        const uint8_t *c = hdr.data() + uint64_t(t) * HDR_BYTES;
        M[t].quantCnt = serial || K == 1 ? get<uint64_t>(c) : static_cast<uint64_t>(get<int>(c));
        uint8_t m = get<uint8_t>(c);
        M[t].method = m & 0x7F;
        M[t].delta = m & 0x80;
        get<uint16_t>(c);
        get<uint16_t>(c);
        M[t].nr = get<uint64_t>(c);
        if (M[t].method == 1) {   // nr holds the segment count S here
            M[t].payload = gOff[t] + (c - (hdr.data() + uint64_t(t) * HDR_BYTES));
            continue;
        }
        M[t].nv = get<uint64_t>(c);
        M[t].sRcl = get<uint64_t>(c);
        M[t].sVcl = get<uint64_t>(c);
        M[t].rbp = get<uint64_t>(c);
        M[t].vbp = get<uint64_t>(c);
        M[t].payload = gOff[t] + (c - (hdr.data() + uint64_t(t) * HDR_BYTES));
    }

    if (in.rejectMethod1)
        for (int t = 0; t < K; ++t)
            if (M[t].method == 1) return false;

    PlanParams pp;
    pp.N = N;
    std::copy_n(ih.dims, 4, pp.dims.begin());
    for (int i = N; i < 4; ++i) pp.dims[i] = 1;
    pp.sizeT = sizeof(T);
    size_t tc = K;
    for (int i = 1; i < N; ++i) tc = std::min(tc, ih.dims[i]);
    pp.threads = static_cast<int>(tc);
    pp.interp = ih.interpId;
    pp.direction = ih.direction;
    pp.anchorStride = ih.anchorStride;
    pp.eb = ih.eb;
    pp.alpha = ih.alpha;
    pp.beta = ih.beta;
    pp.avx = in.avx;
    pp.serial = serial;
    pp.fusedAll = serial || !in.conf->legacy;
    Plan P = build_plan(pp);
    for (int t = 0; t < K; ++t) {
        uint64_t expect = t < pp.threads ? P.streamLen[t] : 0;
        if (expect != M[t].quantCnt) throw std::runtime_error("plan disagrees with stream");
    }
    auto t3 = Clock::now();

    // Tight bounds use order-1 tANS: find the segments and chains of those streams.
    std::vector<uint32_t> segFirst(K, 0), segCnt(K, 0);
    uint32_t numSegs = 0;
    for (int t = 0; t < pp.threads; ++t) {
        segFirst[t] = numSegs;
        if (M[t].method == 1) {
            segCnt[t] = static_cast<uint32_t>(std::max<uint64_t>(M[t].nr, 1));
            numSegs += segCnt[t];
        }
    }
    std::vector<O1Seg> segs(numSegs);
    Dev<O1Seg> dsegs(numSegs);
    Dev<uint64_t> o1start(K), o1quant(K);
    Dev<uint32_t> dsegFirst(K), dsegCnt(K);
    if (numSegs) {
        std::vector<uint64_t> st(K), qc(K);
        for (int t = 0; t < K; ++t) {
            st[t] = M[t].payload;
            qc[t] = M[t].quantCnt;
        }
        CK(cudaMemcpy(o1start.p, st.data(), K * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(o1quant.p, qc.data(), K * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsegFirst.p, segFirst.data(), K * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsegCnt.p, segCnt.data(), K * 4, cudaMemcpyHostToDevice));
        o1_walk_kernel<<<K, 1>>>(raw.p, dsegs.p, o1start.p, dsegFirst.p, o1quant.p, dsegCnt.p);
        CK(cudaGetLastError());
        CK(cudaMemcpy(segs.data(), dsegs.p, numSegs * sizeof(O1Seg), cudaMemcpyDeviceToHost));
    }

    // Class bytes: one chain per run/value class array, one worker per 64K block.
    std::vector<FseChain> chains;
    uint64_t clsTotal = 0, totalBlks = 0;
    std::vector<uint64_t> rOff(K), vOff(K);
    for (int t = 0; t < pp.threads; ++t) {
        if (M[t].method != 0) continue;
        rOff[t] = clsTotal;
        chains.push_back({M[t].payload, clsTotal, M[t].nr, static_cast<uint32_t>(totalBlks), 0});
        totalBlks += (M[t].nr + FSE_FBLK - 1) / FSE_FBLK;
        clsTotal = (clsTotal + M[t].nr + 3) & ~uint64_t(3);
        vOff[t] = clsTotal;
        chains.push_back({M[t].payload + M[t].sRcl, clsTotal, M[t].nv, static_cast<uint32_t>(totalBlks), 0});
        totalBlks += (M[t].nv + FSE_FBLK - 1) / FSE_FBLK;
        clsTotal = (clsTotal + M[t].nv + 3) & ~uint64_t(3);
    }
    std::vector<uint64_t> segCls(numSegs, 0);
    for (uint32_t g = 0; g < numSegs; ++g) {
        if (segs[g].mode == 1 && segs[g].totov) {
            segCls[g] = clsTotal;
            chains.push_back({segs[g].ovfOff, clsTotal, segs[g].totov, static_cast<uint32_t>(totalBlks), 0});
            totalBlks += (segs[g].totov + FSE_FBLK - 1) / FSE_FBLK;
            clsTotal = (clsTotal + segs[g].totov + 3) & ~uint64_t(3);
        }
    }
    Dev<FseChain> dch(chains.size());
    Dev<FseBlk> blks(totalBlks);
    Dev<uint8_t> cls(clsTotal + 8);
    CK(cudaMemcpy(dch.p, chains.data(), chains.size() * sizeof(FseChain), cudaMemcpyHostToDevice));
    if (totalBlks) {
        fse_walk_kernel<<<static_cast<int>(chains.size() + FG_BLOCK - 1) / FG_BLOCK, FG_BLOCK>>>(raw.p, dch.p, blks.p, static_cast<int>(chains.size()));
        CK(cudaGetLastError());
        fse_block_kernel<<<static_cast<int>(totalBlks), 1>>>(raw.p, cls.p, blks.p);
        CK(cudaGetLastError());
    }
    CK(cudaDeviceSynchronize());
    auto t4 = Clock::now();

    // Run expansion. Bit offsets come from a prefix sum of the classes, value positions from a
    // prefix sum of the run lengths; both restart at every stream.
    uint64_t nstream = 0;
    for (auto l : P.streamLen) nstream += l;
    Dev<int16_t> stream(nstream);
    CK(cudaMemset(stream.p, 0, nstream * sizeof(int16_t)));
    Dev<uint64_t> bits(clsTotal), lens(clsTotal);
    uint64_t base = 0;
    std::vector<uint64_t> codeBase(K, 0);
    std::vector<RleChain> rc;
    std::vector<RleWork> rw, vw;
    for (int t = 0; t < pp.threads; ++t) {
        codeBase[t] = base;
        base += P.streamLen[t];
        if (M[t].method != 0) continue;
        const uint64_t rman = M[t].payload + M[t].sRcl + M[t].sVcl, vman = rman + M[t].rbp;
        const uint32_t id = static_cast<uint32_t>(rc.size());
        rc.push_back({rOff[t], vOff[t], M[t].nr, M[t].nv, 0, 0, 0, rman, vman, codeBase[t]});
        for (uint64_t o = 0; o < M[t].nr; o += FG_BLOCK) rw.push_back({id, static_cast<uint32_t>(o)});
        for (uint64_t o = 0; o < M[t].nv; o += FG_BLOCK) vw.push_back({id, static_cast<uint32_t>(o)});
    }
    if (clsTotal) {
        auto clsp = thrust::device_pointer_cast(cls.p);
        auto bitp = thrust::device_pointer_cast(bits.p);
        auto lenp = thrust::device_pointer_cast(lens.p);
        thrust::exclusive_scan(clsp, clsp + clsTotal, bitp, uint64_t(0));
        if (!rc.empty()) {
            Dev<RleChain> drc(rc.size());
            Dev<RleWork> drw(rw.size() + 1), dvw(vw.size() + 1);
            CK(cudaMemcpy(drc.p, rc.data(), rc.size() * sizeof(RleChain), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(drw.p, rw.data(), rw.size() * sizeof(RleWork), cudaMemcpyHostToDevice));
            if (!vw.empty()) CK(cudaMemcpy(dvw.p, vw.data(), vw.size() * sizeof(RleWork), cudaMemcpyHostToDevice));
            CK(cudaMemset(lens.p, 0, clsTotal * 8));
            if (!rw.empty()) {
                run_len_kernel<<<static_cast<int>(rw.size()), FG_BLOCK>>>(cls.p, bits.p, raw.p, lens.p, drc.p, drw.p);
                CK(cudaGetLastError());
            }
            if (!vw.empty()) {
                thrust::inclusive_scan(lenp, lenp + clsTotal, lenp);
                value_kernel<<<static_cast<int>(vw.size()), FG_BLOCK>>>(cls.p, bits.p, raw.p, lens.p, stream.p, drc.p, dvw.p);
                CK(cudaGetLastError());
            }
        }
    }
    for (int t = 0; t < pp.threads; ++t) {
        if (M[t].method == 0 && M[t].delta) {
            auto sp = thrust::device_pointer_cast(stream.p);
            thrust::inclusive_scan(sp + codeBase[t], sp + codeBase[t] + P.streamLen[t], sp + codeBase[t]);
        }
    }
    if (numSegs) {
        uint64_t ovTot = 0;
        std::vector<uint64_t> ovfBase(numSegs);
        for (uint32_t g = 0; g < numSegs; ++g) {
            ovfBase[g] = ovTot;
            ovTot += segs[g].totov;
        }
        Dev<int16_t> ovf(ovTot);
        Dev<uint64_t> dovb(numSegs), dcls(numSegs), chainEnds(numSegs * 16);
        Dev<uint32_t> tables(uint64_t(numSegs) * O1_NCTX * O1_CELLS);
        Dev<uint8_t> ctx(65536);
        auto lut = o1_ctx_lut();
        CK(cudaMemcpy(ctx.p, lut.data(), 65536, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dovb.p, ovfBase.data(), numSegs * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dcls.p, segCls.data(), numSegs * 8, cudaMemcpyHostToDevice));
        auto o0 = Clock::now();
        o1_table_kernel<<<static_cast<int>(numSegs) * O1_NCTX, 1>>>(raw.p, dsegs.p, tables.p);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        auto o1 = Clock::now();
        o1_ovf_kernel<<<static_cast<int>(numSegs), 1>>>(raw.p, dsegs.p, cls.p, dcls.p, dovb.p, ovf.p);
        CK(cudaGetLastError());
        std::vector<O1Chain> oc;
        std::vector<uint64_t> ends;
        for (int t = 0; t < pp.threads; ++t) {
            for (uint32_t g = segFirst[t]; g < segFirst[t] + segCnt[t]; ++g) {
                const O1Seg &s = segs[g];
                if (!s.N) continue;
                const uint64_t cs = (s.N + s.K - 1) / s.K;
                uint64_t ooff = ovfBase[g];
                for (uint32_t k = 0; k < s.K; ++k) {
                    O1Chain c{};
                    const uint64_t lo = k * cs;
                    c.count = lo >= s.N ? 0 : static_cast<uint32_t>(std::min(cs, s.N - lo));
                    c.out = codeBase[t] + s.lo + lo;
                    c.ovf = ooff;
                    c.tables = uint64_t(g) * O1_NCTX * O1_CELLS;
                    ooff += s.nvk[k];
                    oc.push_back(c);
                    ends.push_back(s.blen[k] ? s.chainOff[k] + s.blen[k] : 0);
                }
            }
        }
        Dev<O1Chain> doc(oc.size());
        Dev<uint64_t> dend(oc.size());
        CK(cudaMemcpy(doc.p, oc.data(), oc.size() * sizeof(O1Chain), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dend.p, ends.data(), oc.size() * 8, cudaMemcpyHostToDevice));
        o1_chain_init_kernel<<<static_cast<int>(oc.size() + FG_BLOCK - 1) / FG_BLOCK, FG_BLOCK>>>(raw.p, doc.p, dend.p, static_cast<int>(oc.size()));
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        auto o2 = Clock::now();
        o1_chain_kernel<<<static_cast<int>(oc.size()), 1>>>(raw.p, tables.p, doc.p, ovf.p, ctx.p, stream.p);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        auto o3 = Clock::now();
        if (getenv("FGPU_TIMES")) printf("  o1: tables %.2f ovf+setup %.2f chains %.2f ms, %zu chains\n", ms(o0, o1), ms(o1, o2), ms(o2, o3), oc.size());
        for (int t = 0; t < pp.threads; ++t) {
            if (M[t].method == 1 && M[t].delta) {
                auto sp = thrust::device_pointer_cast(stream.p);
                thrust::inclusive_scan(sp + codeBase[t], sp + codeBase[t] + P.streamLen[t], sp + codeBase[t]);
            }
        }
    }
    CK(cudaDeviceSynchronize());
    auto t5 = Clock::now();

    if (!in.refCodes.empty()) check_stages(in, P, K, M, raw.p, stream.p, codeBase);

    // Outliers: anchors first in each stream, then the escapes.
    uint64_t nanchor = 0, nun = 0;
    std::vector<uint64_t> perTid(pp.threads);
    for (int t = 0; t < pp.threads; ++t) {
        nanchor += P.anchorLen[t];
        perTid[t] = M[t].unpredCnt - P.anchorLen[t];
        nun += perTid[t];
    }
    Dev<U> anchors(nanchor), unpred(nun);
    {
        std::vector<CopyChunk> a, b;
        uint64_t ao = 0, uo = 0;
        for (int t = 0; t < pp.threads; ++t) {
            uint64_t s = M[t].rawBase + M[t].unpredOff;
            add_copy(a, s, ao * sizeof(T), P.anchorLen[t] * sizeof(T));
            add_copy(b, s + P.anchorLen[t] * sizeof(T), uo * sizeof(T), perTid[t] * sizeof(T));
            ao += P.anchorLen[t];
            uo += perTid[t];
        }
        launch_chunks(reinterpret_cast<uint8_t *>(anchors.p), raw.p, a);
        launch_chunks(reinterpret_cast<uint8_t *>(unpred.p), raw.p, b);
    }
    Dev<uint64_t> escPos(nun);
    Dev<U> escVal(nun);
    locate_escapes<T>(P, stream.p, nun, perTid, unpred.p, escPos, escVal);
    auto t6 = Clock::now();

    Dev<T> data(P.num);
    Dev<FgNest> nests(P.nests.size());
    Dev<FgWork> work(P.work.size());
    CK(cudaMemcpy(nests.p, P.nests.data(), P.nests.size() * sizeof(FgNest), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(work.p, P.work.data(), P.work.size() * sizeof(FgWork), cudaMemcpyHostToDevice));
    run_phases<T>(P, data.p, stream.p, nests.p, work.p, escPos.p, escVal.p, nun, anchors.p);
    auto t7 = Clock::now();
    if (out) CK(cudaMemcpy(out, data.p, P.num * sizeof(T), cudaMemcpyDeviceToHost));
    auto t8 = Clock::now();
    if (device) *device = data.release();
    if (tm && in.stats) {
        tm->streams = pp.threads;
        for (int t = 0; t < pp.threads; ++t) {
            tm->method1 += M[t].method == 1;
            tm->delta += M[t].delta;
        }
        for (auto &b : zblks) tm->zstdBlocks += b.type != 0;
        for (auto &g : segs) {
            tm->ovfRaw += g.totov && g.mode == 0;
            tm->ovfCoded += g.totov && g.mode == 1;
        }
        if (totalBlks) {
            std::vector<FseBlk> bl(totalBlks);
            CK(cudaMemcpy(bl.data(), blks.p, totalBlks * sizeof(FseBlk), cudaMemcpyDeviceToHost));
            for (auto &b : bl) (b.flag == 0 ? tm->fseRaw : b.flag == 2 ? tm->fseRle : tm->fseCoded)++;
        }
    }
    if (tm) {
        tm->parse = ms(t0, t1) + ms(t2z, t3);
        tm->upload = ms(t1, t2);
        tm->zstd = ms(t2, t2z);
        tm->fse = ms(t3, t4);
        tm->rle = ms(t4, t5);
        tm->escape = ms(t5, t6);
        tm->recon = ms(t6, t7);
        tm->download = ms(t7, t8);
    }
    return true;
}

}  // namespace

bool decode_pure(const PureInput &in, void *out, PureTiming *t, void **device) {
    return in.sizeT == 4 ? pure<float>(in, out, t, device) : pure<double>(in, out, t, device);
}
