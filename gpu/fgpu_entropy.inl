// Device-side zstd block decoding and order-1 tANS. Included by fgpu_pure.inl.
//
// Falcons streams barely use zstd: blocks are stored raw except usually the first one of a
// stream, so one thread per stream is enough for the compressed blocks.

// Reads at negative offsets return zero bits, as the zstd format requires at stream ends.
struct SlowBits {
    const uint8_t *p;
    int64_t off;
    __device__ inline void init(const uint8_t *src, uint32_t size) {
        p = src;
        off = int64_t(size - 1) * 8 + (31 - __clz(uint32_t(src[size - 1])));
    }
    __device__ inline uint64_t read(uint32_t nb) {
        off -= nb;
        int64_t ao = off;
        int ab = nb;
        if (off < 0) {
            ab += static_cast<int>(off);
            ao = 0;
        }
        uint64_t res = 0;
        if (ab > 0) {
            const uint32_t sh = ao & 7, need = (sh + ab + 7) >> 3;
            const uint8_t *b = p + (ao >> 3);
            for (uint32_t i = 0; i < need; ++i) res |= uint64_t(b[i]) << (8 * i);
            res = (res >> sh) & ((1ull << ab) - 1);
        }
        if (off < 0 && off > -64) res <<= -off;
        return res;
    }
};

struct ZBlk {
    uint64_t src;
    uint64_t dst, dstEnd;   // regenerated range in raw
    uint32_t size;          // compressed size; regenerated size for RLE blocks
    uint32_t type;
};

struct ZStream {
    uint32_t first, count;
};

constexpr uint32_t ZS_LIT = 128 * 1024 + 64;
constexpr uint32_t ZS_HUF = 2048;
constexpr uint32_t ZS_LL = 512, ZS_OF = 256, ZS_ML = 512;
constexpr uint32_t ZSCRATCH = ZS_LIT + 2 * ZS_HUF + 4 * (ZS_LL + ZS_OF + ZS_ML + 64);

__device__ inline void lz_copy(uint8_t *op, uint64_t dist, uint32_t len) {
    const uint8_t *m = op - dist;
    for (uint32_t i = 0; i < len; ++i) op[i] = m[i];
}

__device__ inline uint32_t huf_read_table(const uint8_t *src, uint8_t *sym, uint8_t *nbits, uint32_t *fseT, uint32_t &maxBits) {
    uint8_t weight[256];
    uint32_t n = 0, used;
    const uint32_t hb = src[0];
    if (hb >= 128) {
        n = hb - 127;
        for (uint32_t i = 0; i < n; ++i) {
            const uint8_t b = src[1 + i / 2];
            weight[i] = i & 1 ? b & 15 : b >> 4;
        }
        used = 1 + (n + 1) / 2;
    } else {
        int16_t norm[256];
        uint32_t maxSV = 255, tl;
        const uint32_t hl = fse_read_ncount(norm, maxSV, tl, src + 1, hb);
        fse_build_table(fseT, norm, maxSV, tl);
        SlowBits br;
        br.init(src + 1 + hl, hb - hl);
        uint32_t st[2];
        st[0] = static_cast<uint32_t>(br.read(tl));
        st[1] = static_cast<uint32_t>(br.read(tl));
        while (true) {
            const uint32_t c0 = fseT[st[0]];
            weight[n++] = (c0 >> 16) & 0xFF;
            st[0] = (c0 & 0xFFFF) + static_cast<uint32_t>(br.read(c0 >> 24));
            if (br.off < 0) {
                weight[n++] = (fseT[st[1]] >> 16) & 0xFF;
                break;
            }
            const uint32_t c1 = fseT[st[1]];
            weight[n++] = (c1 >> 16) & 0xFF;
            st[1] = (c1 & 0xFFFF) + static_cast<uint32_t>(br.read(c1 >> 24));
            if (br.off < 0) {
                weight[n++] = (fseT[st[0]] >> 16) & 0xFF;
                break;
            }
        }
        used = 1 + hb;
    }
    uint32_t sum = 0;
    for (uint32_t i = 0; i < n; ++i) sum += weight[i] ? 1u << (weight[i] - 1) : 0;
    maxBits = 31 - __clz(sum) + 1;
    const uint32_t left = (1u << maxBits) - sum;
    weight[n++] = (31 - __clz(left)) + 1;

    uint32_t rankCount[13] = {0}, rankIdx[14];
    uint8_t bitsOf[256];
    for (uint32_t i = 0; i < n; ++i) {
        bitsOf[i] = weight[i] ? maxBits + 1 - weight[i] : 0;
        rankCount[bitsOf[i]]++;
    }
    rankIdx[maxBits] = 0;
    for (uint32_t i = maxBits; i >= 1; --i) {
        rankIdx[i - 1] = rankIdx[i] + rankCount[i] * (1u << (maxBits - i));
        for (uint32_t j = rankIdx[i]; j < rankIdx[i - 1]; ++j) nbits[j] = i;
    }
    for (uint32_t i = 0; i < n; ++i) {
        if (!bitsOf[i]) continue;
        const uint32_t len = 1u << (maxBits - bitsOf[i]);
        for (uint32_t j = 0; j < len; ++j) sym[rankIdx[bitsOf[i]] + j] = i;
        rankIdx[bitsOf[i]] += len;
    }
    return used;
}

__device__ inline void huf_stream(const uint8_t *src, uint32_t size, uint8_t *out, uint32_t count, const uint8_t *sym, const uint8_t *nbits, uint32_t maxBits) {
    SlowBits br;
    br.init(src, size);
    const uint32_t mask = (1u << maxBits) - 1;
    uint32_t state = static_cast<uint32_t>(br.read(maxBits));
    for (uint32_t i = 0; i < count; ++i) {
        out[i] = sym[state];
        const uint32_t b = nbits[state];
        state = ((state << b) + static_cast<uint32_t>(br.read(b))) & mask;
    }
}

__constant__ int16_t LL_DEF[36] = {4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1};
__constant__ int16_t OF_DEF[29] = {1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1};
__constant__ int16_t ML_DEF[53] = {1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1};
__constant__ uint32_t LL_BASE[36] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28, 32, 40, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
__constant__ uint8_t LL_BITS[36] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
__constant__ uint32_t ML_BASE[53] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027, 2051, 4099, 8195, 16387, 32771, 65539};
__constant__ uint8_t ML_BITS[53] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

// Returns bytes consumed, 0 on an invalid mode.
__device__ inline uint32_t seq_table(int mode, const uint8_t *src, uint32_t avail, uint32_t *table, uint32_t &log, const __constant__ int16_t *def, uint32_t defSyms, uint32_t defLog) {
    if (mode == 0) {
        int16_t norm[256];
        for (uint32_t i = 0; i < defSyms; ++i) norm[i] = def[i];
        fse_build_table(table, norm, defSyms - 1, defLog);
        log = defLog;
        return 0;
    }
    if (mode == 1) {
        table[0] = uint32_t(src[0]) << 16;
        log = 0;
        return 1;
    }
    if (mode == 2) {
        int16_t norm[256];
        uint32_t maxSV = 255, tl;
        const uint32_t hl = fse_read_ncount(norm, maxSV, tl, src, avail);
        fse_build_table(table, norm, maxSV, tl);
        log = tl;
        return hl;
    }
    return 0;  // repeat: keep the previous table
}

__global__ void zstd_kernel(uint8_t *__restrict__ raw, const uint8_t *__restrict__ cmp, const ZStream *__restrict__ streams, const ZBlk *__restrict__ blks, uint8_t *__restrict__ scratch, uint32_t *__restrict__ bad) {
    const uint32_t id = blockIdx.x;
    const ZStream S = streams[id];
    uint8_t *sc = scratch + uint64_t(id) * ZSCRATCH;
    uint8_t *lit = sc;
    uint8_t *hufSym = sc + ZS_LIT, *hufBits = hufSym + ZS_HUF;
    uint32_t *llT = reinterpret_cast<uint32_t *>(hufBits + ZS_HUF);
    uint32_t *ofT = llT + ZS_LL, *mlT = ofT + ZS_OF, *wT = mlT + ZS_ML;
    uint32_t hufMax = 0, llLog = 0, ofLog = 0, mlLog = 0;
    uint64_t rep[3] = {1, 4, 8};

    for (uint32_t b = 0; b < S.count; ++b) {
        const ZBlk B = blks[S.first + b];
        const uint8_t *ip = cmp + B.src;
        uint8_t *op = raw + B.dst;
        if (B.type == 1) {
            for (uint32_t i = 0; i < B.size; ++i) op[i] = ip[0];
            continue;
        }
        // Literals section
        const uint8_t h0 = ip[0];
        const uint32_t ltype = h0 & 3, sf = (h0 >> 2) & 3;
        uint32_t regen, csize = 0, hsz;
        bool four = false;
        if (ltype < 2) {
            if ((sf & 1) == 0) {
                regen = h0 >> 3;
                hsz = 1;
            } else if (sf == 1) {
                regen = (h0 >> 4) | (uint32_t(ip[1]) << 4);
                hsz = 2;
            } else {
                regen = (h0 >> 4) | (uint32_t(ip[1]) << 4) | (uint32_t(ip[2]) << 12);
                hsz = 3;
            }
        } else if (sf < 2) {
            const uint32_t v = h0 | (uint32_t(ip[1]) << 8) | (uint32_t(ip[2]) << 16);
            regen = (v >> 4) & 0x3FF;
            csize = (v >> 14) & 0x3FF;
            hsz = 3;
            four = sf == 1;
        } else if (sf == 2) {
            const uint32_t v = rd32le(ip);
            regen = (v >> 4) & 0x3FFF;
            csize = v >> 18;
            hsz = 4;
            four = true;
        } else {
            const uint64_t v = rd32le(ip) | (uint64_t(ip[4]) << 32);
            regen = static_cast<uint32_t>(v >> 4) & 0x3FFFF;
            csize = static_cast<uint32_t>(v >> 22) & 0x3FFFF;
            hsz = 5;
            four = true;
        }
        ip += hsz;
        if (ltype == 0) {
            for (uint32_t i = 0; i < regen; ++i) lit[i] = ip[i];
            ip += regen;
        } else if (ltype == 1) {
            for (uint32_t i = 0; i < regen; ++i) lit[i] = ip[0];
            ip += 1;
        } else {
            const uint8_t *lsrc = ip;
            uint32_t lsize = csize;
            if (ltype == 2) {
                const uint32_t used = huf_read_table(lsrc, hufSym, hufBits, wT, hufMax);
                lsrc += used;
                lsize -= used;
            }
            if (!four) {
                huf_stream(lsrc, lsize, lit, regen, hufSym, hufBits, hufMax);
            } else {
                const uint32_t s0 = lsrc[0] | (uint32_t(lsrc[1]) << 8), s1 = lsrc[2] | (uint32_t(lsrc[3]) << 8), s2 = lsrc[4] | (uint32_t(lsrc[5]) << 8);
                const uint32_t s3 = lsize - 6 - s0 - s1 - s2;
                const uint32_t each = (regen + 3) / 4;
                const uint8_t *q = lsrc + 6;
                huf_stream(q, s0, lit, each, hufSym, hufBits, hufMax);
                q += s0;
                huf_stream(q, s1, lit + each, each, hufSym, hufBits, hufMax);
                q += s1;
                huf_stream(q, s2, lit + 2 * each, each, hufSym, hufBits, hufMax);
                q += s2;
                huf_stream(q, s3, lit + 3 * each, regen - 3 * each, hufSym, hufBits, hufMax);
            }
            ip += csize;
        }
        // Sequences section
        const uint8_t *bend = cmp + B.src + B.size;
        uint32_t nseq = ip[0];
        if (nseq < 128) {
            ip += 1;
        } else if (nseq < 255) {
            nseq = ((nseq - 128) << 8) + ip[1];
            ip += 2;
        } else {
            nseq = ip[1] + (uint32_t(ip[2]) << 8) + 0x7F00;
            ip += 3;
        }
        uint32_t lp = 0;
        if (nseq) {
            const uint8_t modes = *ip++;
            ip += seq_table(modes >> 6, ip, static_cast<uint32_t>(bend - ip), llT, llLog, LL_DEF, 36, 6);
            ip += seq_table((modes >> 4) & 3, ip, static_cast<uint32_t>(bend - ip), ofT, ofLog, OF_DEF, 29, 5);
            ip += seq_table((modes >> 2) & 3, ip, static_cast<uint32_t>(bend - ip), mlT, mlLog, ML_DEF, 53, 6);
            SlowBits br;
            br.init(ip, static_cast<uint32_t>(bend - ip));
            uint32_t sl = static_cast<uint32_t>(br.read(llLog));
            uint32_t so = static_cast<uint32_t>(br.read(ofLog));
            uint32_t sm = static_cast<uint32_t>(br.read(mlLog));
            for (uint32_t n = 0; n < nseq; ++n) {
                const uint32_t cl = llT[sl], co = ofT[so], cm = mlT[sm];
                const uint32_t ofc = (co >> 16) & 0xFF, llc = (cl >> 16) & 0xFF, mlc = (cm >> 16) & 0xFF;
                uint64_t ofv = (1ull << ofc) + br.read(ofc);
                const uint32_t ml = ML_BASE[mlc] + static_cast<uint32_t>(br.read(ML_BITS[mlc]));
                const uint32_t ll = LL_BASE[llc] + static_cast<uint32_t>(br.read(LL_BITS[llc]));
                if (n + 1 < nseq) {
                    sl = (cl & 0xFFFF) + static_cast<uint32_t>(br.read(cl >> 24));
                    sm = (cm & 0xFFFF) + static_cast<uint32_t>(br.read(cm >> 24));
                    so = (co & 0xFFFF) + static_cast<uint32_t>(br.read(co >> 24));
                }
                uint64_t dist;
                if (ofv <= 3) {
                    const uint32_t idx = static_cast<uint32_t>(ofv) - 1 + (ll == 0 ? 1 : 0);
                    dist = idx == 3 ? rep[0] - 1 : rep[idx];
                    if (idx > 1) rep[2] = rep[1];
                    if (idx > 0) {
                        rep[1] = rep[0];
                        rep[0] = dist;
                    }
                } else {
                    dist = ofv - 3;
                    rep[2] = rep[1];
                    rep[1] = rep[0];
                    rep[0] = dist;
                }
                for (uint32_t i = 0; i < ll; ++i) op[i] = lit[lp + i];
                lp += ll;
                op += ll;
                lz_copy(op, dist, ml);
                op += ml;
            }
        }
        for (uint32_t i = lp; i < regen; ++i) *op++ = lit[i];
        if (static_cast<uint64_t>(op - raw) != B.dstEnd) bad[id] = 1;
    }
}

// ---- order-1 tANS (method 1) ----

constexpr int O1_NCTX = 15, O1_PB = 13, O1_NSYM = 256;
constexpr uint32_t O1_CELLS = 1u << O1_PB;
constexpr int O1_NFBYTES = O1_NCTX * O1_NSYM * 2;

struct O1Seg {
    uint64_t lo;          // first code, relative to the stream
    uint64_t N;
    uint64_t nfOff;       // normalized frequencies in raw
    uint64_t chainOff[16];
    uint64_t blen[16];
    uint64_t nvk[16];
    uint64_t ovfOff;      // first byte after the mode byte
    uint64_t totov;
    uint64_t sVcl, vbp;
    uint64_t next;
    uint32_t K, mode;
};

__device__ __forceinline__ uint64_t rd64le(const uint8_t *p) { return rd32le(p) | (uint64_t(rd32le(p + 4)) << 32); }

__global__ void o1_walk_kernel(const uint8_t *__restrict__ raw, O1Seg *__restrict__ segs, const uint64_t *__restrict__ start, const uint32_t *__restrict__ segFirst, const uint64_t *__restrict__ quantCnt, const uint32_t *__restrict__ segCnt) {
    const uint32_t t = blockIdx.x;
    const uint32_t S = segCnt[t];
    if (!S) return;
    const uint64_t Nall = quantCnt[t];
    const uint64_t seg = (Nall + S - 1) / S;
    uint64_t p = start[t];
    for (uint32_t g = 0; g < S; ++g) {
        O1Seg s{};
        s.lo = g * seg;
        uint64_t hi = s.lo + seg;
        if (hi > Nall) hi = Nall;
        s.N = hi - s.lo;
        const uint32_t K = static_cast<uint32_t>(rd64le(raw + p + 8));
        s.K = K;
        p += 16;
        for (uint32_t k = 0; k < K; ++k, p += 8) s.blen[k] = rd64le(raw + p);
        for (uint32_t k = 0; k < K; ++k, p += 8) {
            s.nvk[k] = rd64le(raw + p);
            s.totov += s.nvk[k];
        }
        s.nfOff = p;
        p += O1_NFBYTES;
        for (uint32_t k = 0; k < K; ++k) {
            s.chainOff[k] = p;
            p += s.blen[k];
        }
        s.mode = raw[p++];
        if (s.totov) {
            if (s.mode == 0) {
                s.ovfOff = p;
                p += s.totov * 2;
            } else {
                s.sVcl = rd64le(raw + p);
                s.vbp = rd64le(raw + p + 8);
                p += 16;
                s.ovfOff = p;
                p += s.sVcl + s.vbp;
            }
        }
        s.next = p;
        segs[segFirst[t] + g] = s;
    }
}

__global__ void o1_table_kernel(const uint8_t *__restrict__ raw, const O1Seg *__restrict__ segs, uint32_t *__restrict__ tables) {
    const uint32_t seg = blockIdx.x / O1_NCTX, c = blockIdx.x % O1_NCTX;
    const uint8_t *nf = raw + segs[seg].nfOff + uint64_t(c) * O1_NSYM * 2;
    int16_t norm[O1_NSYM];
    for (int s = 0; s < O1_NSYM; ++s) norm[s] = static_cast<int16_t>(nf[2 * s] | (uint16_t(nf[2 * s + 1]) << 8));
    fse_build_table(tables + (uint64_t(seg) * O1_NCTX + c) * O1_CELLS, norm, O1_NSYM - 1, O1_PB);
}

// Overflow codes of one segment: raw int16, or classes (already FSE-decoded into cls) + mantissas.
__global__ void o1_ovf_kernel(const uint8_t *__restrict__ raw, const O1Seg *__restrict__ segs, const uint8_t *__restrict__ cls, const uint64_t *__restrict__ clsOff, const uint64_t *__restrict__ ovfBase, int16_t *__restrict__ ovf) {
    const uint32_t id = blockIdx.x;
    const O1Seg s = segs[id];
    if (!s.totov) return;
    int16_t *out = ovf + ovfBase[id];
    if (s.mode == 0) {
        for (uint64_t i = 0; i < s.totov; ++i) out[i] = static_cast<int16_t>(raw[s.ovfOff + 2 * i] | (uint16_t(raw[s.ovfOff + 2 * i + 1]) << 8));
        return;
    }
    const uint8_t *man = raw + s.ovfOff + s.sVcl;
    const uint8_t *cl = cls + clsOff[id];
    uint64_t bit = 0;
    for (uint64_t i = 0; i < s.totov; ++i) {
        const uint32_t vk = cl[i];
        const uint32_t vv = (1u << vk) | static_cast<uint32_t>(read_msb(man, bit, vk));
        bit += vk;
        const uint32_t zz = vv - 1;
        out[i] = static_cast<int16_t>((zz >> 1) ^ (~(zz & 1) + 1));
    }
}

struct O1Chain {
    uint64_t bitEnd;      // absolute bit position of the start marker in raw
    uint64_t out;         // absolute position in the code array
    uint64_t ovf;         // first overflow code of this chain
    uint64_t tables;      // first context table of the segment, in cells
    uint32_t count;
    uint32_t pad;
};

__global__ void o1_chain_kernel(const uint8_t *__restrict__ raw, const uint32_t *__restrict__ tables, const O1Chain *__restrict__ chains, const int16_t *__restrict__ ovf, const uint8_t *__restrict__ ctxLut, int16_t *__restrict__ codes) {
    __shared__ uint32_t decoded[O1_NSYM];   // per symbol: code in the high half, context * cells in the low bits
    const O1Chain c = chains[blockIdx.x];
    if (!c.count) return;
    for (uint32_t s = 0; s < O1_NSYM - 1; ++s) {
        const int code = static_cast<int>((s >> 1) ^ (~(s & 1) + 1));
        decoded[s] = (static_cast<uint32_t>(code) << 16) | (uint32_t(ctxLut[static_cast<uint16_t>(code)]) * 8192 >> 13);
    }
    BackCache br;
    br.init(raw, c.bitEnd);
    const uint32_t *T = tables + c.tables;
    uint32_t state = br.read(O1_PB);
    uint32_t ctx = 0;
    uint64_t oi = c.ovf;
    int16_t *out = codes + c.out;
    for (uint32_t i = 0; i < c.count; ++i) {
        const uint32_t cell = T[(ctx * O1_CELLS) + state];
        const uint32_t sym = (cell >> 16) & 0xFF;
        state = (cell & 0xFFFF) + br.read(cell >> 24);
        if (sym != O1_NSYM - 1) {
            const uint32_t d = decoded[sym];
            out[i] = static_cast<int16_t>(d >> 16);
            ctx = d & 0xFFFF;
        } else {
            const int16_t code = ovf[oi++];
            out[i] = code;
            ctx = ctxLut[static_cast<uint16_t>(code)];
        }
    }
}

// The bit reader must stop at the sentinel of the chain's last byte.
__global__ void o1_chain_init_kernel(const uint8_t *__restrict__ raw, O1Chain *__restrict__ chains, const uint64_t *__restrict__ ends, int count) {
    const uint32_t id = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= uint32_t(count)) return;
    const uint64_t end = ends[id];
    if (end) chains[id].bitEnd = (end - 1) * 8 + (31 - __clz(uint32_t(raw[end - 1])));
}

inline std::vector<uint8_t> o1_ctx_lut() {
    std::vector<uint8_t> ctx(65536);
    constexpr int B = (O1_NCTX - 1) / 2, Bf = (B + 1) / 2;
    for (int u = 0; u < 65536; ++u) {
        int ctxv, code = static_cast<int16_t>(u);
        int sgn = code < 0;
        int a = sgn ? -code : code;
        if (a == 0) {
            ctxv = 0;
        } else {
            int m;
            if (a <= Bf) {
                m = a - 1;
            } else {
                m = Bf - 1;
                int v = Bf;
                while (v < a && m < B - 1) {
                    v += (v >> 1) + 1;
                    m++;
                }
            }
            if (m >= B) m = B - 1;
            ctxv = 1 + m * 2 + sgn;
        }
        ctx[u] = static_cast<uint8_t>(ctxv);
    }
    return ctx;
}
