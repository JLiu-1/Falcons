// FSE encoding, ported from zfse/fse_compress.c so a 64K block produces the same bytes as the CPU.
// Included by fgpu_encode.inl.

constexpr uint32_t FE_LOG_DEFAULT = 11, FE_LOG_MAX = 13, FE_LOG_MIN = 5;

__device__ __forceinline__ uint32_t hb32(uint32_t v) { return 31 - __clz(v); }

struct SymTT {
    int32_t deltaFindState;
    uint32_t deltaNbBits;
};

// Forward bit writer, same layout as BIT_CStream_t.
struct FwdBits {
    uint8_t *start, *ptr;
    uint64_t cont;
    uint32_t pos;
    __device__ __forceinline__ void init(uint8_t *dst) {
        start = ptr = dst;
        cont = 0;
        pos = 0;
    }
    __device__ __forceinline__ void add(uint64_t v, uint32_t nb) {
        cont |= (v & ((1ull << nb) - 1)) << pos;
        pos += nb;
    }
    __device__ __forceinline__ void flush() {
        const uint32_t n = pos >> 3;
        for (uint32_t i = 0; i < n; ++i) ptr[i] = static_cast<uint8_t>(cont >> (8 * i));
        ptr += n;
        pos &= 7;
        cont = n == 8 ? 0 : cont >> (n * 8);
    }
    __device__ inline uint32_t close() {
        add(1, 1);
        flush();
        if (pos) *ptr = static_cast<uint8_t>(cont);
        return static_cast<uint32_t>(ptr - start) + (pos > 0);
    }
};

__constant__ uint32_t FE_RTB[8] = {0, 473195, 504333, 520860, 550000, 700000, 750000, 830000};

__device__ inline uint32_t fe_min_log(uint64_t n, uint32_t maxSV) {
    const uint32_t a = hb32(static_cast<uint32_t>(n)) + 1, b = hb32(maxSV) + 2;
    return a < b ? a : b;
}

__device__ inline uint32_t fe_opt_log(uint32_t maxLog, uint64_t n, uint32_t maxSV) {
    uint32_t bits = hb32(static_cast<uint32_t>(n - 1)) - 2, log = maxLog;
    const uint32_t minBits = fe_min_log(n, maxSV);
    if (bits < log) log = bits;
    if (minBits > log) log = minBits;
    if (log < FE_LOG_MIN) log = FE_LOG_MIN;
    if (log > FE_LOG_MAX) log = FE_LOG_MAX;
    return log;
}

__device__ inline bool fe_norm_m2(int16_t *norm, uint32_t tableLog, const uint32_t *count, uint64_t total, uint32_t maxSV) {
    const int16_t NYA = -2;
    uint32_t distributed = 0;
    const uint32_t lowThreshold = static_cast<uint32_t>(total >> tableLog);
    uint32_t lowOne = static_cast<uint32_t>((total * 3) >> (tableLog + 1));
    for (uint32_t s = 0; s <= maxSV; ++s) {
        if (count[s] == 0) {
            norm[s] = 0;
        } else if (count[s] <= lowThreshold) {
            norm[s] = -1;
            distributed++;
            total -= count[s];
        } else if (count[s] <= lowOne) {
            norm[s] = 1;
            distributed++;
            total -= count[s];
        } else {
            norm[s] = NYA;
        }
    }
    uint32_t toDist = (1u << tableLog) - distributed;
    if (toDist == 0) return true;
    if ((total / toDist) > lowOne) {
        lowOne = static_cast<uint32_t>((total * 3) / (toDist * 2));
        for (uint32_t s = 0; s <= maxSV; ++s) {
            if (norm[s] == NYA && count[s] <= lowOne) {
                norm[s] = 1;
                distributed++;
                total -= count[s];
            }
        }
        toDist = (1u << tableLog) - distributed;
    }
    if (distributed == maxSV + 1) {
        uint32_t maxV = 0, maxC = 0;
        for (uint32_t s = 0; s <= maxSV; ++s)
            if (count[s] > maxC) {
                maxV = s;
                maxC = count[s];
            }
        norm[maxV] += static_cast<int16_t>(toDist);
        return true;
    }
    if (total == 0) {
        for (uint32_t s = 0; toDist > 0; s = (s + 1) % (maxSV + 1))
            if (norm[s] > 0) {
                toDist--;
                norm[s]++;
            }
        return true;
    }
    const uint64_t vStepLog = 62 - tableLog, mid = (1ull << (vStepLog - 1)) - 1;
    const uint64_t rStep = (((1ull << vStepLog) * toDist) + mid) / total;
    uint64_t tmp = mid;
    for (uint32_t s = 0; s <= maxSV; ++s) {
        if (norm[s] == NYA) {
            const uint64_t end = tmp + (count[s] * rStep);
            const uint32_t weight = static_cast<uint32_t>(end >> vStepLog) - static_cast<uint32_t>(tmp >> vStepLog);
            if (weight < 1) return false;
            norm[s] = static_cast<int16_t>(weight);
            tmp = end;
        }
    }
    return true;
}

// Returns false where FSE_normalizeCount reports an error.
__device__ inline bool fe_normalize(int16_t *norm, uint32_t tableLog, const uint32_t *count, uint64_t total, uint32_t maxSV) {
    const uint64_t scale = 62 - tableLog, step = (1ull << 62) / total, vStep = 1ull << (scale - 20);
    int still = 1 << tableLog;
    uint32_t largest = 0;
    int16_t largestP = 0;
    const uint32_t lowThreshold = static_cast<uint32_t>(total >> tableLog);
    for (uint32_t s = 0; s <= maxSV; ++s) {
        if (count[s] == total) return true;
        if (count[s] == 0) {
            norm[s] = 0;
            continue;
        }
        if (count[s] <= lowThreshold) {
            norm[s] = -1;
            still--;
        } else {
            int16_t p = static_cast<int16_t>((count[s] * step) >> scale);
            if (p < 8) {
                const uint64_t rest = vStep * FE_RTB[p];
                p += (count[s] * step) - (static_cast<uint64_t>(p) << scale) > rest;
            }
            if (p > largestP) {
                largestP = p;
                largest = s;
            }
            norm[s] = p;
            still -= p;
        }
    }
    if (-still >= (norm[largest] >> 1)) return fe_norm_m2(norm, tableLog, count, total, maxSV);
    norm[largest] += static_cast<int16_t>(still);
    return true;
}

__device__ inline uint32_t fe_write_ncount(uint8_t *out0, const int16_t *norm, uint32_t maxSV, uint32_t tableLog) {
    uint8_t *out = out0;
    const int tableSize = 1 << tableLog;
    uint32_t bits = tableLog - FE_LOG_MIN;
    int bitCount = 4, remaining = tableSize + 1, threshold = tableSize, nbBits = tableLog + 1;
    uint32_t symbol = 0;
    const uint32_t alphabet = maxSV + 1;
    int previousIs0 = 0;
    auto emit16 = [&]() {
        out[0] = static_cast<uint8_t>(bits);
        out[1] = static_cast<uint8_t>(bits >> 8);
        out += 2;
        bits >>= 16;
    };
    while (symbol < alphabet && remaining > 1) {
        if (previousIs0) {
            uint32_t start = symbol;
            while (symbol < alphabet && !norm[symbol]) symbol++;
            if (symbol == alphabet) break;
            while (symbol >= start + 24) {
                start += 24;
                bits += 0xFFFFu << bitCount;
                emit16();
            }
            while (symbol >= start + 3) {
                start += 3;
                bits += 3u << bitCount;
                bitCount += 2;
            }
            bits += (symbol - start) << bitCount;
            bitCount += 2;
            if (bitCount > 16) {
                emit16();
                bitCount -= 16;
            }
        }
        int count = norm[symbol++];
        const int max = (2 * threshold - 1) - remaining;
        remaining -= count < 0 ? -count : count;
        count++;
        if (count >= threshold) count += max;
        bits += static_cast<uint32_t>(count) << bitCount;
        bitCount += nbBits;
        bitCount -= count < max;
        previousIs0 = count == 1;
        while (remaining < threshold) {
            nbBits--;
            threshold >>= 1;
        }
        if (bitCount > 16) {
            emit16();
            bitCount -= 16;
        }
    }
    out[0] = static_cast<uint8_t>(bits);
    out[1] = static_cast<uint8_t>(bits >> 8);
    out += (bitCount + 7) / 8;
    return static_cast<uint32_t>(out - out0);
}

// tableSymbol: 1 << tableLog bytes, stateTable: same count of u16, tt: maxSV + 1 entries.
__device__ inline void fe_build_ctable(uint16_t *stateTable, SymTT *tt, uint8_t *tableSymbol, const int16_t *norm, uint32_t maxSV, uint32_t tableLog) {
    const uint32_t tableSize = 1u << tableLog, mask = tableSize - 1, step = (tableSize >> 1) + (tableSize >> 3) + 3;
    uint32_t cumul[258];
    uint32_t high = tableSize - 1;
    cumul[0] = 0;
    for (uint32_t u = 1; u <= maxSV + 1; ++u) {
        if (norm[u - 1] == -1) {
            cumul[u] = cumul[u - 1] + 1;
            tableSymbol[high--] = static_cast<uint8_t>(u - 1);
        } else {
            cumul[u] = cumul[u - 1] + norm[u - 1];
        }
    }
    cumul[maxSV + 1] = tableSize + 1;
    uint32_t position = 0;
    for (uint32_t s = 0; s <= maxSV; ++s) {
        const int freq = norm[s];
        for (int i = 0; i < freq; ++i) {
            tableSymbol[position] = static_cast<uint8_t>(s);
            position = (position + step) & mask;
            while (position > high) position = (position + step) & mask;
        }
    }
    for (uint32_t u = 0; u < tableSize; ++u) {
        const uint8_t s = tableSymbol[u];
        stateTable[cumul[s]++] = static_cast<uint16_t>(tableSize + u);
    }
    uint32_t total = 0;
    for (uint32_t s = 0; s <= maxSV; ++s) {
        const int f = norm[s];
        if (f == 0) {
            tt[s].deltaFindState = 0;
            tt[s].deltaNbBits = ((tableLog + 1) << 16) - (1u << tableLog);
        } else if (f == -1 || f == 1) {
            tt[s].deltaNbBits = (tableLog << 16) - (1u << tableLog);
            tt[s].deltaFindState = static_cast<int32_t>(total) - 1;
            total++;
        } else {
            const uint32_t maxBitsOut = tableLog - hb32(f - 1), minStatePlus = static_cast<uint32_t>(f) << maxBitsOut;
            tt[s].deltaNbBits = (maxBitsOut << 16) - minStatePlus;
            tt[s].deltaFindState = static_cast<int32_t>(total) - f;
            total += f;
        }
    }
}

__device__ __forceinline__ uint64_t fe_init2(const uint16_t *stateTable, const SymTT *tt, uint32_t sym) {
    const SymTT t = tt[sym];
    const uint32_t nb = (t.deltaNbBits + (1u << 15)) >> 16;
    const uint64_t v = (static_cast<uint64_t>(nb) << 16) - t.deltaNbBits;
    return stateTable[(v >> nb) + t.deltaFindState];
}

__device__ __forceinline__ void fe_encode(FwdBits &b, uint64_t &state, const uint16_t *stateTable, const SymTT *tt, uint32_t sym) {
    const SymTT t = tt[sym];
    const uint32_t nb = static_cast<uint32_t>((state + t.deltaNbBits) >> 16);
    b.add(state, nb);
    state = stateTable[(state >> nb) + t.deltaFindState];
}

constexpr uint32_t FE_SCRATCH = (1u << FE_LOG_MAX) * 3 + 256 * sizeof(SymTT);

// FSE_compress. Returns 0 (keep raw), 1 (single symbol) or the compressed size.
__device__ inline uint32_t fe_compress(uint8_t *dst, const uint8_t *src, uint32_t n, uint8_t *scratch) {
    if (n <= 1) return 0;
    uint32_t count[256];
    for (int i = 0; i < 256; ++i) count[i] = 0;
    for (uint32_t i = 0; i < n; ++i) count[src[i]]++;
    uint32_t maxSV = 255, maxCount = 0;
    while (count[maxSV] == 0) maxSV--;
    for (uint32_t s = 0; s <= maxSV; ++s)
        if (count[s] > maxCount) maxCount = count[s];
    if (maxCount == n) return 1;
    if (maxCount == 1) return 0;
    if (maxCount < (n >> 7)) return 0;
    const uint32_t tableLog = fe_opt_log(FE_LOG_DEFAULT, n, maxSV);
    int16_t norm[256];
    fe_normalize(norm, tableLog, count, n, maxSV);
    uint8_t *op = dst + fe_write_ncount(dst, norm, maxSV, tableLog);

    uint16_t *stateTable = reinterpret_cast<uint16_t *>(scratch);
    SymTT *tt = reinterpret_cast<SymTT *>(scratch + 2 * (1u << FE_LOG_MAX));
    uint8_t *tableSymbol = scratch + 2 * (1u << FE_LOG_MAX) + 256 * sizeof(SymTT);
    fe_build_ctable(stateTable, tt, tableSymbol, norm, maxSV, tableLog);
    if (n <= 2) return 0;

    FwdBits b;
    b.init(op);
    uint32_t ip = n, size = n;
    uint64_t s1, s2;
    if (n & 1) {
        s1 = fe_init2(stateTable, tt, src[--ip]);
        s2 = fe_init2(stateTable, tt, src[--ip]);
        fe_encode(b, s1, stateTable, tt, src[--ip]);
        b.flush();
    } else {
        s2 = fe_init2(stateTable, tt, src[--ip]);
        s1 = fe_init2(stateTable, tt, src[--ip]);
    }
    size -= 2;
    if (size & 2) {
        fe_encode(b, s2, stateTable, tt, src[--ip]);
        fe_encode(b, s1, stateTable, tt, src[--ip]);
        b.flush();
    }
    while (ip > 0) {
        fe_encode(b, s2, stateTable, tt, src[--ip]);
        fe_encode(b, s1, stateTable, tt, src[--ip]);
        fe_encode(b, s2, stateTable, tt, src[--ip]);
        fe_encode(b, s1, stateTable, tt, src[--ip]);
        b.flush();
    }
    b.add(s2, tableLog);
    b.flush();
    b.add(s1, tableLog);
    b.flush();
    const uint32_t c = b.close();
    if (c == 0) return 0;
    const uint32_t total = static_cast<uint32_t>(op - dst) + c;
    return total >= n - 1 ? 0 : total;
}
