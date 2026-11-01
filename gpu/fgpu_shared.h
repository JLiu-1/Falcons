// POD types shared between the host plan builder and the CUDA kernels.
#pragma once
#include <cstdint>

constexpr int FG_BLOCK = 256;
constexpr int16_t FG_ESCAPE = -32768;

enum FgOp : uint8_t {
    FG_ZERO,     // root point: pred = 0
    FG_EQUAL,
    FG_LINEAR,
    FG_LINEAR1,
    FG_QUAD1,
    FG_QUAD2,
    FG_CUBIC,
    FG_ANCHOR,   // value comes from the anchor array, no prediction
};

// A rectangular loop nest of target points. Iteration i0..i3 (i3 innermost) maps to
//   data  index = data0 + sum(i_k * dstr[k])
//   stream pos  = pos0  + sum(i_k * pstr[k])
struct FgNest {
    uint64_t data0;
    uint64_t pos0;
    int64_t dstr[4];
    uint64_t pstr[4];
    int64_t opnd[4];   // operand element offsets from the target
    uint32_t cnt[4];
    uint32_t points;
    uint32_t tid;      // plan-time stream id, consumed when pos0 is rebased
    uint8_t op;
    uint8_t fused;     // 1: AVX lane, fma(q, ebx2, pred). 0: pred + q * ebx2
    uint8_t pad[6];
};

struct FgWork {
    uint32_t nest;
    uint32_t start;
};

// One kernel launch. All nests of a phase are independent of each other.
struct FgPhase {
    double ebx2, ebx2r;
    double ebT;    // error bound for comparing a value of the element type, rounded toward zero
    uint32_t nest0, nestN;
    uint64_t work0, workN;
    uint64_t points;
};
