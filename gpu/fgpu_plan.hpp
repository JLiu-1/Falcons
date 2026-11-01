// Host-side execution plan for Falcons interpolation decoding.
//
// The plan restates InterpolationDecomposition_OMP's loops as FgNest loop nests. It has to
// reproduce three things exactly: which stream (tid) owns each point, the point's position
// inside that stream, and whether the AVX decoder recovers it through a fused or unfused path.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "fgpu_shared.h"

namespace fgpu {

struct PlanParams {
    int N = 3;
    std::array<size_t, 4> dims{1, 1, 1, 1};
    size_t sizeT = 4;
    int threads = 1;        // decomposition team size
    int interp = 1;         // 0 linear, 1 cubic
    int direction = 0;
    size_t anchorStride = 0;
    double eb = 0, alpha = -1, beta = -1;
    bool fusedAll = true;   // data version >= 1.0.1, or the serial layout: every point is fused
    bool avx = true;        // 1.0.0 -P streams only. false: written by a scalar build, nothing fused, 1D stays on tid 0
    bool serial = false;    // serial layout: one stream
};

// Leaves elements uninitialized so a large work table costs no serial first touch.
template <class V>
struct NoInit : std::allocator<V> {
    template <class U> struct rebind { using other = NoInit<U>; };
    template <class U> void construct(U *) {}
};

struct Plan {
    std::vector<FgNest> nests;
    std::vector<FgPhase> phases;
    std::vector<FgWork, NoInit<FgWork>> work;
    std::vector<uint64_t> streamLen;   // codes per tid, as the loops consume them
    std::vector<uint64_t> anchorLen;   // anchors per tid
    size_t num = 0;
};

struct Lvl {
    uint64_t cnt;
    int64_t dstr;
    uint64_t pstr;
};

class PlanBuilder {
   public:
    explicit PlanBuilder(const PlanParams &p) : P(p) {
        if (P.N < 1 || P.N > 4) throw std::invalid_argument("unsupported dimension");
        off[P.N - 1] = 1;
        for (int i = P.N - 2; i >= 0; --i) off[i] = off[i + 1] * P.dims[i + 1];
        step = 32 / P.sizeT;
        cur.assign(P.threads, 0);
    }

    Plan build() {
        out.num = 1;
        for (int i = 0; i < P.N; ++i) out.num *= P.dims[i];
        out.anchorLen.assign(P.threads, 0);

        int level = -1;
        size_t maxDim = 1;
        bool useAnchor = false;
        for (int i = 0; i < P.N; ++i) {
            level = std::max(level, static_cast<int>(std::ceil(std::log2(P.dims[i]))));
            if (P.dims[i] > P.anchorStride) useAnchor = true;
            maxDim = std::max(maxDim, P.dims[i]);
        }
        size_t anchor = useAnchor ? P.anchorStride : 0;
        if (anchor > 0) {
            int maxLevel = static_cast<int>(std::log2(anchor)) + 1;
            if (maxLevel <= level) level = maxLevel;
        }
        int startLevel = level;
        begin_phases(2 * P.eb);
        if (anchor == 0) {
            int64_t o[4] = {0, 0, 0, 0};
            emit(0, 0, FG_ZERO, 0, o, 1, 0, {}, cur[0]++);
        } else {
            emit_anchors(anchor);
            startLevel--;
        }

        double curEb = P.eb;
        for (int lv = startLevel; lv > 0; --lv) {
            if (P.alpha < 0) {
                curEb = lv >= 3 ? P.eb * 0.5 : P.eb;
            } else if (P.alpha >= 1) {
                double ratio = std::pow(P.alpha, lv - 1);
                if (ratio > P.beta) ratio = P.beta;
                curEb = P.eb / ratio;
            }
            skip = lv == 1;
            ebx2 = 2 * curEb;
            size_t s = size_t(1) << (lv - 1);
            if (P.N == 1) {
                begin_phases(ebx2);
                sweep_1d(s);
            } else if (P.N == 2) {
                sweep_2d(s);
            } else if (P.N == 3) {
                sweep_3d(s);
            } else {
                sweep_generic(s);
            }
        }
        out.streamLen = cur;
        finish();
        return std::move(out);
    }

   private:
    static void chunk(uint64_t n, uint64_t T, uint64_t tid, uint64_t &start, uint64_t &cnt) {
        uint64_t q = n / T, r = n % T;
        if (tid < r) {
            cnt = q + 1;
            start = tid * cnt;
        } else {
            cnt = q;
            start = q * tid + r;
        }
    }

    static uint64_t iters(uint64_t b, uint64_t e, uint64_t st) { return e > b ? (e - b - 1) / st + 1 : 0; }

    double type_eb(double e) const {
        if (P.sizeT == 8) return e;
        float f = static_cast<float>(e);
        if (static_cast<double>(f) > e) f = std::nextafter(f, 0.0f);
        return f;
    }

    void begin_phases(double e) {
        phase0 = out.phases.size();
        FgPhase ph{};
        ph.ebx2 = e;
        // The quantizer's constructor derives the reciprocal differently from set_eb, and the root
        // point is quantized before the first set_eb.
        ph.ebx2r = out.phases.empty() ? 0.5 / P.eb : 1.0 / e;
        ph.ebT = type_eb(e * 0.5);
        out.phases.push_back(ph);
        out.phases.push_back(ph);
        ebx2 = e;
        phNests.resize(out.phases.size());
    }

    // Levels are listed outermost first and padded on the outside with single iterations.
    void emit(int tid, int phase, FgOp op, int fused, const int64_t mult[4], int64_t s, uint64_t data0,
              std::vector<Lvl> lv, uint64_t rel) {
        uint64_t pts = 1;
        for (auto &l : lv) pts *= l.cnt;
        if (pts == 0) return;
        if (lv.size() > 4) throw std::logic_error("nest too deep");
        // A nest above 2^32 points is cut along its outermost non-trivial level.
        if (pts > 0xF0000000ull) {
            size_t i = 0;
            while (lv[i].cnt == 1) ++i;
            uint64_t half = lv[i].cnt / 2;
            auto a = lv, b = lv;
            a[i].cnt = half;
            b[i].cnt = lv[i].cnt - half;
            emit(tid, phase, op, fused, mult, s, data0, a, rel);
            emit(tid, phase, op, fused, mult, s, data0 + half * lv[i].dstr, b, rel + half * lv[i].pstr);
            return;
        }
        FgNest n{};
        n.data0 = data0;
        n.pos0 = rel;
        for (int i = 0; i < 4; ++i) {
            n.cnt[i] = 1;
            n.opnd[i] = mult[i] * s;
        }
        int shift = 4 - static_cast<int>(lv.size());
        for (size_t i = 0; i < lv.size(); ++i) {
            n.cnt[i + shift] = static_cast<uint32_t>(lv[i].cnt);
            n.dstr[i + shift] = lv[i].dstr;
            n.pstr[i + shift] = lv[i].pstr;
        }
        n.points = static_cast<uint32_t>(pts);
        n.tid = tid;
        n.op = op;
        n.fused = (P.fusedAll || (P.avx && fused)) ? 1 : 0;
        phNests.resize(out.phases.size());
        phNests[phase0 + phase].push_back(n);
    }

    void emit_anchors(size_t A) {
        std::vector<Lvl> lv;
        std::array<uint64_t, 4> c{};
        uint64_t inner = 1;
        for (int i = P.N - 1; i >= 0; --i) {
            c[i] = iters(0, P.dims[i], A);
            if (i > 0) inner *= c[i];
        }
        for (int t = 0; t < P.threads; ++t) {
            uint64_t st, cn;
            chunk(c[0], P.threads, t, st, cn);
            if (!cn) continue;
            lv.clear();
            lv.push_back({cn, int64_t(A * off[0]), inner});
            uint64_t stride = inner;
            for (int i = 1; i < P.N; ++i) {
                stride /= c[i];
                lv.push_back({c[i], int64_t(A * off[i]), stride});
            }
            int64_t o[4] = {0, 0, 0, 0};
            emit(t, 0, FG_ANCHOR, 0, o, 1, st * A * off[0], lv, out.anchorLen[t]);
            out.anchorLen[t] += cn * inner;
        }
    }

    // ---- one row-per-target line along direction d: interp_{linear,cubic}_and_quantize_1D[_line] ----
    struct Part {
        uint64_t first, stepIdx, cnt;  // target index, spacing, count
        FgOp op;
        int fused;
        std::array<int64_t, 4> m;
    };

    std::vector<Part> line_parts(uint64_t n) const {
        std::vector<Part> v;
        uint64_t odd = n / 2, even = n - odd;
        auto edge = [&](uint64_t idx) { return int64_t(idx << 1); };
        if (P.interp == 0) {
            uint64_t K = odd > step ? (odd - 1) / step : 0, V = K * step;
            v.push_back({1, 2, V, FG_LINEAR, 1, {-1, 1, 0, 0}});
            if (odd > V + 1) v.push_back({2 * V + 1, 2, odd - 1 - V, FG_LINEAR, 0, {-1, 1, 0, 0}});
            int64_t last = 2 * odd - 1;
            if (n < 3)
                v.push_back({uint64_t(last), 2, 1, FG_EQUAL, 0, {edge(even - 1) - last, 0, 0, 0}});
            else
                v.push_back({uint64_t(last), 2, 1, FG_LINEAR1, 0, {edge(even - 2) - last, edge(even - 1) - last, 0, 0}});
        } else {
            if (even < 2)
                v.push_back({1, 2, 1, FG_EQUAL, 0, {-1, 0, 0, 0}});
            else if (even < 3)
                v.push_back({1, 2, 1, FG_LINEAR, 0, {-1, 1, 0, 0}});
            else
                v.push_back({1, 2, 1, FG_QUAD1, 0, {-1, 1, 3, 0}});
            uint64_t K = even > step + 2 ? (even - 3) / step : 0, V = K * step;
            v.push_back({3, 2, V, FG_CUBIC, 1, {-3, -1, 1, 3}});
            if (even > V + 3) v.push_back({2 * V + 3, 2, even - 3 - V, FG_CUBIC, 0, {-3, -1, 1, 3}});
            if (odd > 1) {
                if (odd < even) {
                    int64_t last = 2 * odd - 1;
                    v.push_back({uint64_t(last), 2, 1, FG_QUAD2, 0,
                                 {edge(even - 3) - last, edge(even - 2) - last, edge(even - 1) - last, 0}});
                } else {
                    if (odd > 2) {
                        int64_t last = 2 * odd - 3;
                        v.push_back({uint64_t(last), 2, 1, FG_QUAD2, 0,
                                     {edge(even - 3) - last, edge(even - 2) - last, edge(even - 1) - last, 0}});
                    }
                    int64_t last = 2 * odd - 1;
                    v.push_back({uint64_t(last), 2, 1, FG_LINEAR1, 0, {edge(even - 2) - last, edge(even - 1) - last, 0, 0}});
                }
            }
        }
        return v;
    }

    // Each iteration of `outer` runs one whole line of n / 2 codes.
    void emit_lines(int t, const std::vector<Part> &parts, int64_t s, uint64_t data0, std::vector<Lvl> outer) {
        uint64_t rel = 0, lines = 1;
        for (auto &l : outer) lines *= l.cnt;
        for (auto &p : parts) {
            auto lv = outer;
            lv.push_back({p.cnt, int64_t(p.stepIdx) * s, 1});
            emit(t, 0, p.op, p.fused, p.m.data(), s, data0 + p.first * s, lv, cur[t] + rel);
            rel += p.cnt;
        }
        cur[t] += rel * lines;
    }

    // ---- rows of target planes handled by the buffered vector kernels ----
    struct Row {
        uint64_t target, cnt;
        FgOp op;
        int fused, phase;
        std::array<int64_t, 4> m;
    };

    std::vector<Row> row_list(uint64_t n, bool lastPass) const {
        std::vector<Row> r;
        int64_t lls = lastPass && skip ? 3 : 2;
        if (P.interp == 0) {
            r.push_back({1, n > 1 ? (n - 1) >> 1 : 0, FG_LINEAR, 1, 0, {-1, 1, 0, 0}});
            if (n % 2 == 0) {
                if (n < 3)
                    r.push_back({n - 1, 1, FG_EQUAL, 1, 0, {-1, 0, 0, 0}});
                else
                    r.push_back({n - 1, 1, FG_LINEAR1, 1, lls == 2 ? 1 : 0, {-lls, -1, 0, 0}});
            }
        } else {
            if (n >= 5)
                r.push_back({1, 1, FG_QUAD1, 1, 0, {-1, 1, 3, 0}});
            else if (n >= 3)
                r.push_back({1, 1, FG_LINEAR, 1, 0, {-1, 1, 0, 0}});
            else
                r.push_back({1, 1, FG_EQUAL, 1, 0, {-1, 0, 0, 0}});
            uint64_t inner = n >= 7 ? (n - 5) >> 1 : 0;
            r.push_back({3, inner, FG_CUBIC, 1, 0, {-3, -1, 1, 3}});
            if (n % 2 == 1) {
                if (n > 3) r.push_back({n - 2, 1, FG_QUAD2, 1, 0, {-3, -1, 1, 0}});
            } else if (n > 4) {
                r.push_back({n - 3, 1, FG_QUAD2, 1, 0, {-3, -1, 1, 0}});
            }
        }
        return r;
    }

    // interpolation_1d_simd_3d_z / _3d_y: `o` is the omp-parallel outer dimension.
    void row_sweep_3d(int o, int d, int r, uint64_t stO, uint64_t stR, size_t s, bool lastPass) {
        uint64_t n = (P.dims[d] - 1) / s + 1;
        if (n <= 1) return;
        int64_t D = s * off[d];
        uint64_t Co = iters(0, P.dims[o], stO), Cr = iters(0, P.dims[r], stR);
        auto rows = row_list(n, lastPass);
        uint64_t perOuter = 0;
        for (auto &x : rows) perOuter += x.cnt;
        bool plane = P.interp == 1 && n % 2 == 0 && n > 2;
        for (int t = 0; t < P.threads; ++t) {
            uint64_t st, cn;
            chunk(Co, P.threads, t, st, cn);
            if (!cn) continue;
            uint64_t rel = 0;
            for (auto &x : rows) {
                emit(t, x.phase, x.op, x.fused, x.m.data(), D, st * stO * off[o] + x.target * D,
                     {{cn, int64_t(stO * off[o]), perOuter * Cr}, {x.cnt, 2 * D, Cr}, {Cr, int64_t(stR * off[r]), 1}},
                     cur[t] + rel);
                rel += x.cnt * Cr;
            }
            cur[t] += cn * perOuter * Cr;
        }
        if (plane) {
            const int64_t m[4] = {-3, -1, 0, 0};
            for (int t = 0; t < P.threads; ++t) {
                uint64_t st, cn;
                chunk(Co, P.threads, t, st, cn);
                if (!cn) continue;
                emit(t, 0, FG_LINEAR1, 0, m, D, st * stO * off[o] + (n - 1) * D,
                     {{cn, int64_t(stO * off[o]), Cr}, {Cr, int64_t(stR * off[r]), 1}}, cur[t]);
                cur[t] += cn * Cr;
            }
        }
    }

    // interpolation_1d_simd_3d_x: collapse(2) over (i, j), one whole line per iteration.
    void line_sweep_3d(int d, uint64_t st0, uint64_t st1, size_t s) {
        const int a = 0, b = 1;
        uint64_t n = (P.dims[d] - 1) / s + 1;
        if (n <= 1) return;
        int64_t D = s * off[d];
        uint64_t C0 = iters(0, P.dims[a], st0), C1 = iters(0, P.dims[b], st1);
        auto parts = line_parts(n);
        int64_t d0 = st0 * off[a], d1 = st1 * off[b];
        uint64_t lineLen = n / 2;
        for (int t = 0; t < P.threads; ++t) {
            uint64_t m0, cn;
            chunk(C0 * C1, P.threads, t, m0, cn);
            uint64_t m1 = m0 + cn;
            while (m0 < m1) {
                uint64_t i = m0 / C1, j = m0 % C1;
                if (j == 0 && m1 - m0 >= C1) {
                    uint64_t rows = (m1 - m0) / C1;
                    emit_lines(t, parts, D, i * d0, {{rows, d0, C1 * lineLen}, {C1, d1, lineLen}});
                    m0 += rows * C1;
                } else {
                    uint64_t take = std::min(C1 - j, m1 - m0);
                    emit_lines(t, parts, D, i * d0 + j * d1, {{take, d1, lineLen}});
                    m0 += take;
                }
            }
        }
    }

    void sweep_3d(size_t s) {
        if (P.direction == 0) {
            begin_phases(ebx2);
            row_sweep_3d(1, 0, 2, 2 * s, 2 * s, s, false);
            begin_phases(ebx2);
            row_sweep_3d(0, 1, 2, s, 2 * s, s, false);
            begin_phases(ebx2);
            line_sweep_3d(2, s, s, s);
        } else if (P.direction == 5) {
            begin_phases(ebx2);
            line_sweep_3d(2, 2 * s, 2 * s, s);
            begin_phases(ebx2);
            row_sweep_3d(0, 1, 2, 2 * s, s, s, false);
            begin_phases(ebx2);
            row_sweep_3d(1, 0, 2, s, s, s, true);
        } else {
            sweep_generic(s);
        }
    }

    // interpolation_1d_simd_2d_y: rows along dim 1, boundaries on tid 0.
    void row_sweep_2d(uint64_t stR, size_t s, bool lastPass) {
        uint64_t n = (P.dims[0] - 1) / s + 1;
        if (n <= 1) return;
        int64_t D = s * off[0];
        uint64_t Cr = iters(0, P.dims[1], stR);
        auto rows = row_list(n, lastPass);
        auto run_tid0 = [&](const Row &x) {
            emit(0, x.phase, x.op, x.fused, x.m.data(), D, x.target * D,
                 {{x.cnt, 2 * D, Cr}, {Cr, int64_t(stR * off[1]), 1}}, cur[0]);
            cur[0] += x.cnt * Cr;
        };
        const Row *interior = nullptr;
        std::vector<const Row *> tail;
        size_t idx = 0;
        if (P.interp == 1) run_tid0(rows[idx++]);
        interior = &rows[idx++];
        for (; idx < rows.size(); ++idx) tail.push_back(&rows[idx]);
        for (int t = 0; t < P.threads; ++t) {
            uint64_t st, cn;
            chunk(interior->cnt, P.threads, t, st, cn);
            if (!cn) continue;
            emit(t, interior->phase, interior->op, interior->fused, interior->m.data(), D,
                 (interior->target + 2 * st) * D, {{cn, 2 * D, Cr}, {Cr, int64_t(stR * off[1]), 1}}, cur[t]);
            cur[t] += cn * Cr;
        }
        for (auto x : tail) run_tid0(*x);
        if (P.interp == 1 && n % 2 == 0 && n > 2) {
            Row x{n - 1, 1, FG_LINEAR1, 1, 0, {-3, -1, 0, 0}};
            run_tid0(x);
        }
    }

    // interpolation_1d_simd_2d_x: parallel for over dim 0, one line along dim 1 per iteration.
    void line_sweep_2d(uint64_t st0, size_t s) {
        uint64_t n = (P.dims[1] - 1) / s + 1;
        if (n <= 1) return;
        int64_t D = s * off[1];
        uint64_t C0 = iters(0, P.dims[0], st0);
        auto parts = line_parts(n);
        uint64_t lineLen = n / 2;
        for (int t = 0; t < P.threads; ++t) {
            uint64_t st, cn;
            chunk(C0, P.threads, t, st, cn);
            if (!cn) continue;
            emit_lines(t, parts, D, st * st0 * off[0], {{cn, int64_t(st0 * off[0]), lineLen}});
        }
    }

    void sweep_2d(size_t s) {
        if (P.direction == 0) {
            begin_phases(ebx2);
            row_sweep_2d(2 * s, s, false);
            begin_phases(ebx2);
            line_sweep_2d(s, s);
        } else {
            begin_phases(ebx2);
            line_sweep_2d(2 * s, s);
            begin_phases(ebx2);
            row_sweep_2d(s, s, true);
        }
    }

    // interpolation_1d_simd with EnableInnerOmp: lanes spread over threads, the rest stays on tid 0.
    void sweep_1d(size_t s) {
        uint64_t n = (P.dims[0] - 1) / s + 1;
        if (n <= 1) return;
        int64_t D = s;
        auto parts = line_parts(n);
        for (auto &p : parts) {
            if (p.fused && !P.serial && (P.fusedAll || P.avx)) {
                uint64_t K = p.cnt / step;
                for (int t = 0; t < P.threads; ++t) {
                    uint64_t st, cn;
                    chunk(K, P.threads, t, st, cn);
                    if (!cn) continue;
                    uint64_t c = cn * step;
                    emit(t, 0, p.op, 1, p.m.data(), D, (p.first + 2 * st * step) * D, {{c, 2 * D, 1}}, cur[t]);
                    cur[t] += c;
                }
            } else {
                emit(0, 0, p.op, p.fused, p.m.data(), D, p.first * D, {{p.cnt, 2 * D, 1}}, cur[0]);
                cur[0] += p.cnt;
            }
        }
    }

    // foreach_omp2: parallel over dim 0, everything goes through the scalar lambda.
    void foreach_n(std::array<uint64_t, 4> b, std::array<uint64_t, 4> e, std::array<uint64_t, 4> st,
                   std::array<int64_t, 4> dimOff, int phase, FgOp op, const int64_t m[4], int64_t s) {
        const int N = P.N;
        std::array<uint64_t, 4> c{1, 1, 1, 1};
        uint64_t inner = 1;
        for (int i = 0; i < N; ++i) {
            c[i] = iters(b[i], e[i], st[i]);
            if (i > 0) inner *= c[i];
        }
        for (int t = 0; t < P.threads; ++t) {
            uint64_t s0, cn;
            chunk(c[0], P.threads, t, s0, cn);
            if (!cn || !inner) continue;
            uint64_t d0 = (b[0] + s0 * st[0]) * dimOff[0];
            for (int i = 1; i < N; ++i) d0 += b[i] * dimOff[i];
            std::vector<Lvl> lv{{cn, int64_t(st[0] * dimOff[0]), inner}};
            uint64_t stride = inner;
            for (int i = 1; i < N; ++i) {
                stride /= c[i];
                lv.push_back({c[i], int64_t(st[i] * dimOff[i]), stride});
            }
            emit(t, phase, op, 0, m, s, d0, lv, cur[t]);
            cur[t] += cn * inner;
        }
    }

    void fastest_dim(int d, std::array<uint64_t, 4> st, size_t s) {
        uint64_t n = (P.dims[d] - 1) / s + 1;
        if (n <= 1) return;
        int64_t D = s * off[d];
        std::array<uint64_t, 4> b{0, 0, 0, 0}, e{1, 1, 1, 1};
        std::array<int64_t, 4> dimOff{1, 1, 1, 1};
        for (int i = 0; i < P.N; ++i) {
            e[i] = P.dims[i];
            dimOff[i] = off[i];
        }
        dimOff[d] = D;
        st[d] = 2;
        if (P.interp == 0) {
            const int64_t lin[4] = {-1, 1, 0, 0};
            b[d] = 1;
            e[d] = n - 1;
            foreach_n(b, e, st, dimOff, 0, FG_LINEAR, lin, D);
            if (n % 2 == 0) {
                b[d] = n - 1;
                e[d] = n;
                const int64_t eq[4] = {-1, 0, 0, 0}, l1[4] = {-2, -1, 0, 0};
                if (n < 3)
                    foreach_n(b, e, st, dimOff, 0, FG_EQUAL, eq, D);
                else
                    foreach_n(b, e, st, dimOff, 1, FG_LINEAR1, l1, D);
            }
        } else {
            const int64_t cub[4] = {-3, -1, 1, 3};
            b[d] = 3;
            e[d] = n >= 3 ? n - 3 : 0;
            foreach_n(b, e, st, dimOff, 0, FG_CUBIC, cub, D);
            std::vector<uint64_t> bnd{1};
            if (n % 2 == 1 && n > 3) bnd.push_back(n - 2);
            if (n % 2 == 0 && n > 4) bnd.push_back(n - 3);
            if (n % 2 == 0 && n > 2) bnd.push_back(n - 1);
            for (auto x : bnd) {
                b[d] = x;
                e[d] = x + 1;
                const int64_t q1[4] = {-1, 1, 3, 0}, q2[4] = {-3, -1, 1, 0}, l[4] = {-1, 1, 0, 0}, l1[4] = {-3, -1, 0, 0},
                              eq[4] = {-1, 0, 0, 0};
                if (x >= 3) {
                    if (x + 3 < n)
                        foreach_n(b, e, st, dimOff, 0, FG_CUBIC, cub, D);
                    else if (x + 1 < n)
                        foreach_n(b, e, st, dimOff, 0, FG_QUAD2, q2, D);
                    else
                        foreach_n(b, e, st, dimOff, 0, FG_LINEAR1, l1, D);
                } else {
                    if (x + 3 < n)
                        foreach_n(b, e, st, dimOff, 0, FG_QUAD1, q1, D);
                    else if (x + 1 < n)
                        foreach_n(b, e, st, dimOff, 0, FG_LINEAR, l, D);
                    else
                        foreach_n(b, e, st, dimOff, 0, FG_EQUAL, eq, D);
                }
            }
        }
    }

    // interpolation_1d_fastest_dim_first over every direction of dim_sequences[direction].
    void sweep_generic(size_t s) {
        std::array<int, 4> seq{0, 1, 2, 3};
        for (int i = 0; i < P.direction; ++i) std::next_permutation(seq.begin(), seq.begin() + P.N);
        std::array<uint64_t, 4> st{1, 1, 1, 1};
        st[seq[0]] = 1;
        for (int i = 1; i < P.N; ++i) st[seq[i]] = 2 * s;
        begin_phases(ebx2);
        fastest_dim(seq[0], st, s);
        for (int i = 1; i < P.N; ++i) {
            st[seq[i - 1]] = s;
            begin_phases(ebx2);
            fastest_dim(seq[i], st, s);
        }
    }

    void finish() {
        phNests.resize(out.phases.size());
        std::vector<uint64_t> base(cur.size() + 1, 0);
        for (size_t i = 0; i < cur.size(); ++i) base[i + 1] = base[i] + cur[i];
        std::vector<uint64_t> abase(cur.size() + 1, 0);
        for (size_t i = 0; i < cur.size(); ++i) abase[i + 1] = abase[i] + out.anchorLen[i];

        std::vector<FgPhase> phases;
        std::vector<uint64_t> nestWork;
        uint64_t workTotal = 0;
        for (size_t i = 0; i < out.phases.size(); ++i) {
            if (phNests[i].empty()) continue;
            FgPhase ph = out.phases[i];
            ph.nest0 = static_cast<uint32_t>(out.nests.size());
            ph.work0 = workTotal;
            for (auto &n : phNests[i]) {
                n.pos0 += n.op == FG_ANCHOR ? abase[n.tid] : base[n.tid];
                nestWork.push_back(workTotal);
                workTotal += (n.points + FG_BLOCK - 1) / FG_BLOCK;
                ph.points += n.points;
                out.nests.push_back(n);
            }
            ph.nestN = static_cast<uint32_t>(out.nests.size()) - ph.nest0;
            ph.workN = workTotal - ph.work0;
            phases.push_back(ph);
        }
        out.phases = std::move(phases);
        out.work.resize(workTotal);
        const int64_t numNests = static_cast<int64_t>(out.nests.size());
#pragma omp parallel for schedule(dynamic, 16)
        for (int64_t id = 0; id < numNests; ++id) {
            const uint64_t blocks = (out.nests[id].points + FG_BLOCK - 1) / FG_BLOCK;
            FgWork *w = out.work.data() + nestWork[id];
            for (uint64_t b = 0; b < blocks; ++b) w[b] = {static_cast<uint32_t>(id), static_cast<uint32_t>(b * FG_BLOCK)};
        }
    }

    PlanParams P;
    Plan out;
    std::array<int64_t, 4> off{};
    std::vector<uint64_t> cur;
    std::vector<std::vector<FgNest>> phNests;
    size_t phase0 = 0;
    uint64_t step = 8;
    double ebx2 = 0;
    bool skip = false;
};

inline Plan build_plan(const PlanParams &p) { return PlanBuilder(p).build(); }

}  // namespace fgpu
