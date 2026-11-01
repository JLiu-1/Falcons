// Differential test: CPU-compress, CPU-decompress, GPU-decompress, compare bytes.
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>

#include "fgpu_recon.h"
#include "fgpu_stream.hpp"

#ifdef __AVX2__
#define FGPU_AVX true
#else
#define FGPU_AVX false   // streams from a scalar build: no fused lanes
#endif

template <class T>
std::vector<T> field(const std::vector<size_t> &dims, unsigned seed, double outlier, double noise) {
    size_t num = 1;
    for (auto d : dims) num *= d;
    std::vector<T> v(num);
    std::mt19937 g(seed);
    std::normal_distribution<double> nd(0, 1);
    std::uniform_real_distribution<double> ud(0, 1);
    double f[4];
    for (auto &x : f) x = 2 + 9 * ud(g);
    for (size_t i = 0; i < num; ++i) {
        size_t r = i;
        double s = 0;
        for (int k = int(dims.size()) - 1; k >= 0; --k) {
            size_t c = r % dims[k];
            r /= dims[k];
            s += std::sin(f[k] * double(c) / double(dims[k]) * 6.28);
        }
        s += noise * nd(g);
        if (ud(g) < outlier) s += 500 * nd(g);
        v[i] = static_cast<T>(s);
    }
    return v;
}

struct Case {
    std::vector<size_t> dims;
    int threads, interp, dir, anchor;
    double eb;
    double alpha;
};

static fgpu::PureTiming cover;
static int failures = 0, passed = 0, skipped = 0, compressed = 0, pureCompressed = 0;

template <class T>
void one(const Case &c, unsigned seed) {
    char desc[200];
    int n = snprintf(desc, sizeof desc, "%s", sizeof(T) == 4 ? "f32" : "f64");
    for (auto d : c.dims) n += snprintf(desc + n, sizeof desc - n, " %zu", d);
    snprintf(desc + n, sizeof desc - n, " t=%d interp=%d dir=%d anchor=%d eb=%g a=%g seed=%u", c.threads, c.interp, c.dir,
             c.anchor, c.eb, c.alpha, seed);

    if (getenv("FGPU_VERBOSE")) puts(desc);
    auto data = field<T>(c.dims, seed, seed % 4 < 2 ? 1e-3 : 0, seed % 4 == 3 ? 0.0 : 0.02);
    Falcons::Config conf;
    conf.setDims(c.dims.begin(), c.dims.end());
    conf.cmprAlgo = Falcons::ALGO_INTERP;
    conf.errorBoundMode = Falcons::EB_ABS;
    conf.absErrorBound = c.eb;
    conf.openmp = c.threads > 1;
    conf.interpAlgo = c.interp;
    conf.interpDirection = c.dir;
    if (c.alpha >= 0) {
        conf.interpAlpha = c.alpha;
        conf.interpBeta = c.alpha > 1 ? 3 : 1;
    }
    if (c.anchor >= 0) conf.interpAnchorStride = c.anchor;
    omp_set_num_threads(c.threads);
    size_t csize = 0;
    std::vector<char> cbuf(Falcons::Falcons_compress_size_bound<T>(conf));
    auto copy = data;
    csize = Falcons_compress(conf, copy.data(), cbuf.data(), cbuf.size());

    Falcons::Config dconf;
    std::vector<T> cpu(data.size());
    T *cp = cpu.data();
    Falcons_decompress<T>(dconf, cbuf.data(), csize, cp);
    if (dconf.cmprAlgo != Falcons::ALGO_INTERP) {
        skipped++;
        return;
    }
    if (FGPU_AVX) {   // a GPU-compressed stream must be the CPU's stream, byte for byte
        auto g = fgpu::compress_hybrid<T>(conf, data.data(), c.threads);
        if (g.empty()) {
            skipped++;
        } else if (g.size() != csize || std::memcmp(g.data(), cbuf.data(), csize) != 0) {
            size_t i = 0;
            while (i < std::min(g.size(), csize) && g[i] == cbuf[i]) ++i;
            printf("FAIL compress %s: %zu vs %zu bytes, first difference at %zu\n", desc, g.size(), csize, i);
            failures++;
        } else {
            compressed++;
        }
        for (int chains : {4, 16}) {
            auto pg = fgpu::compress_pure_stream<T>(conf, data.data(), c.threads, chains);
            if (pg.empty()) {
                skipped++;
                continue;
            }
            std::vector<T> viaCpu(data.size());
            T *vp = viaCpu.data();
            Falcons::Config pc;
            Falcons_decompress<T>(pc, pg.data(), pg.size(), vp);
            // GPU -> GPU closes the fourth direction of the interoperability square.
            std::vector<T> viaGpu(data.size());
            {
                size_t ph = static_cast<unsigned char>(pg[0]);
                fgpu::StreamConfig psc{pc.N, false};
                fgpu::PureInput ppi;
                ppi.conf = &psc;
                ppi.sizeT = sizeof(T);
                ppi.cmp = reinterpret_cast<unsigned char *>(pg.data()) + ph;
                ppi.cmpSize = pg.size() - ph;
                fgpu::decode_pure(ppi, viaGpu.data(), nullptr);
            }
            if (std::memcmp(viaGpu.data(), cpu.data(), cpu.size() * sizeof(T)) != 0) {
                printf("FAIL gpu->gpu K=%d %s\n", chains, desc);
                failures++;
            }
            if (std::memcmp(viaCpu.data(), cpu.data(), cpu.size() * sizeof(T)) != 0) {
                size_t diff = 0;
                for (size_t i = 0; i < cpu.size(); ++i) diff += std::memcmp(&viaCpu[i], &cpu[i], sizeof(T)) != 0;
                printf("FAIL pure-compress K=%d %s: %zu/%zu differ\n", chains, desc, diff, cpu.size());
                failures++;
            } else {
                pureCompressed++;
            }
        }
    }
    try {
        size_t hdr = static_cast<unsigned char>(cbuf[0]);
        auto S = fgpu::decode_streams<T>(dconf, reinterpret_cast<unsigned char *>(cbuf.data()) + hdr, csize - hdr, FGPU_AVX);
        std::vector<T> gpu(data.size());
        fgpu::ReconInput in;
        in.plan = &S.plan;
        in.sizeT = sizeof(T);
        in.streams.assign(S.codes.begin(), S.codes.end());
        in.anchors = S.anchors.data();
        in.unpred = S.unpred.data();
        in.numUnpred = S.unpred.size();
        in.unpredPerTid = S.unpredPerTid;
        fgpu::reconstruct(in, gpu.data(), nullptr);
        std::vector<T> pure(data.size());
        fgpu::StreamConfig sc{dconf.N, dconf.falconsDataVer != versionInt(Falcons_DATA_VER)};
        fgpu::PureInput pi;
        pi.conf = &sc;
        pi.sizeT = sizeof(T);
        pi.cmp = reinterpret_cast<unsigned char *>(cbuf.data()) + hdr;
        pi.cmpSize = csize - hdr;
        pi.avx = FGPU_AVX;
        pi.stats = true;
        if (getenv("FGPU_CHECK")) pi.refCodes.assign(S.codes.begin(), S.codes.end());
        fgpu::PureTiming pt;
        void *dev = nullptr;
        if (!fgpu::decode_pure(pi, nullptr, &pt, &dev)) {
            printf("FAIL pure %s: declined the stream\n", desc);
            failures++;
            return;
        }
        fgpu::download(pure.data(), dev, pure.size() * sizeof(T));
        fgpu::free_device(dev);
        cover.method1 += pt.method1;
        cover.delta += pt.delta;
        cover.zstdBlocks += pt.zstdBlocks;
        cover.ovfRaw += pt.ovfRaw;
        cover.ovfCoded += pt.ovfCoded;
        cover.fseRaw += pt.fseRaw;
        cover.fseRle += pt.fseRle;
        cover.fseCoded += pt.fseCoded;
        if (std::memcmp(pure.data(), cpu.data(), cpu.size() * sizeof(T)) != 0) {
            size_t diff = 0;
            for (size_t i = 0; i < cpu.size(); ++i) diff += std::memcmp(&pure[i], &cpu[i], sizeof(T)) != 0;
            printf("FAIL pure %s: %zu/%zu differ\n", desc, diff, cpu.size());
            failures++;
        }
        if (std::memcmp(gpu.data(), cpu.data(), cpu.size() * sizeof(T)) != 0) {
            size_t diff = 0, first = cpu.size();
            for (size_t i = 0; i < cpu.size(); ++i)
                if (std::memcmp(&gpu[i], &cpu[i], sizeof(T))) {
                    if (!diff) first = i;
                    diff++;
                }
            printf("FAIL %s: %zu/%zu differ, first %zu (cpu %.17g gpu %.17g)\n", desc, diff, cpu.size(), first,
                   double(cpu[first]), double(gpu[first]));
            failures++;
        } else {
            passed++;
        }
    } catch (const std::exception &e) {
        printf("ERR  %s: %s\n", desc, e.what());
        failures++;
    }
}

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    int maxCases = argc > 1 ? atoi(argv[1]) : 1000000;
    std::vector<Case> cases;
    std::vector<std::vector<size_t>> shapes3 = {{65, 64, 63}, {32, 33, 34}, {7, 40, 9}, {40, 7, 30}, {4, 5, 66}, {6, 2, 8},
                                                {2, 3, 2},    {3, 70, 4},   {128, 40, 33}, {64, 64, 64}, {9, 100, 6}};
    std::vector<std::vector<size_t>> shapes2 = {{200, 300}, {64, 65}, {6, 40}, {40, 6}, {2, 80}, {80, 2}, {513, 129}, {3, 3}};
    std::vector<std::vector<size_t>> shapes1 = {{5000}, {4097}, {256}, {63}, {64}, {7}, {2}, {100000}};
    std::vector<std::vector<size_t>> shapes4 = {{9, 8, 7, 6}, {17, 16, 5, 4}, {3, 20, 20, 3}, {2, 33, 9, 8}};
    std::vector<int> threads = {1, 2, 3, 7, 8, 32};
    for (auto &s : shapes3)
        for (int t : threads)
            for (int interp : {0, 1})
                for (int dir : {0, 5, 1, 3})
                    cases.push_back({s, t, interp, dir, -1, 1e-2, -1});
    for (auto &s : shapes2)
        for (int t : threads)
            for (int interp : {0, 1})
                for (int dir : {0, 1})
                    cases.push_back({s, t, interp, dir, -1, 1e-2, -1});
    for (auto &s : shapes1)
        for (int t : threads)
            for (int interp : {0, 1}) cases.push_back({s, t, interp, 0, -1, 1e-2, -1});
    for (auto &s : shapes4)
        for (int t : {1, 2, 3, 8})
            for (int interp : {0, 1})
                for (int dir : {0, 7, 23}) cases.push_back({s, t, interp, dir, -1, 1e-2, -1});
    // anchor stride and error-bound schedule
    for (int t : {1, 8})
        for (int anchor : {0, 4, 8, 16, 64})
            for (double a : {1.0, 1.25, 2.0, -1.0})
                for (double eb : {1e-1, 1e-3}) cases.push_back({{70, 60, 50}, t, 1, 0, anchor, eb, a});
    for (int t : {1, 4, 16})
        for (double eb : {1e-3, 1e-4, 1e-7}) {
            cases.push_back({{300, 200, 90}, t, 1, 0, -1, eb, -1});
            cases.push_back({{1500, 1000}, t, 1, 0, -1, eb, -1});
            cases.push_back({{1500000}, t, 1, 0, -1, eb, -1});
            cases.push_back({{40, 40, 40, 40}, t, 1, 0, -1, eb, -1});
        }
    std::mt19937 g(1);
    std::shuffle(cases.begin(), cases.end(), g);
    if (int(cases.size()) > maxCases) cases.resize(maxCases);
    unsigned seed = 1;
    for (auto &c : cases) {
        one<float>(c, seed++);
        one<double>(c, seed++);
    }
    printf("coverage: method1 %d delta %d zstd-blocks %d ovf raw/coded %d/%d fse raw/rle/coded %d/%d/%d\n", cover.method1, cover.delta,
           cover.zstdBlocks, cover.ovfRaw, cover.ovfCoded, cover.fseRaw, cover.fseRle, cover.fseCoded);
    printf("passed %d failed %d skipped %d, gpu compress identical %d, pure-gpu compress decoded by cpu %d\n", passed, failures, skipped, compressed, pureCompressed);
    return failures != 0;
}
