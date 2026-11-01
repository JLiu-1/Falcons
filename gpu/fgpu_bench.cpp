// End-to-end benchmark on a raw file: CPU decode vs the GPU decoders, bytes compared.
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>

#include "fgpu_recon.h"
#include "fgpu_stream.hpp"

using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

template <class T>
int run(const char *path, std::vector<size_t> dims, double rel, int threads, int reps) {
    size_t num = 1;
    for (auto d : dims) num *= d;
    std::vector<T> data(num);
    std::ifstream(path, std::ios::binary).read(reinterpret_cast<char *>(data.data()), num * sizeof(T));
    Falcons::Config conf;
    conf.setDims(dims.begin(), dims.end());
    conf.cmprAlgo = Falcons::ALGO_INTERP;
    conf.errorBoundMode = Falcons::EB_REL;
    conf.relErrorBound = rel;
    conf.openmp = true;
    omp_set_num_threads(threads);
    std::vector<char> cbuf(Falcons::Falcons_compress_size_bound<T>(conf));
    auto copy = data;
    auto c0 = Clock::now();
    size_t csize = Falcons_compress(conf, copy.data(), cbuf.data(), cbuf.size());
    double cpuComp = ms(c0, Clock::now());
    for (int r = 0; r < 2; ++r) {
        copy = data;
        c0 = Clock::now();
        csize = Falcons_compress(conf, copy.data(), cbuf.data(), cbuf.size());
        cpuComp = std::min(cpuComp, ms(c0, Clock::now()));
    }
    printf("cpu compress: %.1f ms\n", cpuComp);
    for (int r = 0; r < reps; ++r) {
        fgpu::CompressTimes ct;
        auto c1 = Clock::now();
        auto g = fgpu::compress_hybrid<T>(conf, data.data(), threads, &ct);
        printf("hybrid compress: %.1f ms (upload %.1f quant %.1f gather %.1f) identical %s\n", ms(c1, Clock::now()), ct.upload, ct.quant, ct.gather,
               g.size() == csize && !std::memcmp(g.data(), cbuf.data(), csize) ? "yes" : "NO");
    }
    printf("compressed %zu -> %zu bytes, ratio %.2f, %d threads\n", num * sizeof(T), csize, double(num * sizeof(T)) / csize, threads);

    for (int chains : {4, 16})
        for (int r = 0; r < reps; ++r) {
            fgpu::PureCompressOutput ct;
            auto c1 = Clock::now();
            auto g = fgpu::compress_pure_stream<T>(conf, data.data(), threads, chains, &ct);
            printf("pure compress K=%d: %.1f ms (upload %.1f quant %.1f rle %.1f fse %.1f tans %.1f assemble %.1f) size %zu\n", chains, ms(c1, Clock::now()), ct.upload, ct.quant, ct.rle,
                   ct.fse, ct.tans, ct.assemble, g.size());
        }
    {   // the array already in device memory, ending with the stream in host memory
        void *dev = fgpu::upload(data.data(), num * sizeof(T));
        fgpu::CompressOutput scratch;
        for (int r = 0; r < reps; ++r) {
            auto c1 = Clock::now();
            fgpu::download(copy.data(), dev, num * sizeof(T));
            auto c2 = Clock::now();
            Falcons_compress(conf, copy.data(), cbuf.data(), cbuf.size());
            printf("device to cpu compress: %.1f ms (download %.1f compress %.1f)\n", ms(c1, Clock::now()), ms(c1, c2), ms(c2, Clock::now()));
            fgpu::CompressTimes ct;
            c1 = Clock::now();
            auto g = fgpu::compress_hybrid<T>(conf, nullptr, threads, &ct, dev, &scratch);
            printf("device hybrid compress: %.1f ms (quant %.1f gather %.1f) identical %s\n", ms(c1, Clock::now()), ct.quant, ct.gather,
                   g.size() == csize && !std::memcmp(g.data(), cbuf.data(), csize) ? "yes" : "NO");
        }
        for (int r = 0; r < reps; ++r) {
            fgpu::PureCompressOutput ct;
            auto c1 = Clock::now();
            auto g = fgpu::compress_pure_stream<T>(conf, nullptr, threads, 4, &ct, dev);
            printf("device pure compress: %.1f ms (quant %.1f rle %.1f fse %.1f tans %.1f assemble %.1f) size %zu\n", ms(c1, Clock::now()), ct.quant, ct.rle, ct.fse, ct.tans,
                   ct.assemble, g.size());
        }
        fgpu::free_device(dev);
    }
    if (getenv("FGPU_TUNED")) {   // streams written by the GPU for the GPU
        for (auto [K, chains] : {std::pair<int, int>{32, 4}, {32, 16}, {128, 4}, {128, 16}}) {
            auto g = fgpu::compress_pure_stream<T>(conf, data.data(), K, chains);
            std::vector<T> o(num);
            Falcons::Config dc;
            Falcons_read_config(dc, g.data(), g.size());
            fgpu::StreamConfig sc{dc.N, false};
            fgpu::PureInput pi;
            pi.conf = &sc;
            pi.sizeT = sizeof(T);
            size_t h = static_cast<unsigned char>(g[0]);
            pi.cmp = reinterpret_cast<unsigned char *>(g.data()) + h;
            pi.cmpSize = g.size() - h;
            double best = 1e18;
            for (int r = 0; r < 3; ++r) {
                fgpu::PureTiming t;
                auto c2 = Clock::now();
                fgpu::decode_pure(pi, o.data(), &t);
                best = std::min(best, ms(c2, Clock::now()) - t.download);
            }
            printf("gpu->gpu streams=%d chains=%d: %zu bytes, ratio %.2f, decode to vram %.1f ms\n", K, chains, g.size(), double(num * sizeof(T)) / g.size(), best);
        }
    }
    std::vector<T> cpu(num), gpu(num);
    Falcons::Config dconf;
    for (int r = 0; r < reps; ++r) {
        T *cp = cpu.data();
        auto t0 = Clock::now();
        Falcons_decompress<T>(dconf, cbuf.data(), csize, cp);
        printf("cpu %d threads: %.1f ms\n", threads, ms(t0, Clock::now()));
    }
    size_t hdr = static_cast<unsigned char>(cbuf[0]);
    const auto *c = reinterpret_cast<unsigned char *>(cbuf.data()) + hdr;
    for (int r = 0; r < reps; ++r) {
        auto t0 = Clock::now();
        auto S = fgpu::decode_streams<T>(dconf, c, csize - hdr);
        auto t1 = Clock::now();
        fgpu::ReconInput in;
        in.plan = &S.plan;
        in.sizeT = sizeof(T);
        in.streams.assign(S.codes.begin(), S.codes.end());
        in.anchors = S.anchors.data();
        in.unpred = S.unpred.data();
        in.numUnpred = S.unpred.size();
        in.unpredPerTid = S.unpredPerTid;
        fgpu::Timing t;
        fgpu::reconstruct(in, gpu.data(), &t);
        printf("hybrid: cpu-entropy+plan %.1f upload %.1f escape %.1f recon %.1f download %.1f ms | to-vram %.1f ms, nests %zu work %zu esc %zu\n",
               ms(t0, t1), t.upload, t.escape, t.recon, t.download, ms(t0, t1) + t.upload + t.escape + t.recon, S.plan.nests.size(),
               S.plan.work.size(), S.unpred.size());
    }
    bool same = std::memcmp(cpu.data(), gpu.data(), num * sizeof(T)) == 0;
    printf("hybrid == cpu: %s\n", same ? "yes" : "NO");
    fgpu::StreamConfig sc{dconf.N, dconf.falconsDataVer != versionInt(Falcons_DATA_VER)};
    fgpu::PureInput pi;
    pi.conf = &sc;
    pi.sizeT = sizeof(T);
    pi.cmp = c;
    pi.cmpSize = csize - hdr;
    for (int r = 0; r < reps; ++r) {
        std::memset(gpu.data(), 0, num * sizeof(T));
        fgpu::PureTiming t;
        auto t0 = Clock::now();
        fgpu::decode_pure(pi, gpu.data(), &t);
        double all = ms(t0, Clock::now());
        printf("pure: parse %.2f upload %.2f zstd %.2f fse %.2f rle %.2f escape %.2f recon %.2f download %.2f | to-vram %.2f ms\n", t.parse, t.upload, t.zstd,
               t.fse, t.rle, t.escape, t.recon, t.download, all - t.download);
    }
    bool same2 = std::memcmp(cpu.data(), gpu.data(), num * sizeof(T)) == 0;
    printf("pure == cpu: %s\n", same2 ? "yes" : "NO");
    return same && same2 ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: fgpu_bench f32|f64 file rel threads reps d0 [d1 ...]\n");
        return 2;
    }
    std::vector<size_t> dims;
    for (int i = 6; i < argc; ++i) dims.push_back(atol(argv[i]));
    if (!strcmp(argv[1], "f32")) return run<float>(argv[2], dims, atof(argv[3]), atoi(argv[4]), atoi(argv[5]));
    return run<double>(argv[2], dims, atof(argv[3]), atoi(argv[4]), atoi(argv[5]));
}
