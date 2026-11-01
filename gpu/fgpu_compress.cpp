// Command line GPU compressor for Falcons. Its output decodes with every Falcons decoder.
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include "fgpu_recon.h"
#include "fgpu_stream.hpp"

static void usage() {
    fprintf(stderr,
            "usage: fgpu_compress -i raw -z output (-f | -d) -M ABS|REL <bound> -n d0 [d1 ...] [options]\n"
            "  -n d0 [d1 d2 d3]  dimensions, slowest first (must be the last argument group)\n"
            "  -m auto|pure|hybrid\n"
            "                    pure: everything on the GPU, zstd layer stored raw\n"
            "                    hybrid: CPU encoders, bytes equal the CPU compressor's\n"
            "                    auto (default): pure, or CPU encoders when a stream needs order-1 tANS\n"
            "  -t streams        -P streams (default 32; 1 writes the serial layout)\n"
            "  --chains K        tANS chains per segment, 4 (default, as the CPU) to 16 (faster GPU decoding)\n"
            "  --linear          linear instead of cubic interpolation\n");
}

template <class T>
int run(const std::string &in, const std::string &out, const std::string &mode, int streams, int chains, Falcons::Config conf) {
    std::vector<T> data(conf.num);
    std::ifstream f(in, std::ios::binary);
    if (!f.read(reinterpret_cast<char *>(data.data()), conf.num * sizeof(T))) throw std::runtime_error("cannot read " + in);
    auto t0 = std::chrono::steady_clock::now();
    bool cpuEncoder = false;
    auto c = mode == "pure"     ? fgpu::compress_pure_stream<T>(conf, data.data(), streams, chains)
             : mode == "hybrid" ? fgpu::compress_hybrid<T>(conf, data.data(), streams)
                                : fgpu::compress_auto<T>(conf, data.data(), streams, &cpuEncoder);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (c.empty()) {
        // Too many outliers for the GPU path, so fall back to the CPU library.
        std::vector<char> b(Falcons::Falcons_compress_size_bound<T>(conf));
        c.resize(Falcons_compress(conf, data.data(), b.data(), b.size()));
        std::memcpy(c.data(), b.data(), c.size());
        puts("used the CPU compressor: data is nearly incompressible");
    }
    std::ofstream(out, std::ios::binary).write(c.data(), c.size());
    printf("%zu -> %zu bytes, ratio %.2f, %.2f ms\n", conf.num * sizeof(T), c.size(), double(conf.num * sizeof(T)) / c.size(), ms);
    return 0;
}

int main(int argc, char **argv) {
    std::string in, out, mode = "auto", ebMode;
    double eb = 0;
    int type = 0, streams = 32, chains = 4;
    bool linear = false;
    std::vector<size_t> dims;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-i" && i + 1 < argc) in = argv[++i];
        else if (a == "-z" && i + 1 < argc) out = argv[++i];
        else if (a == "-m" && i + 1 < argc) mode = argv[++i];
        else if (a == "-t" && i + 1 < argc) streams = atoi(argv[++i]);
        else if (a == "--chains" && i + 1 < argc) chains = atoi(argv[++i]);
        else if (a == "--linear") linear = true;
        else if (a == "-f") type = 4;
        else if (a == "-d") type = 8;
        else if (a == "-M" && i + 2 < argc) {
            ebMode = argv[++i];
            eb = atof(argv[++i]);
        } else if (a == "-n") {
            while (i + 1 < argc && argv[i + 1][0] != '-') dims.push_back(atol(argv[++i]));
        } else {
            usage();
            return 2;
        }
    }
    if (in.empty() || out.empty() || !type || dims.empty() || (ebMode != "ABS" && ebMode != "REL") || chains < 1 || chains > 16) {
        usage();
        return 2;
    }
    Falcons::Config conf;
    conf.setDims(dims.begin(), dims.end());
    conf.cmprAlgo = Falcons::ALGO_INTERP;
    conf.errorBoundMode = ebMode == "ABS" ? Falcons::EB_ABS : Falcons::EB_REL;
    (ebMode == "ABS" ? conf.absErrorBound : conf.relErrorBound) = eb;
    conf.openmp = streams > 1;
    conf.interpAlgo = linear ? Falcons::INTERP_ALGO_LINEAR : Falcons::INTERP_ALGO_CUBIC;
    try {
        return type == 4 ? run<float>(in, out, mode, streams, chains, conf) : run<double>(in, out, mode, streams, chains, conf);
    } catch (const std::exception &e) {
        fprintf(stderr, "fgpu_compress: %s\n", e.what());
        return 1;
    }
}
