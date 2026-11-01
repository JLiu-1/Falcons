// Command line GPU decoder for Falcons streams.
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include "fgpu_recon.h"
#include "fgpu_stream.hpp"

using Clock = std::chrono::steady_clock;

static void usage() {
    fprintf(stderr,
            "usage: fgpu_decode -i input.falcons (-f | -d) [-o output] [-m auto|pure|hybrid|cpu] [--scalar-stream] [-c reference]\n"
            "  -f, -d            element type: float or double\n"
            "  -m                auto (default): pure unless a stream uses order-1 tANS, then hybrid\n"
            "  --scalar-stream   the -P stream was written by a scalar (non-AVX) Falcons build\n"
            "  -c                compare the output with a raw reference file, byte for byte\n");
}

static std::vector<char> slurp(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<char> v(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(v.data(), v.size());
    return v;
}

template <class T>
int decode(const std::vector<char> &cmp, std::string mode, bool avx, const std::string &out, const std::string &ref) {
    Falcons::Config conf;
    Falcons_read_config(conf, cmp.data(), cmp.size());
    std::vector<T> data(conf.num);
    if (conf.cmprAlgo != Falcons::ALGO_INTERP) mode = "cpu";   // Lorenzo/regression, nopred, lossless

    const size_t hdr = static_cast<unsigned char>(cmp[0]);
    const auto *payload = reinterpret_cast<const uint8_t *>(cmp.data()) + hdr;
    const size_t psize = cmp.size() - hdr;
    fgpu::StreamConfig sc{conf.N, conf.falconsDataVer != versionInt(Falcons_DATA_VER)};
    fgpu::PureInput pi;
    pi.conf = &sc;
    pi.sizeT = sizeof(T);
    pi.cmp = payload;
    pi.cmpSize = psize;
    pi.avx = avx;

    auto t0 = Clock::now();
    if (mode == "auto" || mode == "pure") {
        pi.rejectMethod1 = mode == "auto";
        if (fgpu::decode_pure(pi, data.data(), nullptr)) mode = "pure";
        else mode = "hybrid";
    }
    if (mode == "hybrid") {
        auto S = fgpu::decode_streams<T>(conf, payload, psize, avx);
        fgpu::ReconInput in;
        in.plan = &S.plan;
        in.sizeT = sizeof(T);
        in.streams.assign(S.codes.begin(), S.codes.end());
        in.anchors = S.anchors.data();
        in.unpred = S.unpred.data();
        in.numUnpred = S.unpred.size();
        in.unpredPerTid = S.unpredPerTid;
        fgpu::reconstruct(in, data.data(), nullptr);
    } else if (mode == "cpu") {
        T *p = data.data();
        Falcons::Config c2;
        Falcons_decompress<T>(c2, cmp.data(), cmp.size(), p);
    }
    double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    printf("decoded %zu elements with mode %s in %.2f ms (data downloaded to host)\n", conf.num, mode.c_str(), ms);

    if (!out.empty()) std::ofstream(out, std::ios::binary).write(reinterpret_cast<char *>(data.data()), conf.num * sizeof(T));
    if (!ref.empty()) {
        auto r = slurp(ref);
        if (r.size() != conf.num * sizeof(T) || std::memcmp(r.data(), data.data(), r.size())) {
            printf("reference: DIFFERENT\n");
            return 1;
        }
        printf("reference: identical\n");
    }
    return 0;
}

int main(int argc, char **argv) {
    std::string in, out, ref, mode = "auto";
    int type = 0;
    bool avx = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-i" && i + 1 < argc) in = argv[++i];
        else if (a == "-o" && i + 1 < argc) out = argv[++i];
        else if (a == "-c" && i + 1 < argc) ref = argv[++i];
        else if (a == "-m" && i + 1 < argc) mode = argv[++i];
        else if (a == "-f") type = 4;
        else if (a == "-d") type = 8;
        else if (a == "--scalar-stream") avx = false;
        else {
            usage();
            return 2;
        }
    }
    if (in.empty() || !type || (mode != "auto" && mode != "pure" && mode != "hybrid" && mode != "cpu")) {
        usage();
        return 2;
    }
    try {
        auto cmp = slurp(in);
        return type == 4 ? decode<float>(cmp, mode, avx, out, ref) : decode<double>(cmp, mode, avx, out, ref);
    } catch (const std::exception &e) {
        fprintf(stderr, "fgpu_decode: %s\n", e.what());
        return 1;
    }
}
