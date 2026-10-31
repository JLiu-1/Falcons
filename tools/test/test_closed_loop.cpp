// Closed-loop test: every stream must decode correctly in any build at any thread count.
//
//   test_closed_loop gen <dir> [cases]    compress a seeded case list, keeping stream and self-decode
//   test_closed_loop check <dir>          decode every stream in <dir> at several thread counts
//
// Run gen with one build and check with another to test interoperability. Each case runs in a forked
// child, so a crash is reported as a failure of that case. Checks: the error bound, NaN/Inf
// preservation, and byte equality with the compressing build's own decode.
#include <omp.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "Falcons/api/falcons.hpp"

struct Case {
    int id, type, algo, threads, interp, dir, anchor, kind, mode;
    double alpha, eb;
    std::vector<size_t> dims;
};

static const char *TYPE_NAME[] = {"f32", "f64", "i16", "i32", "i64", "u16", "u32", "u64"};
static const char *KIND_NAME[] = {"smooth", "noisy", "constant", "white", "spiky", "nonfinite"};

static std::vector<Case> make_cases(int count) {
    std::mt19937 g(20261101);
    auto pick = [&](std::initializer_list<int> v) { return *(v.begin() + g() % v.size()); };
    const std::vector<std::vector<size_t>> shapes = {
        {100000}, {4097}, {256}, {64}, {63}, {7}, {3}, {2}, {200000}, {200, 300}, {64, 65}, {6, 40}, {40, 6}, {2, 80}, {80, 2},
        {513, 129}, {3, 3}, {65, 64, 63}, {32, 33, 34}, {7, 40, 9}, {40, 7, 30}, {4, 5, 66}, {6, 2, 8}, {2, 3, 2}, {3, 70, 4},
        {128, 40, 33}, {64, 8, 64}, {100, 90, 2}, {9, 8, 7, 6}, {17, 16, 5, 4}, {3, 20, 20, 3}, {2, 33, 9, 8}, {20, 2, 20, 2}};
    std::vector<Case> v;
    for (int i = 0; i < count; ++i) {
        Case c{};
        c.id = i;
        c.type = pick({0, 0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 6, 7});
        c.algo = pick({Falcons::ALGO_INTERP, Falcons::ALGO_INTERP, Falcons::ALGO_INTERP, Falcons::ALGO_INTERP_LORENZO,
                       Falcons::ALGO_LORENZO_REG, Falcons::ALGO_NOPRED, Falcons::ALGO_LOSSLESS});
        c.threads = pick({1, 2, 3, 7, 8, 16, 32, 64, 100});
        c.interp = g() % 2;
        c.dims = shapes[g() % shapes.size()];
        size_t N = c.dims.size();
        c.dir = N == 3 ? pick({0, 0, 5, 5, 1, 2, 3, 4}) : N == 2 ? pick({0, 1}) : N == 4 ? static_cast<int>(g() % 24) : 0;
        c.anchor = pick({-1, -1, -1, 0, 4, 8, 16, 64});
        c.kind = c.type < 2 ? g() % 6 : g() % 5;
        c.mode = pick({Falcons::EB_ABS, Falcons::EB_REL});
        c.alpha = pick({-1, -1, 1, 2}) * (g() % 2 ? 1.0 : 1.25);
        c.eb = c.type < 2 ? std::pow(10.0, -static_cast<int>(1 + g() % 7)) : pick({1, 2, 8, 100});
        if (c.type >= 2) c.mode = Falcons::EB_ABS;
        v.push_back(c);
    }
    return v;
}

static std::string describe(const Case &c) {
    char b[256];
    int n = snprintf(b, sizeof b, "#%d %s algo=%d T=%d interp=%d dir=%d anchor=%d %s mode=%d eb=%g a=%g dims", c.id, TYPE_NAME[c.type], c.algo,
                     c.threads, c.interp, c.dir, c.anchor, KIND_NAME[c.kind], c.mode, c.eb, c.alpha);
    for (auto d : c.dims) n += snprintf(b + n, sizeof b - n, " %zu", d);
    return b;
}

template <class T>
std::vector<T> make_data(const Case &c) {
    size_t num = 1;
    for (auto d : c.dims) num *= d;
    std::vector<T> v(num);
    // Drawn by hand: under -mfma the compiler fuses a*b+c here and inside <random>, which would give
    // AVX and scalar builds different input data.
    std::mt19937 g(c.id + 1);
    auto ud = [&](std::mt19937 &r) { return (r() >> 5) / 134217728.0; };
    auto nd = [&](std::mt19937 &r) {
        volatile double t = 0;
        for (int i = 0; i < 12; ++i) t = t + ud(r);
        return t - 6;
    };
    const double scale = std::is_floating_point_v<T> ? 1 : 1000;
    for (size_t i = 0; i < num; ++i) {
        size_t r = i;
        double s = 0;
        for (int k = int(c.dims.size()) - 1; k >= 0; --k) {
            size_t x = r % c.dims[k];
            r /= c.dims[k];
            volatile double t = std::sin(3.0 * (k + 1) * double(x) / double(c.dims[k]) * 6.28);
            s = s + t;
        }
        switch (c.kind) {
            case 0: break;
            case 1: {
                volatile double t = 0.02 * nd(g);
                s = s + t;
                break;
            }
            case 2: s = 1.5; break;
            case 3: s = 100 * ud(g); break;
            default: {
                volatile double t = ud(g) < 1e-3 ? 500 * nd(g) : 0;
                s = s + t;
                break;
            }
        }
        double w = s * scale;
        if constexpr (!std::is_signed_v<T>) w = std::fabs(w);
        v[i] = static_cast<T>(w);
        if constexpr (std::is_floating_point_v<T>) {
            if (c.kind == 5) {
                double u = ud(g);
                if (u < 1e-3) v[i] = std::numeric_limits<T>::quiet_NaN();
                else if (u < 2e-3) v[i] = std::numeric_limits<T>::infinity();
            }
        }
    }
    return v;
}

static void write_file(const std::string &p, const void *d, size_t n) { std::ofstream(p, std::ios::binary).write(static_cast<const char *>(d), n); }
static std::vector<char> read_file(const std::string &p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return {};
    std::vector<char> v(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(v.data(), v.size());
    return v;
}

// Returns the number of points that break the bound or lost a non-finite value.
template <class T>
size_t count_errors(const std::vector<T> &a, const T *b, double eb) {
    size_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            if (!std::isfinite(a[i])) {
                bad += std::memcmp(&a[i], &b[i], sizeof(T)) != 0;
                continue;
            }
        }
        bad += std::fabs(static_cast<long double>(a[i]) - static_cast<long double>(b[i])) > eb;
    }
    return bad;
}

template <class T>
int gen_case(const Case &c, const std::string &dir) {
    auto data = make_data<T>(c);
    Falcons::Config conf;
    conf.setDims(c.dims.begin(), c.dims.end());
    conf.cmprAlgo = c.algo;
    conf.errorBoundMode = c.mode;
    if (c.mode == Falcons::EB_ABS) conf.absErrorBound = c.eb;
    else conf.relErrorBound = c.eb;
    conf.openmp = true;
    conf.interpAlgo = c.interp;
    conf.interpDirection = c.dir;
    if (c.anchor >= 0) conf.interpAnchorStride = c.anchor;
    if (c.alpha > 0) {
        conf.interpAlpha = c.alpha;
        conf.interpBeta = c.alpha > 1 ? 3 : 1;
    }
    omp_set_num_threads(c.threads);
    std::vector<char> cmp(Falcons::Falcons_compress_size_bound<T>(conf));
    auto copy = data;
    size_t size = Falcons_compress(conf, copy.data(), cmp.data(), cmp.size());
    Falcons::Config d;
    std::vector<T> out(data.size());
    T *p = out.data();
    Falcons_decompress<T>(d, cmp.data(), size, p);
    double eb = c.eb;
    if (c.mode == Falcons::EB_REL) {
        double lo = 0, hi = 0;
        bool first = true;
        for (auto x : data)
            if (std::isfinite(static_cast<double>(x))) {
                lo = first ? double(x) : std::min(lo, double(x));
                hi = first ? double(x) : std::max(hi, double(x));
                first = false;
            }
        eb = c.eb * (hi - lo);
    }
    size_t bad = count_errors(data, p, eb * (1 + 1e-9));
    const std::string base = dir + "/" + std::to_string(c.id);
    write_file(base + ".cmp", cmp.data(), size);
    write_file(base + ".ref", out.data(), out.size() * sizeof(T));
    if (bad) {
        printf("BOUND %s: %zu points\n", describe(c).c_str(), bad);
        return 1;
    }
    return 0;
}

template <class T>
int check_case(const Case &c, const std::string &dir, int threads) {
    const std::string base = dir + "/" + std::to_string(c.id);
    auto cmp = read_file(base + ".cmp");
    auto ref = read_file(base + ".ref");
    if (cmp.empty()) return 0;
    omp_set_num_threads(threads);
    Falcons::Config d;
    T *p = nullptr;
    Falcons_decompress<T>(d, cmp.data(), cmp.size(), p);
    int rc = std::memcmp(p, ref.data(), ref.size()) != 0;
    delete[] p;
    if (rc) printf("DIFF  %s: decoded with %d threads\n", describe(c).c_str(), threads);
    return rc;
}

template <template <class> class F, class... A>
int by_type(int type, A &&...a) {
    switch (type) {
        case 0: return F<float>()(a...);
        case 1: return F<double>()(a...);
        case 2: return F<int16_t>()(a...);
        case 3: return F<int32_t>()(a...);
        case 4: return F<int64_t>()(a...);
        case 5: return F<uint16_t>()(a...);
        case 6: return F<uint32_t>()(a...);
        default: return F<uint64_t>()(a...);
    }
}
template <class T> struct Gen { int operator()(const Case &c, const std::string &d) { return gen_case<T>(c, d); } };
template <class T> struct Check { int operator()(const Case &c, const std::string &d, int t) { return check_case<T>(c, d, t); } };

template <class Fn>
int isolated(const Case &c, const char *what, int threads, Fn fn) {
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        alarm(120);
        try {
            _exit(fn());
        } catch (const std::exception &e) {
            printf("THROW %s %s T=%d: %s\n", describe(c).c_str(), what, threads, e.what());
            fflush(stdout);
            _exit(1);
        }
    }
    int st;
    waitpid(pid, &st, 0);
    if (WIFSIGNALED(st)) {
        printf("CRASH %s %s T=%d: signal %d\n", describe(c).c_str(), what, threads, WTERMSIG(st));
        return 1;
    }
    return WEXITSTATUS(st) != 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: test_closed_loop gen|check <dir> [cases]\n");
        return 2;
    }
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const std::string mode = argv[1], dir = argv[2];
    int count = argc > 3 ? atoi(argv[3]) : 2000;
    if (mode == "list") {
        for (auto &c : make_cases(count)) puts(describe(c).c_str());
        return 0;
    }
    if (mode == "gen") mkdir(dir.c_str(), 0755);
    else count = 1000000;
    auto cases = make_cases(mode == "gen" ? count : 0);
    int failed = 0, runs = 0;
    if (mode == "gen") {
        const char *only = getenv("CASE");
        for (auto &c : cases) {
            if (only && atoi(only) != c.id) continue;
            ++runs;
            if (only && getenv("NOFORK")) {
                by_type<Gen>(c.type, c, dir);
                continue;
            }
            failed += isolated(c, "compress", c.threads, [&] { return by_type<Gen>(c.type, c, dir); });
        }
    } else {
        std::vector<char> list = read_file(dir + "/count");
        count = list.empty() ? 2000 : atoi(std::string(list.begin(), list.end()).c_str());
        cases = make_cases(count);
        for (auto &c : cases) {
            if (read_file(dir + "/" + std::to_string(c.id) + ".cmp").empty()) continue;
            for (int t : {c.threads, 1, 2, 5, 64}) {
                ++runs;
                failed += isolated(c, "decompress", t, [&] { return by_type<Check>(c.type, c, dir, t); });
            }
        }
    }
    if (mode == "gen") write_file(dir + "/count", std::to_string(count).data(), std::to_string(count).size());
    printf("%s: %d runs, %d failed\n", mode.c_str(), runs, failed);
    return failed != 0;
}
