// Host interface to the CUDA reconstruction stage.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "fgpu_plan.hpp"

namespace fgpu {

struct Timing {
    double upload = 0, escape = 0, recon = 0, download = 0;
};

struct ReconInput {
    const Plan *plan = nullptr;
    size_t sizeT = 4;
    std::vector<const int16_t *> streams;    // per tid, lengths in plan->streamLen
    const void *anchors = nullptr;           // plan->anchorLen values, concatenated in tid order
    const void *unpred = nullptr;            // outliers left after the anchors, concatenated in tid order
    uint64_t numUnpred = 0;
    std::vector<uint64_t> unpredPerTid;
};

struct CompressInput {
    const Plan *plan = nullptr;
    size_t sizeT = 4;
    const void *deviceData = nullptr;   // source array already in device memory; skips the upload
};

struct CompressOutput {
    bool ok = false;                  // false: too many outliers for the GPU path
    std::vector<int16_t, NoInit<int16_t>> codes;   // all streams, tid order
    std::vector<uint8_t> anchors;     // tid order
    std::vector<uint64_t> escPos;     // sorted stream positions
    std::vector<uint8_t> escVal;      // original values at escPos
    double upload = 0, quant = 0, gather = 0;
};

// Runs prediction and quantization on the GPU. Codes equal the CPU compressor's.
void compress_gpu(const CompressInput &in, const void *data, CompressOutput &out);

struct PureCompressInput {
    const Plan *plan = nullptr;
    size_t sizeT = 4;
    bool serial = false;
    int o1Chains = 4;              // tANS chains per segment: 4 matches the CPU, up to 16 decodes faster
    const void *deviceData = nullptr;   // source array already in device memory; skips the upload
    CompressOutput *codesIfTans = nullptr;   // if set and a stream needs order-1 tANS, stop and hand back the codes
    std::vector<uint8_t> header;   // interpolation header, placed at the front of the last stream
};

struct PureCompressOutput {
    bool ok = false;
    std::vector<std::vector<uint8_t>> blocks;   // one lossless block per stream
    double upload = 0, quant = 0, rle = 0, fse = 0, tans = 0, assemble = 0;
};

// Runs the whole compressor on the GPU. Only container assembly stays on the host.
void compress_pure(const PureCompressInput &in, const void *data, PureCompressOutput &out);

struct InterpHeader {
    size_t dims[4];
    uint32_t blocksize;
    int interpId, direction;
    size_t anchorStride;
    double alpha, beta, eb;
    int radius;
};

// Same test as Falcons_stream_is_omp: the OMP layout has offsets[0] == 0 at bytes 4..11.
inline bool stream_is_serial(const uint8_t *cmp) {
    for (int i = 4; i < 12; ++i)
        if (cmp[i]) return true;
    return false;
}

struct StreamConfig {
    size_t N;
    bool legacy;   // data version 1.0.0
};

struct PureTiming {
    double parse = 0, upload = 0, zstd = 0, fse = 0, rle = 0, escape = 0, recon = 0, download = 0;
    int streams = 0, method1 = 0, delta = 0, zstdBlocks = 0, ovfRaw = 0, ovfCoded = 0, fseRaw = 0, fseRle = 0, fseCoded = 0;
};

struct PureInput {
    const StreamConfig *conf = nullptr;
    size_t sizeT = 4;
    const uint8_t *cmp = nullptr;   // payload after the Falcons config header
    size_t cmpSize = 0;
    bool avx = true;
    bool rejectMethod1 = false;              // return false instead of decoding order-1 tANS streams
    bool stats = false;                      // fill the coverage counters of PureTiming
    std::vector<const int16_t *> refCodes;   // optional: per-stream codes to check against, for debugging
};

// Decodes on the GPU alone. The host only walks container metadata.
// out: host buffer to download into, or null. device: if non-null, receives the decoded array in
// device memory; the caller releases it with free_device. Returns false when the stream needs the
// hybrid decoder: rejectMethod1 was set, or zstd did not cut its frames at 128 KB.
bool decode_pure(const PureInput &in, void *out, PureTiming *t, void **device = nullptr);

void *upload(const void *src, size_t bytes);   // copies to new device memory, released with free_device
void free_device(void *p);
void trim_pool();   // returns cached device blocks to the driver
void download(void *dst, const void *device, size_t bytes);

// max - min of a device array, computed in the element type like the CPU's data_range. NaNs are
// skipped, and the result is 0 when every value is NaN.
double device_range(const void *device, size_t count, size_t sizeT);

// Reconstructs into device memory and, if out is non-null, downloads to out. If device is
// non-null it receives the array in device memory, released with free_device.
void reconstruct(const ReconInput &in, void *out, Timing *t, void **device = nullptr);

}  // namespace fgpu
