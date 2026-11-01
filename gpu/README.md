# Falcons GPU codec

CUDA compression and decompression that interoperates with the CPU Falcons library in every direction:

| Written by | Read by | Guarantee |
| --- | --- | --- |
| CPU | GPU | output byte-identical to the CPU decoder |
| GPU hybrid | CPU, GPU | the stream itself is byte-identical to the CPU compressor's |
| GPU pure | CPU, GPU | valid stream, same quantization codes, so the same decoded values |

Bit-exact results matter because the compressor checks its error bound against its own exact
reconstruction: a 1 ULP drift could propagate through the interpolation levels.

Requires data version 1.0.1 or 1.0.0 streams and the CPU fixes described in
`docs/cpu-closed-loop-fixes.md`.

## Decoders and compressors

| Mode | CPU does | GPU does |
| --- | --- | --- |
| decode `pure` | walks container metadata | zstd, FSE, RLE, tANS, escapes, reconstruction |
| decode `hybrid` | zstd and entropy decoding | escapes, reconstruction |
| compress `pure` | assembles the container, overflow codes of tANS | prediction, quantization, RLE, FSE, tANS |
| compress `hybrid` | all entropy coding | prediction, quantization |

`-m auto` picks the better one per stream: decode switches to `hybrid` when a stream has compressed zstd
blocks, which is how order-1 tANS streams look, and compress switches to the CPU encoders when a stream
needs order-1 tANS. Lorenzo/regression, no-prediction, lossless and integer streams stay on the CPU;
`fgpu_decode` falls back to the CPU library for the first three.

Pure compression stores the zstd layer as raw blocks. Files are 0.2-0.5% larger at tight bounds, and
streams with too many outliers (more than `max(2^20, codes / 64)`) go to the CPU compressor.

## Build

Requires CUDA 12.8 or newer for native `sm_120`. The older `/usr/bin/nvcc` 12.0 cannot target it.
Configure Falcons first so that `version.hpp` and `libFalcons_fse.a` exist:

    cmake -S .. -B ../build -DENABLE_AVX2=ON && cmake --build ../build --target Falcons_fse
    make FALCONS_BUILD=../build

`--fmad=false` in the Makefile is mandatory. Without it, nvcc contracts the predictor and
dequantization into FMA and the output stops matching the CPU.

## Use

    fgpu_compress -i data.raw -z data.falcons -f -M REL 1e-3 -n 512 512 512
    fgpu_decode   -i data.falcons -f -o data.out
    fgpu_decode   -i data.falcons -d -c reference.raw        # byte-compare

From code, pass `device` to `decode_pure` or `reconstruct` to keep the array in device memory, and
`deviceData` to compress an array already there. Moving data over PCIe is the slow part: 512^3 floats take
about 23 ms each way.

    fgpu::CompressOutput scratch;   // reused between calls
    auto a = fgpu::compress_hybrid<float>(conf, nullptr, 32, nullptr, device, &scratch);
    auto b = fgpu::compress_pure_stream<float>(conf, nullptr, 32, 4, nullptr, device);

Both return the stream in host memory and leave the device array unchanged, using device memory for a working
copy. You choose the path: there is no automatic fallback, and an empty result means the data is nearly
incompressible. Hybrid downloads 2 bytes per point and encodes on the CPU, and `scratch` saves it 30 ms per call
at 512^3 at the cost of keeping that buffer. Relative bounds take the value range from the device array, skipping
NaNs. The CPU's range on data with NaNs depends on where they fall, so hybrid is byte-identical to the CPU
compressor except in that case.

`--scalar-stream` is only for 1.0.0 `-P` streams written by a non-AVX build. Version 1.0.1 streams do not
depend on the build that wrote them. SVE2-built `-P` streams are not supported.

Device blocks are cached between calls, since `cudaMalloc` costs milliseconds at these sizes. Call
`trim_pool()` to return them.

## Performance

JHTDB 512^3 f32, 32 threads, RTX 5090 and Threadripper 7985WX, warm process. CPU times end in host memory,
GPU times end in device memory. Add 23 ms to bring data back to the host. The first call of a process
costs about 200 ms more for context creation.

| REL | CPU decode | hybrid | pure | pure, GPU-tuned stream |
| --- | --- | --- | --- | --- |
| 1e-3 | 28 ms | 22 ms | **7 ms** | |
| 1e-4 | 50 ms | 35 ms | 176 ms | **22 ms** |
| 1e-5 | 37-50 ms | 45 ms | 240 ms | **36 ms** |
| 1e-6 | 58 ms | 50 ms | 360 ms | 73 ms |

GPU-tuned means written by `fgpu_compress -m pure -t 128 --chains 16`: more independent tANS chains, at a
cost of 1.8% in ratio at 1e-4. The stream stays valid for the CPU, which then needs 128 threads' worth of
oversubscription to read it, so prefer it for data that stays on the GPU.

Compression at REL 1e-3: CPU 33 ms, GPU pure 40 ms from host memory, of which 23 ms is the upload. On
4.2 billion points (Basilisk, 16.8 GB): CPU 651 ms, GPU about 105 ms plus upload; decode 715 ms against 145 ms.

Compressing an array already in device memory, ending with the stream in host memory, 32 threads, median of
6 runs in one process on a shared machine. The CPU path downloads the array first. Going from 32 to 64 CPU
threads gave no gain.

| Data, REL | Ratio | download + CPU | `compress_hybrid` with `scratch` | `compress_pure_stream` |
| --- | --- | --- | --- | --- |
| f32, 1e-3 | 75.0 | 60 ms | **21 ms** | 23 ms |
| f32, 1e-4 | 15.8 | 69 ms | **54 ms** | 295 ms |
| f32, 1e-5 | 6.5 | **78 ms** | 83 ms | 403 ms |
| f32, 1e-6 | 3.8 | **94 ms** | 125 ms | 451 ms |
| f64, 1e-3 | 149.7 | 110 ms | **26 ms** | 48 ms |
| f64, 1e-5 | 12.9 | 118 ms | **85 ms** | 399 ms |

## How it works

`fgpu_plan.hpp` restates the loops of `InterpolationDecomposition_OMP` as a few thousand rectangular
loop nests instead of per-point tables. A nest gives each point's data index, its position in the
owning thread's stream, and the predictor. Nests within a phase are independent, and one launch runs a
phase. The only dependency inside a sweep is the linear edge at `n - 1` reading `n - 3`, which gets a
second phase. Thread-bound streams are modeled with OpenMP's static schedule, so the plan needs the
compressor's team size, which equals the stream count.

Compression and decompression share the plan and the predictors. `quant_kernel` mirrors the CPU
quantizer: float difference, double rounding to nearest even, fused reconstruction, float error check.

Zstd is nearly a no-op on these streams, so most blocks are stored raw and go through a parallel byte
copy. Compressed blocks run in one thread per stream, assuming the 128 KB block cut of zstd level 3;
when that check fails, `decode_pure` returns false. FSE works per 64K block with its table in shared
memory, and is a port of `fse_compress.c`, so a block produces the same bytes as the CPU. RLE goes
through prefix sums.

## Limits

Order-1 tANS is the bottleneck: each chain is strictly sequential at about 12M symbols/s, one L2 access per
symbol, and parallelism stops at `streams * segments * chains`. Raising `O1T_K` from 4 to 16 in the CPU
compressor keeps size and CPU decoding unchanged and speeds `pure` up 1.6-3x, but at 32 streams `hybrid` stays
faster, and CPU compression was up to about 20% slower in a run on a loaded machine. `O1T_K` therefore stays at 4;
`fgpu_compress --chains 16` gets the same effect for data that stays on the GPU.

Reconstruction runs at 60-80 Gpoints/s and is limited by memory access: moving the per-block state on chip
and removing the divisions changed nothing. The remaining fixed cost in the 7 ms is the 0.7 ms memset of the
dense code array.

Pure compression needs device memory for the data, the codes and the non-zero entries, about 1.5 times
the data size plus 2 bytes per code at tight bounds; allocation failures raise an exception.

## Tests

`fgpu_difftest` compresses and decompresses with the CPU library and checks every direction byte for byte
across dimensions 1-4, float/double, linear/cubic, all directions, thread counts, anchors and error
schedules, then prints which entropy paths were hit. `fgpu_difftest_scalar` repeats the decoding checks
against a scalar build. Set `FGPU_CHECK=1` to compare zstd and entropy output stage by stage.
