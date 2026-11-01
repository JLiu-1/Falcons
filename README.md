# Falcons: Pushing the Speed-Ratio Limits of Lossy Compression for Parallel Scientific Applications

## Changes on this branch (`fix/cpu-omp-bugs`)

### Why

In upstream Falcons, a compressed stream could fail to decompress, or decompress to wrong data without an error, when the decompressing setup differed from the compressing one. The affected differences are the OpenMP thread count and the AVX2 versus scalar build. Some configurations also exceeded the error bound or crashed during compression. For example, a `-P` stream compressed with 32 threads decompressed to wrong data with 16 threads and crashed with 64.

This branch makes compression and decompression close the loop across AVX2 and scalar builds at any thread count, with as few code and structural changes as possible.

### What changed

| Problem | Effect before | Fix |
| --- | --- | --- |
| Decompression used the caller's thread count, not the stream's | Wrong data or crash | Thread count now comes from the stream, and the decoder is chosen by stream layout (`Falcons_stream_is_omp`) |
| Per-thread buffer bound `(num / T) * 4` was too small on skewed shapes | Error bound exceeded (0.118 against 0.01), crashes | Bound now accounts for the shortest dimension, and `each_num` is `size_t` |
| Threads greater than `min(dims[1..])` | Compression crash | Encoding uses the decomposition's actual team size, and `omp_set_dynamic` is turned off |
| OMP quantizer mixed fused and unfused arithmetic | AVX2 and scalar results differed | Floating-point types always use `fma` |
| 1D `-P` stream layout depended on the build | AVX2 and scalar could not read each other's streams | Scalar path uses the same chunk layout as AVX2 |
| AVX2 lanes compared errors against `float(eb)` | AVX2 and scalar accepted different values | The float bound is rounded toward zero |
| `SkipOverwrite` ignored in line kernels | Different data stored on the lossless fallback | Loops use `quantize_point<CompMode, SkipOverwrite>` |
| `ALGO_NOPRED` used a dangling pointer and had no `+radius` bias | Unusable, crashes | Buffer is allocated with `new int[]`, and the bias is applied both ways |
| Buffer estimates left out outliers and predictor tables | Heap overflow in Lorenzo/regression paths | Estimates added, and OMP output is checked against `cmpCap` |
| 3D directions 1-4 reached the wrong dimensions | Heap corruption or exceeded bound | They now use `interpolation_1d_fastest_dim_first` |
| Integer `-P` subtraction wrapped around | Error bound exceeded | Integer types use the serial compression path |

[docs/cpu-closed-loop-fixes.md](docs/cpu-closed-loop-fixes.md) has the cause and fix of each problem, in Chinese.

### Compatibility

* The data version is now 1.0.1. Older binaries refuse the new streams.
* Version 1.0.0 streams stay readable, and decode to the same bits as with the old binary.
* Version 1.0.0 1D `-P` streams can still only be read by the kind of build that wrote them (AVX2 or scalar).

### Behavior changes

* If fewer threads are available than the stream has (for example because of `OMP_THREAD_LIMIT`), decompression raises an exception instead of decoding wrong data.
* Integer `-P` compression no longer runs in parallel.

### Verification

`tools/test/test_closed_loop.cpp` runs randomized configurations covering all algorithms, 8 data types, 1-4 dimensions, 1-100 threads, all directions, and smooth, noisy, constant, spiky and NaN/Inf data.

* 1000 configurations compressed by each build and decompressed by both builds at 5 thread counts: 20000 runs, 0 failures.
* AddressSanitizer: 500 compressions and 2500 decompressions, 0 failures.
* AVX2 and scalar compressed output: 1000 of 1000 byte-identical.

### Not fixed

* `Falcons_compress` takes a `const T*` but rewrites the input in place, and the lossless fallback stores the rewritten data. That fallback is therefore within the error bound, not strictly lossless.
* The SVE2 path was not tested or changed.
* A `-P` stream cannot hold more than 2^31 codes per thread.

## Changes on branch `feature/gpu-codec`

This branch adds the GPU codec on top of the CPU fixes in `fix/cpu-omp-bugs`. It only adds the `gpu/` directory, and the CPU library stays the same as on that branch.

### Why

The goal is to decompress Falcons data on the GPU without giving up compatibility: data compressed on the CPU has to decompress on the GPU to the same bits, and data compressed on the GPU has to decompress on the CPU. Bit-exact results matter because the compressor checks its error bound against its own exact reconstruction, so a drift of 1 ULP could propagate through the interpolation levels. Moving streams between builds and thread counts relies on the CPU fixes above.

### What was added

| Written by | Read by | Guarantee |
| --- | --- | --- |
| CPU | GPU | Output byte-identical to the CPU decoder |
| GPU hybrid | CPU, GPU | Stream byte-identical to the CPU compressor's |
| GPU pure | CPU, GPU | Valid stream with the same quantization codes, so the same decoded values |

* `fgpu_decode` and `fgpu_compress` are the command-line tools. `pure` runs everything but the container handling on the GPU, and `hybrid` keeps entropy coding on the CPU.
* `-m auto` chooses per stream. Lorenzo/regression, no-prediction and lossless streams fall back to the CPU decoder.
* GPU compression covers float and double interpolation, without the CPU's automatic tuning.
* `fgpu_difftest` checks every direction against the CPU library byte for byte.

[gpu/README.md](gpu/README.md) has the build steps, usage, design, performance numbers and limits. The codec builds separately with its own Makefile and needs CUDA 12.8 or newer.

## Build & install

```bash
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=[INSTALL_DIR] .. -DENABLE_AVX2=ON
make
make install
```

Executables land in `[INSTALL_DIR]/bin`, headers in `[INSTALL_DIR]/include`.

**SIMD / AVX2**
* `-DENABLE_AVX2=ON` turns on the AVX2 code paths (x86, needs CPU support) — a large speedup.
* On ARM use `-DENABLE_SVE2=ON` instead.
* With neither flag you get a portable scalar build. **Scalar and AVX2 builds produce byte-identical streams**, and any stream decodes in either build at any thread count. SVE2 is untested in this respect; see [docs/cpu-closed-loop-fixes.md](docs/cpu-closed-loop-fixes.md).
* The SIMD flag is attached to the `Falcons` CMake target, so **everything that links `Falcons` inherits it automatically** — the `falcons` binary and the HDF5 filter. You only set it once, at configure time.

**Optional components**
* `-DBUILD_H5Z_FILTER=ON` also builds the HDF5 filter (requires the HDF5 development package).

## How to run

### Falcons executable
`tools/falcons/falcons` — command-line compression / decompression. Run it with no arguments to see usage.

### C++ API
* After installation, CMake consumers can use:

  ```cmake
  find_package(Falcons CONFIG REQUIRED)
  target_link_libraries(my_application PRIVATE Falcons::Falcons)
  ```

  The exported target provides the required FSE library and Zstd dependency.
* Include `Falcons/api/falcons.hpp` with a C++17 compiler. Configuration types live in namespace `Falcons`; the public entry points are `Falcons_compress` and `Falcons_decompress`.
```cpp
#include "Falcons/api/falcons.hpp"

Falcons::Config conf(dim0, dim1, dim2);          // n-D dimensions
conf.errorBoundMode = Falcons::EB_ABS;
conf.absErrorBound  = 1e-3;

size_t outSize;
char  *comp = Falcons_compress(conf, data, outSize);          // T* data -> compressed bytes
T     *dec  = Falcons_decompress<T>(conf, comp, outSize);     // -> reconstructed data

delete[] comp;
delete[] dec;
```
* **AVX2/SVE2:** compile the translation unit that includes the header with `-mavx2 -mfma` (x86) or `-march=armv8.6-a+sve2` (ARM). If you build through this project's CMake, linking the `Falcons` target adds the flag for you.

### Python API (pyfalcons)
* Install from this source tree with `pip install ./tools/pyfalcons`. See [the Python bindings](tools/pyfalcons).
```python
import numpy as np
from pyfalcons import falcons, falconsConfig, falconsErrorBoundMode, falconsAlgorithm

data   = np.random.rand(100, 200, 300).astype(np.float32)
config = falconsConfig()
config.errorBoundMode = falconsErrorBoundMode.ABS
config.absErrorBound  = 1e-3
# optional: config.cmprAlgo = falconsAlgorithm.INTERP_LORENZO

compressed, ratio       = falcons.compress(data, config, copy=True)
decompressed, config    = falcons.decompress(compressed, np.float32, data.shape)
max_err, psnr, nrmse    = falcons.verify(data, decompressed.reshape(data.shape))
```
* `falcons.compress` passes the NumPy buffer directly by default (`copy=False`), so Falcons may overwrite the input array in place. Pass `copy=True` to compress a private C-contiguous copy and leave the input unchanged. `falcons.decompress(compressed, dtype, shape)` recovers the Falcons configuration from the compressed stream, so passing the original `config` is optional.
* The wheel builds Falcons from source (with bundled Zstd). For a source build, `setup.py` **auto-detects the build machine's SIMD** — `-march=native` enables AVX2 on x86 and SVE2 on ARM, and stays scalar otherwise. Set `PYFALCONS_SIMD` to force a target (`avx2`, `sve2`, `native`, `none`, or a custom `-march=…`); see [`tools/pyfalcons/README.md`](tools/pyfalcons/README.md) for details.

### HDF5 filter (H5Z-Falcons)
* Located in `tools/H5Z-Falcons`; library target `hdf5falcons`. Build with `-DBUILD_H5Z_FILTER=ON` (requires HDF5).
* Add `-DENABLE_AVX2=ON` to vectorize the compression that runs *inside* the filter — the filter's CMake sets the SIMD flags explicitly, so it is AVX2/SVE2-accelerated just like the rest of Falcons.
* `falconsToHDF5` and `HDF5ToFalcons` (under `tools/H5Z-Falcons/test`) are provided for testing — e.g. `falconsToHDF5 -f data.dat 100 100 100` writes `data.dat.falcons.h5`, and `HDF5ToFalcons data.dat.falcons.h5` reconstructs it. The type flag is `-f` (float), `-d` (double), or `-i8/-u8/-i16/-u16/-i32/-u32/-i64/-u64` for integers.
* To use it as a dynamically-loaded HDF5 filter, point `HDF5_PLUGIN_PATH` at the built `libhdf5falcons`, then request filter id `40024` on your dataset-creation property list.
