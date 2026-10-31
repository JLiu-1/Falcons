# Falcons: Pushing the Speed-Ratio Limits of Lossy Compression for Parallel Scientific Applications

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
