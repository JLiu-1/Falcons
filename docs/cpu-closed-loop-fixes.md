# CPU 压缩/解压闭环修复说明

本次修复的目标：Falcons 压缩的流，在 AVX 和标量构建之间、在任意 OpenMP 线程数下，都要能解对。

结论：修复后，1000 个随机配置由两种构建各压缩一遍，再由两种构建各用 5 种线程数解压，共 20000 次解压，
0 失败；AddressSanitizer 下 500 次压缩、2500 次解压，0 失败；AVX 和标量构建的压缩产物 1000/1000 逐字节相同。

## 兼容性

- 数据版本从 1.0.0 升到 1.0.1。旧程序读新流时会报版本不符。
- 新程序仍能读 1.0.0 的流，结果和旧程序逐位相同。已用修复前的二进制生成 f32/f64、1D/3D、AVX/标量的旧流，
  在 1、2、8、64 线程下验证。
- 1.0.0 的 1D `-P` 流仍然只能由和写入端相同的构建读取（AVX 写的 AVX 读，标量写的标量读）。旧版本同样如此，流里也没有记录写入端的构建类型。1.0.1 的流没有这个限制。

## 问题和修法

严重程度：**静默错误**指不报错但结果错，**崩溃**指段错误或堆损坏，**不一致**指结果都在误差界内但构建之间不同。

### 1. 解压线程数和压缩不同时解错或崩溃（静默错误 / 崩溃）

现象：32 线程压缩的 `-P` 流，16 线程解压得到全错的数据且不报错，64 或 2 线程段错误，1 线程抛异常。

原因：每条量化码流绑定一个线程，点到线程的映射来自 OpenMP 静态调度，所以解压必须用和压缩相同的线程数。
而解压用的是调用方当前的线程数。另外，选择串行或 OMP 解码器时看的也是调用方线程数，不看流本身的布局。

修法：
- `FalconsGenericCompressor_OMP::decompress` 按流里的流数设置线程数，结束后恢复。
- 新增 `Falcons_stream_is_omp`，按流布局选择解码器：OMP 布局的第 4–11 字节是 `offsets[0] == 0`，串行布局在第 8 字节开始是 zstd 帧。
- 线程数上不去（比如 `OMP_THREAD_LIMIT` 小于流数）时抛出明确的异常，不再解错。

### 2. 每线程缓冲区溢出，静默超出误差界（静默错误）

现象：`2×33×9×8`、8 线程压缩时最大误差 0.118，而误差界是 0.01。还会随机段错误。

原因：`each_num = (num / T) << 2` 假设每个线程最多拿到平均值的 4 倍。短的外层维度和只在 tid 0 上执行的边界面都能超过它，
溢出后覆盖相邻线程的码流。`each_num` 还是 `int`，大数据低线程数时会溢出。

修法：上界改为 `min(num, (num / T) * 4 + 8 * (num / 最短维度) + 256)`，类型改为 `size_t`。
用执行计划对 40 万种随机形状和线程数做了穷举检查，0 次越界。短维度的数据虚拟内存占用会变大，但只有用到的页才会占物理内存。

### 3. 线程数大于 `min(dims[1..])` 时压缩崩溃（崩溃）

原因：分解器把线程数限制到 `min(T, dims[1..])`，但编码阶段仍按 `omp_get_max_threads()` 写流，访问 `local_quant_inds[tid]` 越界。

修法：编码阶段用分解器实际的线程数。分解器用 `actual_team_size()` 测量真实的 team 大小，不再假定请求一定被满足；
同时关闭 `omp_set_dynamic`，保证所有并行区域的 team 相同。

### 4. OMP 量化器融合和非融合混用（不一致）

现象：AVX 压缩的 `-P` f64 流，标量构建解压后 55% 的点不同，最大差 2.9e-15。

原因：`LinearQuantizerOMP` 的标量路径是 `pred + q * eb`，AVX lane 是 `fmadd`，串行量化器用 `std::fma`。

修法：新增 `LinearQuantizerOMP::dequantize`，浮点类型一律用 `fma`。读 1.0.0 流时走旧路径（`set_legacy_unfused`），这是升数据版本的原因。

### 5. 1D `-P` 流 AVX 和标量不能互解（崩溃 / 静默错误）

原因：AVX 构建把向量块分给各线程，标量构建全部放在线程 0，流布局不同。

修法：标量路径的 `interp_{linear,cubic}_and_quantize_1D_line` 按相同的块大小和调度分配。读 1.0.0 流时保持旧布局。

### 6. AVX lane 用 `float(eb)` 比较误差（不一致）

原因：`float(eb)` 可能向上舍入，AVX 路径会接受略超误差界的值（相对约 1e-7），标量路径用 double 比较，两者判定不同。

修法：把 float 版误差界向零舍入。对 float 类型的误差值，比较结果和 double 比较严格等价。串行和 OMP 两处都改了。

### 7. SkipOverwrite 在线路 kernel 里没生效（不一致）

原因：标量路径和 AVX 尾部用 `quantize_func`，总是覆盖数据；AVX lane 在 SkipOverwrite 时不覆盖。平时没有影响，
但压缩比低于 3 时回退 LOSSLESS，存的是被就地改写过的数据，内容就不同了。

修法：这些循环统一用 `quantize_point<CompMode, SkipOverwrite>`。

### 8. ALGO_NOPRED 完全不可用（崩溃）

原因有两个：`compress()` 返回局部 `vector` 的 `data()`，是悬空指针；量化码改成有符号后没有加 `+radius` 偏置，Huffman 表下标为负。

修法：改为 `new int[]`，由通用压缩器释放；压缩加偏置，解压减偏置，和 Lorenzo 路径一致。

### 9. LORENZO_REG 等路径缓冲区溢出（崩溃）

原因：`FalconsGenericCompressor` 的缓冲区估计不含 Blockwise 的离群值和预测器表，白噪声小数据下堆溢出。
OMP 版的每线程缓冲区也没有计入离群值，写入总量也不检查 `cmpCap`。

修法：`BlockwiseDecomposition` 补上 `size_est()`；按实际离群值数量估算缓冲区；OMP 版写入前检查 `cmpCap`，
不够时抛 `length_error`，走已有的 LOSSLESS 降级路径。

### 10. 3D 方向 1–4 无法使用（崩溃 / 静默错误）

原因：SIMD sweep 只支持方向 0（xyz）和 5（zyx），其他方向走到 else 分支后访问了错误的维度，出现堆损坏或超出误差界。

修法：其他方向改走本来就支持任意方向的 `interpolation_1d_fastest_dim_first`。

### 11. 整数类型的 `-P` 压缩超出误差界（静默错误）

现象：i16/u16 白噪声数据少数点超界。

原因：`LinearQuantizerOMP` 没有串行量化器的溢出安全整数运算，减法会回绕。

修法：整数类型压缩改走串行路径，只改一行条件。代价是整数 `-P` 压缩不再并行。旧的整数 `-P` 流仍可读。

## 没有修的

- `Falcons_compress` 的参数是 `const T*`，但压缩会就地改写数据，LOSSLESS 回退存的也是改写后的数据，
  所以回退结果是“误差界内”而不是严格无损。改它要多一份完整拷贝，这里没有动。
- 解压时可用线程数少于流数（`OMP_THREAD_LIMIT`，或在禁用嵌套的并行区域里调用）会抛异常，而不是自动降级。
  调度和线程绑定写在各个循环里，要支持只能重写结构。
- SVE2 路径没有编译环境，未测试，也没有改。向量长度不是 256 位时，1D `-P` 流的布局和 AVX 不同，
  float 误差界的舍入也还是旧行为。
- 单线程码数超过 2^31 的 `-P` 流无法写入，因为格式里的计数是 `int`。

## 验证方法

`tools/test/test_closed_loop.cpp` 用种子生成随机配置，覆盖全部算法、8 种数据类型、1–4 维、1–100 线程、全部方向、
anchor、误差调度，以及平滑、噪声、常量、白噪声、尖峰、NaN/Inf 数据。每个用例在子进程里运行，崩溃只记为该用例失败。

    g++ -O2 -DNDEBUG -std=c++17 -fopenmp -mavx2 -mfma -Iinclude -Ibuild/include \
        tools/test/test_closed_loop.cpp build/libFalcons_fse.a -lzstd -o cl_avx
    g++ -O2 -DNDEBUG -std=c++17 -fopenmp -Iinclude -Ibuild/include \
        tools/test/test_closed_loop.cpp build/libFalcons_fse.a -lzstd -o cl_scalar
    ./cl_avx gen a 1000 && ./cl_scalar gen s 1000
    ./cl_avx check a && ./cl_scalar check a && ./cl_avx check s && ./cl_scalar check s
    cmp a/0.cmp s/0.cmp    # 两种构建的压缩产物逐字节相同

`gen` 检查误差界和 NaN/Inf 是否保留，`check` 用 5 种线程数解压，并和压缩端自己的解压结果逐字节比较。
