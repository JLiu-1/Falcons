/**
 * @file Interpolation_quantizer_Omp.inl
 * @author Bing Lu
 * @brief Falcons parallel SIMD interpolation and quantization kernels.
 *
 * Copyright (c) 2026 Bing Lu.
 */

namespace Falcons {
#ifdef __AVX2__
    // uint16 quant-code pack/widen helpers (shared with the serial .inl via the include guard).
#ifndef Falcons_QUANT16_HELPERS
#define Falcons_QUANT16_HELPERS
    static ALWAYS_INLINE void store_quant8(int16_t* p, __m256i v) {
        const __m128i hi = _mm256_extracti128_si256(v, 1);
        const __m128i packed = _mm_packs_epi32(_mm256_castsi256_si128(v), hi);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(p), packed);
    }
    static ALWAYS_INLINE __m256i load_quant8(const int16_t* p) {
        return _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    }
    static ALWAYS_INLINE void store_quant4(int16_t* p, __m128i v) {
        _mm_storel_epi64(reinterpret_cast<__m128i*>(p), _mm_packs_epi32(v, v));
    }
    static ALWAYS_INLINE __m128i load_quant4(const int16_t* p) {
        return _mm_cvtepi16_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
    }
#endif // Falcons_QUANT16_HELPERS

    template <class T, uint N, class QuantizerOMP>
    template<typename U, typename>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::quantize_1D_float (__m256& sum, __m256& ori_avx, __m256& quant_avx, T tmp[8]) {

        __m256d quant_avx_low  = _mm256_cvtps_pd(_mm256_castps256_ps128(quant_avx));
        quant_avx_low  = _mm256_round_pd(_mm256_mul_pd(quant_avx_low,  ebx2_r_avx),  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        __m256d mask_low = _mm256_and_pd(
            _mm256_cmp_pd(quant_avx_low, nradius_avx, _CMP_GT_OQ),
            _mm256_cmp_pd(quant_avx_low, radius_avx, _CMP_LT_OQ)
        );
        quant_avx_low = _mm256_blendv_pd(zero_avx_d, quant_avx_low, mask_low);

        __m256d quant_avx_high = _mm256_cvtps_pd(_mm256_extractf128_ps(quant_avx, 1));
        quant_avx_high = _mm256_round_pd(_mm256_mul_pd(quant_avx_high, ebx2_r_avx), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256d mask_high = _mm256_and_pd(
            _mm256_cmp_pd(quant_avx_high, nradius_avx, _CMP_GT_OQ),
            _mm256_cmp_pd(quant_avx_high, radius_avx, _CMP_LT_OQ)
        );
        quant_avx_high = _mm256_blendv_pd(zero_avx_d, quant_avx_high, mask_high);

        // dequantization for decompression
        __m256d decompressed_low = _mm256_fmadd_pd(quant_avx_low, ebx2_avx,
                                _mm256_cvtps_pd(_mm256_castps256_ps128(sum)));
        __m256d decompressed_high = _mm256_fmadd_pd(quant_avx_high, ebx2_avx,
                                _mm256_cvtps_pd(_mm256_extractf128_ps(sum, 1)));

        quant_avx = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(quant_avx_low)),
                        _mm256_cvtpd_ps(quant_avx_high), 1);

        __m256 decompressed = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm256_cvtpd_ps(decompressed_low)),
            _mm256_cvtpd_ps(decompressed_high), 1);

        __m256 err_dequan = _mm256_sub_ps(decompressed, ori_avx);

        _mm256_storeu_ps(tmp, decompressed);

        __m256 mask = _mm256_and_ps(
                _mm256_cmp_ps(err_dequan, nrel_eb_avx_f, _CMP_GE_OQ),
                _mm256_cmp_ps(err_dequan, rel_eb_avx_f, _CMP_LE_OQ)
        );

        quant_avx = _mm256_blendv_ps(nradius_avx_f, quant_avx, mask);
    }

    template <class T, uint N, class QuantizerOMP>
    template<typename U, typename>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::quantize_1D_double (__m256d& sum, __m256d& ori_avx, __m256d& quant_avx, T tmp[4]) {
        quant_avx = _mm256_round_pd(_mm256_mul_pd(quant_avx,  ebx2_r_avx),  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        __m256d mask = _mm256_and_pd(
            _mm256_cmp_pd(quant_avx, nradius_avx, _CMP_GT_OQ),
            _mm256_cmp_pd(quant_avx, radius_avx, _CMP_LT_OQ)
        );
        quant_avx = _mm256_blendv_pd(zero_avx_d, quant_avx, mask); // out-of-range -> 0 (valid code) not escape sentinel; matches SVE2/float, fixes unpred-FIFO desync

        __m256d decompressed = _mm256_fmadd_pd(quant_avx, ebx2_avx, sum);
        _mm256_storeu_pd(tmp, decompressed);
        __m256d err_dequan = _mm256_sub_pd(decompressed, ori_avx);

        mask = _mm256_and_pd(
                _mm256_cmp_pd(err_dequan, nrel_eb_avx_d, _CMP_GE_OQ),
                _mm256_cmp_pd(err_dequan, rel_eb_avx_d, _CMP_LE_OQ)
        );
        quant_avx = _mm256_blendv_pd(nradius_avx, quant_avx, mask);
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool EnableInnerOmp, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_linear_and_quantize_1D_line(
        T *data, const size_t &len, size_t& offset, size_t& cur_ij_offset, int& tid, QuantizeFunc &&quantize_func) {
        if (len <= 1) {
            return;
        }
        auto odd_len = len / 2;
        auto even_len = len - odd_len;
        T *odd_data = data + cur_ij_offset + offset;
        size_t odd_offset = offset << 1;
        size_t i = 0;

        if constexpr (std::is_same_v<T, float>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256 factor = _mm256_set1_ps(0.5f);
            if constexpr (EnableInnerOmp) {
                const size_t vec_chunks = odd_len > step ? (odd_len - 1) / step : 0;
                #pragma omp parallel for
                for (size_t chunk = 0; chunk < vec_chunks; ++chunk) {
                    size_t cur_i = chunk * step;
                    int cur_tid = omp_get_thread_num();
                    T a[8], b[8];
                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        size_t even = (cur_i + j) << 1;
                        a[j] = data[cur_ij_offset + even * offset];
                        b[j] = data[cur_ij_offset + (even + 2) * offset];
                    }
                    __m256 sum = _mm256_mul_ps(_mm256_add_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b)), factor);
                    size_t start = cur_i;
                    quantize_float<CompMode, step, false, SkipOverwrite>(sum, start, odd_data, odd_offset, odd_len, cur_tid);
                }
                i = vec_chunks * step;
            } else {
                for (; i + step < odd_len; i += step) {
                    T a[8], b[8];
                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        size_t even = (i + j) << 1;
                        a[j] = data[cur_ij_offset + even * offset];
                        b[j] = data[cur_ij_offset + (even + 2) * offset];
                    }
                    __m256 sum = _mm256_mul_ps(_mm256_add_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b)), factor);
                    size_t start = i;
                    quantize_float<CompMode, step, false, SkipOverwrite>(sum, start, odd_data, odd_offset, odd_len, tid);
                }
            }
        } else if constexpr (std::is_same_v<T, double>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256d factor = _mm256_set1_pd(0.5);
            if constexpr (EnableInnerOmp) {
                const size_t vec_chunks = odd_len > step ? (odd_len - 1) / step : 0;
                #pragma omp parallel for
                for (size_t chunk = 0; chunk < vec_chunks; ++chunk) {
                    size_t cur_i = chunk * step;
                    int cur_tid = omp_get_thread_num();
                    T a[4], b[4];
                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        size_t even = (cur_i + j) << 1;
                        a[j] = data[cur_ij_offset + even * offset];
                        b[j] = data[cur_ij_offset + (even + 2) * offset];
                    }
                    __m256d sum = _mm256_mul_pd(_mm256_add_pd(_mm256_loadu_pd(a), _mm256_loadu_pd(b)), factor);
                    size_t start = cur_i;
                    quantize_double<CompMode, step, false, SkipOverwrite>(sum, start, odd_data, odd_offset, odd_len, cur_tid);
                }
                i = vec_chunks * step;
            } else {
                for (; i + step < odd_len; i += step) {
                    T a[4], b[4];
                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        size_t even = (i + j) << 1;
                        a[j] = data[cur_ij_offset + even * offset];
                        b[j] = data[cur_ij_offset + (even + 2) * offset];
                    }
                    __m256d sum = _mm256_mul_pd(_mm256_add_pd(_mm256_loadu_pd(a), _mm256_loadu_pd(b)), factor);
                    size_t start = i;
                    quantize_double<CompMode, step, false, SkipOverwrite>(sum, start, odd_data, odd_offset, odd_len, tid);
                }
            }
        }
        for (; i + 1 < odd_len; ++i) {
            size_t start = ((i << 1) + 1) * offset;
            quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + start, data[cur_ij_offset + start],
                          interp_linear(data[cur_ij_offset + (i << 1) * offset],
                                        data[cur_ij_offset + ((i << 1) + 2) * offset]), tid);
        }
        T pred_edge;
        if (len < 3) {
            pred_edge = data[cur_ij_offset + ((even_len - 1) << 1) * offset];
        } else {
            pred_edge = interp_linear1(data[cur_ij_offset + ((even_len - 2) << 1) * offset],
                                       data[cur_ij_offset + ((even_len - 1) << 1) * offset]);
        }
        int last = 2 * odd_len - 1;
        quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[cur_ij_offset + last * offset], pred_edge, tid);
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool EnableInnerOmp, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_cubic_and_quantize_1D_line(
        T *data, const size_t &len, size_t& offset, size_t& cur_ij_offset, int& tid, QuantizeFunc &&quantize_func) {
        if (len <= 1) {
            return;
        }
        auto odd_len = len / 2;
        auto even_len = len - odd_len;
        T pred_first;
        if (even_len < 2) {
            pred_first = data[cur_ij_offset];
        } else if (even_len < 3) {
            pred_first = interp_linear(data[cur_ij_offset], data[cur_ij_offset + 2 * offset]);
        } else {
            pred_first = interp_quad_1(data[cur_ij_offset], data[cur_ij_offset + 2 * offset],
                                       data[cur_ij_offset + 4 * offset]);
        }
        quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + offset, data[cur_ij_offset + offset], pred_first, tid);

        T *odd_data = data + cur_ij_offset + offset;
        size_t odd_offset = offset << 1;
        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256 nine = _mm256_set1_ps(9.0f);
            const __m256 factor = _mm256_set1_ps(1.0f / 16.0f);
            if constexpr (EnableInnerOmp) {
                const size_t vec_chunks = even_len > step + 2 ? (even_len - 3) / step : 0;
                #pragma omp parallel for
                for (size_t chunk = 0; chunk < vec_chunks; ++chunk) {
                    size_t cur_i = chunk * step;
                    int cur_tid = omp_get_thread_num();
                    T a[8], b[8], c[8], d[8];
                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        size_t even = (cur_i + j) << 1;
                        a[j] = data[cur_ij_offset + even * offset];
                        b[j] = data[cur_ij_offset + (even + 2) * offset];
                        c[j] = data[cur_ij_offset + (even + 4) * offset];
                        d[j] = data[cur_ij_offset + (even + 6) * offset];
                    }
                    __m256 sum = _mm256_mul_ps(
                        _mm256_sub_ps(
                            _mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_loadu_ps(b), _mm256_loadu_ps(c)), nine),
                                          _mm256_loadu_ps(a)),
                            _mm256_loadu_ps(d)),
                        factor);
                    size_t start = cur_i + 1;
                    quantize_float<CompMode, step, false, SkipOverwrite>(sum, start, odd_data, odd_offset, odd_len, cur_tid);
                }
                i = vec_chunks * step;
            } else {
                for (; i + step + 2 < even_len; i += step) {
                    T a[8], b[8], c[8], d[8];
                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        size_t even = (i + j) << 1;
                        a[j] = data[cur_ij_offset + even * offset];
                        b[j] = data[cur_ij_offset + (even + 2) * offset];
                        c[j] = data[cur_ij_offset + (even + 4) * offset];
                        d[j] = data[cur_ij_offset + (even + 6) * offset];
                    }
                    __m256 sum = _mm256_mul_ps(
                        _mm256_sub_ps(
                            _mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_loadu_ps(b), _mm256_loadu_ps(c)), nine),
                                          _mm256_loadu_ps(a)),
                            _mm256_loadu_ps(d)),
                        factor);
                    size_t start = i + 1;
                    quantize_float<CompMode, step, false, SkipOverwrite>(sum, start, odd_data, odd_offset, odd_len, tid);
                }
            }
        } else if constexpr (std::is_same_v<T, double>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256d nine = _mm256_set1_pd(9.0);
            const __m256d factor = _mm256_set1_pd(1.0 / 16.0);
            if constexpr (EnableInnerOmp) {
                const size_t vec_chunks = even_len > step + 2 ? (even_len - 3) / step : 0;
                #pragma omp parallel for
                for (size_t chunk = 0; chunk < vec_chunks; ++chunk) {
                    size_t cur_i = chunk * step;
                    int cur_tid = omp_get_thread_num();
                    T a[4], b[4], c[4], d[4];
                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        size_t even = (cur_i + j) << 1;
                        a[j] = data[cur_ij_offset + even * offset];
                        b[j] = data[cur_ij_offset + (even + 2) * offset];
                        c[j] = data[cur_ij_offset + (even + 4) * offset];
                        d[j] = data[cur_ij_offset + (even + 6) * offset];
                    }
                    __m256d sum = _mm256_mul_pd(
                        _mm256_sub_pd(
                            _mm256_sub_pd(_mm256_mul_pd(_mm256_add_pd(_mm256_loadu_pd(b), _mm256_loadu_pd(c)), nine),
                                          _mm256_loadu_pd(a)),
                            _mm256_loadu_pd(d)),
                        factor);
                    size_t start = cur_i + 1;
                    quantize_double<CompMode, step, false, SkipOverwrite>(sum, start, odd_data, odd_offset, odd_len, cur_tid);
                }
                i = vec_chunks * step;
            } else {
                for (; i + step + 2 < even_len; i += step) {
                    T a[4], b[4], c[4], d[4];
                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        size_t even = (i + j) << 1;
                        a[j] = data[cur_ij_offset + even * offset];
                        b[j] = data[cur_ij_offset + (even + 2) * offset];
                        c[j] = data[cur_ij_offset + (even + 4) * offset];
                        d[j] = data[cur_ij_offset + (even + 6) * offset];
                    }
                    __m256d sum = _mm256_mul_pd(
                        _mm256_sub_pd(
                            _mm256_sub_pd(_mm256_mul_pd(_mm256_add_pd(_mm256_loadu_pd(b), _mm256_loadu_pd(c)), nine),
                                          _mm256_loadu_pd(a)),
                            _mm256_loadu_pd(d)),
                        factor);
                    size_t start = i + 1;
                    quantize_double<CompMode, step, false, SkipOverwrite>(sum, start, odd_data, odd_offset, odd_len, tid);
                }
            }
        }
        for (; i + 3 < even_len; ++i) {
            size_t start = ((i << 1) + 3) * offset;
            quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + start, data[cur_ij_offset + start],
                          interp_cubic(data[cur_ij_offset + (i << 1) * offset],
                                       data[cur_ij_offset + ((i << 1) + 2) * offset],
                                       data[cur_ij_offset + ((i << 1) + 4) * offset],
                                       data[cur_ij_offset + ((i << 1) + 6) * offset]), tid);
        }
        if (odd_len > 1) {
            if (odd_len < even_len) {
                T edge_pred = interp_quad_2(data[cur_ij_offset + ((even_len - 3) << 1) * offset],
                                            data[cur_ij_offset + ((even_len - 2) << 1) * offset],
                                            data[cur_ij_offset + ((even_len - 1) << 1) * offset]);
                int last = 2 * odd_len - 1;
                quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[cur_ij_offset + last * offset], edge_pred, tid);
            } else {
                if (odd_len > 2) {
                    T edge_pred = interp_quad_2(data[cur_ij_offset + ((even_len - 3) << 1) * offset],
                                                data[cur_ij_offset + ((even_len - 2) << 1) * offset],
                                                data[cur_ij_offset + ((even_len - 1) << 1) * offset]);
                    int last = 2 * odd_len - 3;
                    quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[cur_ij_offset + last * offset], edge_pred, tid);
                }
                T edge_pred = interp_linear1(data[cur_ij_offset + ((even_len - 2) << 1) * offset],
                                             data[cur_ij_offset + ((even_len - 1) << 1) * offset]);
                int last = 2 * odd_len - 1;
                quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[cur_ij_offset + last * offset], edge_pred, tid);
            }
        }
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_linear_and_quantize_1D(const T * buf, const size_t &len, T* data,
        size_t&  offset, size_t& cur_ij_offset, int& tid, QuantizeFunc &&quantize_func) {
        if(len == 1)
            return;

        auto odd_len = len / 2;
        auto even_len = len - odd_len;
        size_t i = 0;

        if constexpr (std::is_same_v<T, float>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256 factor = _mm256_set1_ps(0.5f);

            for (; i + step < odd_len; i += step) {
                __m256 va = _mm256_loadu_ps(buf + i );
                __m256 vb = _mm256_loadu_ps(buf + i + 1);
                __m256 sum = _mm256_add_ps(va, vb);
                sum = _mm256_mul_ps(sum, factor);
                size_t start = (i << 1) + 1;

                if constexpr (CompMode == COMPMODE::COMP) {
                    T ori[8];
                    size_t base = start * offset;
                    size_t offsetx2 = offset << 1;

                    ori[0] = data[base];
                    ori[1] = data[base + offsetx2];
                    ori[2] = data[base + (offsetx2 << 1)];
                    ori[3] = data[base + 3 * offsetx2];
                    ori[4] = data[base + (offsetx2 << 2)];
                    ori[5] = data[base + 5 * offsetx2];
                    ori[6] = data[base + 6 * offsetx2];
                    ori[7] = data[base + 7 * offsetx2];


                    __m256 ori_avx = _mm256_loadu_ps(ori);
                    __m256 quant_avx = _mm256_sub_ps(ori_avx, sum); // prediction error
                    float tmp[8];
                    quantize_1D_float(sum, ori_avx, quant_avx, tmp);
                    __m256i quant_avx_i = _mm256_cvtps_epi32(quant_avx);
                    if constexpr (!SkipOverwrite) {
                        #pragma unroll
                        for (size_t j = 0; j < step; ++j) {
                            data[(start + (j << 1)) * offset] = tmp[j];
                        }
                    }
                    unsigned esc = static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(
                                       _mm256_cmpeq_epi32(quant_avx_i, _mm256_set1_epi32(-32768)))));
                    while (esc) {
                        int k = __builtin_ctz(esc);
                        if constexpr (!SkipOverwrite) data[(start + (k << 1)) * offset] = ori[k];
                        quantizer.save_unpred2(ori[k], tid);
                        esc &= esc - 1;
                    }
                    store_quant8(local_quant_inds[tid] + local_quant_index[tid].value, quant_avx_i);
                    local_quant_index[tid].value += step;
                }
                else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
                    __m256i quant_avx_i = load_quant8(local_quant_inds[tid] + local_quant_index[tid].value);

                    __m256d decompressed_low  = _mm256_cvtepi32_pd(_mm256_castsi256_si128(quant_avx_i));
                    decompressed_low = _mm256_fmadd_pd(decompressed_low, ebx2_avx, _mm256_cvtps_pd(_mm256_castps256_ps128(sum)));

                    __m256d decompressed_high = _mm256_cvtepi32_pd(_mm256_extracti128_si256(quant_avx_i, 1));
                    decompressed_high = _mm256_fmadd_pd(decompressed_high, ebx2_avx, _mm256_cvtps_pd(_mm256_extractf128_ps(sum, 1)));

                     __m256 decompressed = _mm256_insertf128_ps(
                        _mm256_castps128_ps256(_mm256_cvtpd_ps(decompressed_low)),
                        _mm256_cvtpd_ps(decompressed_high), 1);
                    float tmp[8];
                    _mm256_storeu_ps(tmp, decompressed);

                    if constexpr (!SkipOverwrite) {
                        #pragma unroll
                        for (size_t j = 0; j < step; ++j) {
                            data[(start + (j << 1)) * offset] = tmp[j];
                        }
                    }
                    unsigned esc = static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(
                                       _mm256_cmpeq_epi32(quant_avx_i, _mm256_set1_epi32(-32768)))));
                    while (esc) {
                        int k = __builtin_ctz(esc);
                        data[(start + (k << 1)) * offset] = quantizer.recover_unpred2(tid);
                        esc &= esc - 1;
                    }
                    local_quant_index[tid].value += step;
                }
            }
            for (; i + 1 < odd_len; ++i) {
                size_t start = ((i << 1) + 1) * offset;
                quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + start, data[start], interp_linear(buf[i], buf[i + 1]), tid);
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256d factor = _mm256_set1_pd(0.5);

            for (; i + step < odd_len; i += step) {
                __m256d va = _mm256_loadu_pd(buf + i);
                __m256d vb = _mm256_loadu_pd(buf + i + 1);

                __m256d sum = _mm256_add_pd(va, vb);
                sum = _mm256_mul_pd(sum, factor);

                size_t start = (i << 1) + 1;
                if constexpr (CompMode == COMPMODE::COMP) {
                    T ori[4];
                    size_t base = start * offset;
                    size_t offsetx2 = offset << 1;

                    ori[0] = data[base];
                    ori[1] = data[base + offsetx2];
                    ori[2] = data[base + (offsetx2 << 1)];
                    ori[3] = data[base + 3 * offsetx2];

                    __m256d ori_avx = _mm256_loadu_pd(ori);
                    __m256d quant_avx = _mm256_sub_pd(ori_avx, sum); // prediction error
                    T tmp[4];
                    quantize_1D_double(sum, ori_avx, quant_avx, tmp);

                    __m128i quant_avx_i = _mm256_cvtpd_epi32(quant_avx);

                    if constexpr (!SkipOverwrite) {
                        #pragma unroll
                        for (size_t j = 0; j < step; ++j) {
                            data[(start + (j << 1)) * offset] = tmp[j];
                        }
                    }
                    unsigned esc = static_cast<unsigned>(_mm_movemask_ps(_mm_castsi128_ps(
                                       _mm_cmpeq_epi32(quant_avx_i, _mm_set1_epi32(-32768)))));
                    while (esc) {
                        int k = __builtin_ctz(esc);
                        if constexpr (!SkipOverwrite) data[(start + (k << 1)) * offset] = ori[k];
                        quantizer.save_unpred2(ori[k], tid);
                        esc &= esc - 1;
                    }
                    store_quant4(local_quant_inds[tid] + local_quant_index[tid].value, quant_avx_i);
                    local_quant_index[tid].value += step;
                }
                else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
                    __m128i quant_avx_i = load_quant4(local_quant_inds[tid] + local_quant_index[tid].value);

                    __m256d decompressed = _mm256_fmadd_pd(_mm256_cvtepi32_pd(quant_avx_i),
                                            ebx2_avx, sum);
                    T tmp[4];
                    _mm256_storeu_pd(tmp, decompressed);

                    if constexpr (!SkipOverwrite) {
                        #pragma unroll
                        for (size_t j = 0; j < step; ++j) {
                            data[(start + (j << 1)) * offset] = tmp[j];
                        }
                    }
                    unsigned esc = static_cast<unsigned>(_mm_movemask_ps(_mm_castsi128_ps(
                                       _mm_cmpeq_epi32(quant_avx_i, _mm_set1_epi32(-32768)))));
                    while (esc) {
                        int k = __builtin_ctz(esc);
                        data[(start + (k << 1)) * offset] = quantizer.recover_unpred2(tid);
                        esc &= esc - 1;
                    }
                    local_quant_index[tid].value += step;
                }

            }
            for (; i + 1 < odd_len; ++i) {
                size_t start = ((i << 1) + 1) * offset;
                quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + start, data[start], interp_linear(buf[i], buf[i + 1]), tid);
            }
        }
        T pred_edge;
        if(len < 3 )
            pred_edge = buf[even_len - 1];
        else
            pred_edge = interp_linear1(buf[even_len - 2], buf[even_len - 1]);
        int last = 2 * odd_len - 1;
        quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset , data[last * offset], pred_edge, tid);
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_cubic_and_quantize_1D(const T * buf, const size_t &len, T* data,
        size_t&  offset, size_t& cur_ij_offset, int& tid, QuantizeFunc &&quantize_func) {
       // assert(len <= max_dim);
        if(len == 1)
            return;

        auto odd_len = len / 2;
        auto even_len = len - odd_len;

        T pred_first;
        if(even_len < 2)
            pred_first = (buf[0]);
        else if(even_len < 3)
            pred_first = interp_linear(buf[0], buf[1]);
        else
            pred_first = interp_quad_1(buf[0], buf[1], buf[2]);
        quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + offset , data[offset], pred_first, tid);

        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256 nine  = _mm256_set1_ps(9.0f);
            const __m256 factor = _mm256_set1_ps(1.0f / 16.0f);

            for (; i + step + 2 < even_len; i += step) {
                __m256 va = _mm256_loadu_ps(buf + i);
                __m256 vb = _mm256_loadu_ps(buf + i + 1);
                __m256 vc = _mm256_loadu_ps(buf + i + 2);
                __m256 vd = _mm256_loadu_ps(buf + i + 3);

                 __m256 sum = _mm256_add_ps(vb, vc);
                 sum = _mm256_mul_ps(sum, nine);
                 sum = _mm256_sub_ps(sum, va);
                sum = _mm256_sub_ps(sum, vd);
                sum = _mm256_mul_ps(sum, factor);

                size_t start = (i << 1) + 3;

                if constexpr (CompMode == COMPMODE::COMP) {
                    T ori[8];
                    size_t base = start * offset;
                    size_t offsetx2 = offset << 1;

                    ori[0] = data[base];
                    ori[1] = data[base + offsetx2];
                    ori[2] = data[base + (offsetx2 << 1)];
                    ori[3] = data[base + 3 * offsetx2];
                    ori[4] = data[base + (offsetx2 << 2)];
                    ori[5] = data[base + 5 * offsetx2];
                    ori[6] = data[base + 6 * offsetx2];
                    ori[7] = data[base + 7 * offsetx2];


                    __m256 ori_avx = _mm256_loadu_ps(ori);
                    __m256 quant_avx = _mm256_sub_ps(ori_avx, sum); // prediction error
                    float tmp[8];
                    quantize_1D_float(sum, ori_avx, quant_avx, tmp);

                    __m256i quant_avx_i = _mm256_cvtps_epi32(quant_avx);
                    if constexpr (!SkipOverwrite) {
                        #pragma unroll
                        for (size_t j = 0; j < step; ++j) {
                            data[(start + (j << 1)) * offset] = tmp[j];
                        }
                    }
                    unsigned esc = static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(
                                       _mm256_cmpeq_epi32(quant_avx_i, _mm256_set1_epi32(-32768)))));
                    while (esc) {
                        int k = __builtin_ctz(esc);
                        if constexpr (!SkipOverwrite) data[(start + (k << 1)) * offset] = ori[k];
                        quantizer.save_unpred2(ori[k], tid);
                        esc &= esc - 1;
                    }

                    store_quant8(local_quant_inds[tid] + local_quant_index[tid].value, quant_avx_i);
                    local_quant_index[tid].value += step;
                }
                else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
                    __m256i quant_avx_i = load_quant8(local_quant_inds[tid] + local_quant_index[tid].value);

                    __m256d decompressed_low  = _mm256_cvtepi32_pd(_mm256_castsi256_si128(quant_avx_i));
                    decompressed_low = _mm256_fmadd_pd(decompressed_low, ebx2_avx, _mm256_cvtps_pd(_mm256_castps256_ps128(sum)));

                    __m256d decompressed_high = _mm256_cvtepi32_pd(_mm256_extracti128_si256(quant_avx_i, 1));
                    decompressed_high = _mm256_fmadd_pd(decompressed_high, ebx2_avx, _mm256_cvtps_pd(_mm256_extractf128_ps(sum, 1)));

                     __m256 decompressed = _mm256_insertf128_ps(
                        _mm256_castps128_ps256(_mm256_cvtpd_ps(decompressed_low)),
                        _mm256_cvtpd_ps(decompressed_high), 1);
                    float tmp[8];
                    _mm256_storeu_ps(tmp, decompressed);

                    if constexpr (!SkipOverwrite) {
                        #pragma unroll
                        for (size_t j = 0; j < step; ++j) {
                            data[(start + (j << 1)) * offset] = tmp[j];
                        }
                    }
                    unsigned esc = static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(
                                       _mm256_cmpeq_epi32(quant_avx_i, _mm256_set1_epi32(-32768)))));
                    while (esc) {
                        int k = __builtin_ctz(esc);
                        data[(start + (k << 1)) * offset] = quantizer.recover_unpred2(tid);
                        esc &= esc - 1;
                    }
                    local_quant_index[tid].value += step;
                }
            }
            for (; i + 3 < even_len; ++i) {
                size_t start = ((i << 1) + 3) * offset;
                quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + start, data[start],
                              interp_cubic(buf[i], buf[i + 1], buf[i + 2], buf[i + 3]), tid);
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256d nine  = _mm256_set1_pd(9.0);
            const __m256d factor = _mm256_set1_pd(1.0 / 16.0);

            for (; i + step + 2 < even_len; i += step) {
                __m256d va = _mm256_loadu_pd(buf + i);
                __m256d vb = _mm256_loadu_pd(buf + i + 1);
                __m256d vc = _mm256_loadu_pd(buf + i + 2);
                __m256d vd = _mm256_loadu_pd(buf + i + 3);

                __m256d sum = _mm256_add_pd(vb, vc);
                 sum = _mm256_mul_pd(sum, nine);
                 sum = _mm256_sub_pd(sum, va);
                sum = _mm256_sub_pd(sum, vd);
                sum = _mm256_mul_pd(sum, factor);
                // _mm256_storeu_pd(p + i + 1, sum);
                size_t start = (i << 1) + 3;
                // T pred[4];
                // _mm256_storeu_pd(pred, sum);

                if constexpr (CompMode == COMPMODE::COMP) {
                    T ori[4];
                    size_t base = start * offset;
                    size_t offsetx2 = offset << 1;

                    ori[0] = data[base];
                    ori[1] = data[base + offsetx2];
                    ori[2] = data[base + (offsetx2 << 1)];
                    ori[3] = data[base + 3 * offsetx2];

                    __m256d ori_avx = _mm256_loadu_pd(ori);
                    __m256d quant_avx = _mm256_sub_pd(ori_avx, sum); // prediction error
                    T tmp[4];
                    quantize_1D_double(sum, ori_avx, quant_avx, tmp);

                    __m128i quant_avx_i = _mm256_cvtpd_epi32(quant_avx);

                    if constexpr (!SkipOverwrite) {
                        #pragma unroll
                        for (size_t j = 0; j < step; ++j) {
                            data[(start + (j << 1)) * offset] = tmp[j];
                        }
                    }
                    unsigned esc = static_cast<unsigned>(_mm_movemask_ps(_mm_castsi128_ps(
                                       _mm_cmpeq_epi32(quant_avx_i, _mm_set1_epi32(-32768)))));
                    while (esc) {
                        int k = __builtin_ctz(esc);
                        if constexpr (!SkipOverwrite) data[(start + (k << 1)) * offset] = ori[k];
                        quantizer.save_unpred2(ori[k], tid);
                        esc &= esc - 1;
                    }
                    store_quant4(local_quant_inds[tid] + local_quant_index[tid].value, quant_avx_i);
                    local_quant_index[tid].value += step;
                }
                else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
                    __m128i quant_avx_i = load_quant4(local_quant_inds[tid] + local_quant_index[tid].value);

                    __m256d decompressed = _mm256_fmadd_pd(_mm256_cvtepi32_pd(quant_avx_i),
                                            ebx2_avx, sum);
                    T tmp[4];
                    _mm256_storeu_pd(tmp, decompressed);

                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        data[(start + (j << 1)) * offset] = tmp[j];
                    }
                    unsigned esc = static_cast<unsigned>(_mm_movemask_ps(_mm_castsi128_ps(
                                       _mm_cmpeq_epi32(quant_avx_i, _mm_set1_epi32(-32768)))));
                    while (esc) {
                        int k = __builtin_ctz(esc);
                        data[(start + (k << 1)) * offset] = quantizer.recover_unpred2(tid);
                        esc &= esc - 1;
                    }
                    local_quant_index[tid].value += step;
                }
            }
            for (; i + 3 < even_len; ++i) {
                size_t start = ((i << 1) + 3) * offset;
                quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + start, data[start],
                              interp_cubic(buf[i], buf[i + 1], buf[i + 2], buf[i + 3]), tid);
            }
        }
        if(odd_len > 1){
            if(odd_len < even_len){//the only boundary is p[len- 1]
                //odd_len < even_len so even_len > 2
                T edge_pred;
                edge_pred = interp_quad_2(buf[even_len - 3], buf[even_len - 2], buf[even_len - 1]);
                int last = 2 * odd_len - 1;
                quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[last * offset], edge_pred, tid);

            }
            else{//the boundary points are is p[len -2 ] and p[len -1 ]
                T edge_pred;
                if(odd_len > 2){ //len - 2
                 //odd_len = even_len so even_len > 2
                    edge_pred = interp_quad_2(buf[even_len - 3],  buf[even_len - 2], buf[even_len - 1]);
                    int last = 2 * odd_len - 3;
                    quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[last * offset], edge_pred, tid);
                }
                //len -1
                //odd_len = even_len so even_len > 1
                    edge_pred = interp_linear1(buf[even_len - 2], buf[even_len - 1]);
                    int last = 2 * odd_len - 1;
                    quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[last * offset], edge_pred, tid);


            }
        }
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_linear_and_quantize(const T * a, const T* b, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func) {
        size_t i = 0;

        if constexpr (std::is_same_v<T, float>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256 factor = _mm256_set1_ps(0.5f);

            for (; i + step <= len; i += step) {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vb = _mm256_loadu_ps(b + i);

                __m256 sum = _mm256_add_ps(va, vb);
                sum = _mm256_mul_ps(sum, factor);

                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode, step, true, SkipOverwrite>(sum, i, data, offset, len, tid);
            }
            if (i < len) {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vb = _mm256_loadu_ps(b + i);

                __m256 sum = _mm256_add_ps(va, vb);
                sum = _mm256_mul_ps(sum, factor);

                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256d factor = _mm256_set1_pd(0.5);

            for (; i + step <= len; i += step) {
                __m256d va = _mm256_loadu_pd(a + i);
                __m256d vb = _mm256_loadu_pd(b + i);

                __m256d sum = _mm256_add_pd(va, vb);
                sum = _mm256_mul_pd(sum, factor);
                // _mm256_storeu_pd(p + i, sum);
                // size_t start = i;
                quantize_double<CompMode, step, true, SkipOverwrite>(sum, i, data, offset, len, tid);
            }
            if (i < len) {
                __m256d va = _mm256_loadu_pd(a + i);
                __m256d vb = _mm256_loadu_pd(b + i);

                __m256d sum = _mm256_add_pd(va, vb);
                sum = _mm256_mul_pd(sum, factor);
                // _mm256_storeu_pd(p + i, sum);
                // size_t start = i;
                quantize_double<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
            }
        }

    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_cubic_and_quantize(const T * a, const T* b, T* c, T*d, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func) {

        size_t i = 0;

        if constexpr (std::is_same_v<T, float>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256 nine  = _mm256_set1_ps(9.0f);
            const __m256 factor = _mm256_set1_ps(1.0f / 16.0f);

            for (; i + step <= len; i += step) {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vb = _mm256_loadu_ps(b + i);
                __m256 vc = _mm256_loadu_ps(c + i);
                __m256 vd = _mm256_loadu_ps(d + i);

                 __m256 sum = _mm256_add_ps(vb, vc);
                 sum = _mm256_mul_ps(sum, nine);
                 sum = _mm256_sub_ps(sum, va);
                sum = _mm256_sub_ps(sum, vd);
                sum = _mm256_mul_ps(sum, factor);

                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode, step, true, SkipOverwrite>(sum, i, data, offset, len, tid);
            }
            if (i < len) {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vb = _mm256_loadu_ps(b + i);
                __m256 vc = _mm256_loadu_ps(c + i);
                __m256 vd = _mm256_loadu_ps(d + i);

                 __m256 sum = _mm256_add_ps(vb, vc);
                 sum = _mm256_mul_ps(sum, nine);
                 sum = _mm256_sub_ps(sum, va);
                sum = _mm256_sub_ps(sum, vd);
                sum = _mm256_mul_ps(sum, factor);

                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
            }

        }
        else if constexpr (std::is_same_v<T, double>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256d nine  = _mm256_set1_pd(9.0);
            const __m256d factor = _mm256_set1_pd(1.0 / 16.0);

            for (; i + step <= len; i += step) {
                __m256d va = _mm256_loadu_pd(a + i);
                __m256d vb = _mm256_loadu_pd(b + i);
                __m256d vc = _mm256_loadu_pd(c + i);
                __m256d vd = _mm256_loadu_pd(d + i);

                __m256d sum = _mm256_add_pd(vb, vc);
                 sum = _mm256_mul_pd(sum, nine);
                 sum = _mm256_sub_pd(sum, va);
                sum = _mm256_sub_pd(sum, vd);

                sum = _mm256_mul_pd(sum, factor);
                // _mm256_storeu_pd(p + i, sum);
                quantize_double<CompMode, step, true, SkipOverwrite>(sum, i, data, offset, len, tid);
            }
            if (i < len) {
                __m256d va = _mm256_loadu_pd(a + i);
                __m256d vb = _mm256_loadu_pd(b + i);
                __m256d vc = _mm256_loadu_pd(c + i);
                __m256d vd = _mm256_loadu_pd(d + i);

                __m256d sum = _mm256_add_pd(vb, vc);
                 sum = _mm256_mul_pd(sum, nine);
                 sum = _mm256_sub_pd(sum, va);
                sum = _mm256_sub_pd(sum, vd);

                sum = _mm256_mul_pd(sum, factor);
                // _mm256_storeu_pd(p + i, sum);
                quantize_double<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
            }
        }

    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_equal_and_quantize(const T * a, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func) {

        size_t i = 0;

        if constexpr (std::is_same_v<T, float>) {
            static constexpr size_t step = AVX_256_parallelism;
            for (; i + step <= len; i += step) {
                __m256 sum = _mm256_loadu_ps(a + i);
                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode, step, true, SkipOverwrite>(sum, i, data, offset, len, tid);
            }
            if (i < len) {
                __m256 sum = _mm256_loadu_ps(a + i);
                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
            }

        }
        else if constexpr (std::is_same_v<T, double>) {
            static constexpr size_t step = AVX_256_parallelism;
            for (; i + step <= len; i += step) {
                __m256d sum = _mm256_loadu_pd(a + i);
                quantize_double<CompMode, step, true, SkipOverwrite>(sum, i, data, offset, len, tid);
            }
            if (i < len) {
                __m256d sum = _mm256_loadu_pd(a + i);
                quantize_double<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
            }
        }
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_linear1_and_quantize(const T * a, const T* b, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func) {
        size_t i = 0;

        if constexpr (std::is_same_v<T, float>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256 half = _mm256_set1_ps(0.5f);
            const __m256 threehalf = _mm256_set1_ps(1.5f);
            for (; i + step <= len; i += step) {
                __m256 vb = _mm256_loadu_ps(b + i);
                __m256 va = _mm256_loadu_ps(a + i);
                vb = _mm256_mul_ps(vb, threehalf);              // 1.5*b
                __m256 sum = _mm256_fnmadd_ps(half, va, vb);    // 1.5*b - 0.5*a  (== interp_linear1, fused)
                quantize_float<CompMode, step, true, SkipOverwrite>(sum, i, data, offset, len, tid);
            }
            if (i < len) {
                __m256 vb = _mm256_loadu_ps(b + i);
                __m256 va = _mm256_loadu_ps(a + i);
                vb = _mm256_mul_ps(vb, threehalf);              // 1.5*b
                __m256 sum = _mm256_fnmadd_ps(half, va, vb);    // 1.5*b - 0.5*a  (== interp_linear1, fused)
                quantize_float<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256d half = _mm256_set1_pd(0.5);
            const __m256d threehalf = _mm256_set1_pd(1.5);
            for (; i + step <= len; i += step) {
                __m256d va = _mm256_loadu_pd(a + i);
                __m256d vb = _mm256_loadu_pd(b + i);
                vb = _mm256_mul_pd(vb, threehalf);              // 1.5*b
                __m256d sum = _mm256_fnmadd_pd(half, va, vb);   // 1.5*b - 0.5*a  (== interp_linear1, fused)
                quantize_double<CompMode, step, true, SkipOverwrite>(sum, i, data, offset, len, tid);
            }
            if (i < len) {
                __m256d va = _mm256_loadu_pd(a + i);
                __m256d vb = _mm256_loadu_pd(b + i);
                vb = _mm256_mul_pd(vb, threehalf);              // 1.5*b
                __m256d sum = _mm256_fnmadd_pd(half, va, vb);   // 1.5*b - 0.5*a  (== interp_linear1, fused)
                quantize_double<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
            }
        }
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_quad1_and_quantize(const T * a, const T* b, const T* c, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func) {
        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256 factor = _mm256_set1_ps(0.125f);
            const __m256 six = _mm256_set1_ps(6.0f);
            const __m256 three = _mm256_set1_ps(3.0f);

            for (; i + step <= len; i += step) {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vb = _mm256_loadu_ps(b + i);
                __m256 vc = _mm256_loadu_ps(c + i);
                vb = _mm256_fmsub_ps(vb, six, vc);           // 6*b - c
                __m256 sum = _mm256_fmadd_ps(va, three, vb); // 3*a + (6b-c)  (fully fused)
                sum = _mm256_mul_ps(sum, factor);            // *0.125
                quantize_float<CompMode, step, true, SkipOverwrite>(sum, i, data, offset, len, tid);
            }
            if (i < len) {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vb = _mm256_loadu_ps(b + i);
                __m256 vc = _mm256_loadu_ps(c + i);
                vb = _mm256_fmsub_ps(vb, six, vc);           // 6*b - c
                __m256 sum = _mm256_fmadd_ps(va, three, vb); // 3*a + (6b-c)  (fully fused)
                sum = _mm256_mul_ps(sum, factor);            // *0.125
                quantize_float<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256d factor = _mm256_set1_pd(0.125);
            const __m256d six = _mm256_set1_pd(6.0);
            const __m256d three = _mm256_set1_pd(3.0);

            for (; i + step <= len; i += step) {
                __m256d va = _mm256_loadu_pd(a + i);
                __m256d vb = _mm256_loadu_pd(b + i);
                __m256d vc = _mm256_loadu_pd(c + i);
                vb = _mm256_fmsub_pd(vb, six, vc);            // 6*b - c
                __m256d sum = _mm256_fmadd_pd(va, three, vb); // 3*a + (6b-c)  (fully fused)
                sum = _mm256_mul_pd(sum, factor);             // *0.125
                quantize_double<CompMode, step, true, SkipOverwrite>(sum, i, data, offset, len, tid);
            }
            if (i < len) {
                __m256d va = _mm256_loadu_pd(a + i);
                __m256d vb = _mm256_loadu_pd(b + i);
                __m256d vc = _mm256_loadu_pd(c + i);
                vb = _mm256_fmsub_pd(vb, six, vc);            // 6*b - c
                __m256d sum = _mm256_fmadd_pd(va, three, vb); // 3*a + (6b-c)  (fully fused)
                sum = _mm256_mul_pd(sum, factor);             // *0.125
                quantize_double<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
            }
        }

    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_quad2_and_quantize(const T * a, const T* b, const T* c, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func) {
        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256 factor = _mm256_set1_ps(0.125f);
            const __m256 six = _mm256_set1_ps(6.0f);
            const __m256 three = _mm256_set1_ps(3.0f);

            for (; i + step <= len; i += step) {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vc = _mm256_loadu_ps(c + i);
                __m256 vb = _mm256_loadu_ps(b + i);
                vb = _mm256_fmsub_ps(vb, six, va);           // 6*b - a
                __m256 sum = _mm256_fmadd_ps(vc, three, vb); // 3*c + (6b-a)  (fully fused)
                sum = _mm256_mul_ps(sum, factor);            // *0.125
                quantize_float<CompMode, step, true, SkipOverwrite>(sum, i, data, offset, len, tid);
            }
            if (i < len) {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vc = _mm256_loadu_ps(c + i);
                __m256 vb = _mm256_loadu_ps(b + i);
                vb = _mm256_fmsub_ps(vb, six, va);           // 6*b - a
                __m256 sum = _mm256_fmadd_ps(vc, three, vb); // 3*c + (6b-a)  (fully fused)
                sum = _mm256_mul_ps(sum, factor);            // *0.125
                quantize_float<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static constexpr size_t step = AVX_256_parallelism;
            const __m256d factor = _mm256_set1_pd(0.125);
            const __m256d six = _mm256_set1_pd(6.0);
            const __m256d three = _mm256_set1_pd(3.0);

            for (; i + step <= len; i += step) {
                __m256d vc = _mm256_loadu_pd(c + i);
                __m256d va = _mm256_loadu_pd(a + i);
                __m256d vb = _mm256_loadu_pd(b + i);
                vb = _mm256_fmsub_pd(vb, six, va);            // 6*b - a
                __m256d sum = _mm256_fmadd_pd(vc, three, vb); // 3*c + (6b-a)  (fully fused)
                sum = _mm256_mul_pd(sum, factor);             // *0.125
                quantize_double<CompMode, step, true, SkipOverwrite>(sum, i, data, offset, len, tid);
            }
            if (i < len) {
                __m256d vc = _mm256_loadu_pd(c + i);
                __m256d va = _mm256_loadu_pd(a + i);
                __m256d vb = _mm256_loadu_pd(b + i);
                vb = _mm256_fmsub_pd(vb, six, va);            // 6*b - a
                __m256d sum = _mm256_fmadd_pd(vc, three, vb); // 3*c + (6b-a)  (fully fused)
                sum = _mm256_mul_pd(sum, factor);             // *0.125
                quantize_double<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
            }
        }

    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, int step, bool FullOnly, bool SkipOverwrite, typename U, typename>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::quantize_float (__m256& sum, size_t& start, T*& data, size_t& offset, size_t& len, int tid) {
        if constexpr (CompMode == COMPMODE::COMP) {
            T ori[8];
            __m256 ori_avx;
            if constexpr (FullOnly) {
                // caller guarantees a full chunk: straight read, skip the per-lane tail routing
                if (offset == 1) ori_avx = _mm256_loadu_ps(data + start);
                else { for (size_t j = 0; j < step; ++j) ori[j] = data[(start + j) * offset]; ori_avx = _mm256_loadu_ps(ori); }
            } else {
                // tail: clamp past-end indices to the last valid element (avoid OOB read)
                for (size_t j = 0; j < step; ++j) {
                    size_t idx = (start + j < len) ? (start + j) : (len - 1);
                    ori[j] = data[idx * offset];
                }
                ori_avx = _mm256_loadu_ps(ori);
            }
            __m256 quant_avx = _mm256_sub_ps(ori_avx, sum); // prediction error
            // calculate quantization code
            __m256d quant_avx_low  = _mm256_cvtps_pd(_mm256_castps256_ps128(quant_avx));
            quant_avx_low  = _mm256_round_pd(_mm256_mul_pd(quant_avx_low,  ebx2_r_avx),  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

            __m256d mask_low = _mm256_and_pd(
                _mm256_cmp_pd(quant_avx_low, nradius_avx, _CMP_GT_OQ),
                _mm256_cmp_pd(quant_avx_low, radius_avx, _CMP_LT_OQ)
            );
            quant_avx_low = _mm256_blendv_pd(zero_avx_d, quant_avx_low, mask_low);

            __m256d quant_avx_high = _mm256_cvtps_pd(_mm256_extractf128_ps(quant_avx, 1));
            quant_avx_high = _mm256_round_pd(_mm256_mul_pd(quant_avx_high, ebx2_r_avx), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256d mask_high = _mm256_and_pd(
                _mm256_cmp_pd(quant_avx_high, nradius_avx, _CMP_GT_OQ),
                _mm256_cmp_pd(quant_avx_high, radius_avx, _CMP_LT_OQ)
            );
            quant_avx_high = _mm256_blendv_pd(zero_avx_d, quant_avx_high, mask_high);

            // dequantization for decompression
            __m256d decompressed_low = _mm256_fmadd_pd(quant_avx_low, ebx2_avx,
                                    _mm256_cvtps_pd(_mm256_castps256_ps128(sum)));
            __m256d decompressed_high = _mm256_fmadd_pd(quant_avx_high, ebx2_avx,
                                    _mm256_cvtps_pd(_mm256_extractf128_ps(sum, 1)));

            quant_avx = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(quant_avx_low)),
                            _mm256_cvtpd_ps(quant_avx_high), 1);

            __m256 decompressed = _mm256_insertf128_ps(
                _mm256_castps128_ps256(_mm256_cvtpd_ps(decompressed_low)),
                _mm256_cvtpd_ps(decompressed_high), 1);

            __m256 err_dequan = _mm256_sub_ps(decompressed, ori_avx);
            float tmp[8];
            _mm256_storeu_ps(tmp, decompressed);

            __m256 mask = _mm256_and_ps(
                    _mm256_cmp_ps(err_dequan, nrel_eb_avx_f, _CMP_GE_OQ),
                    _mm256_cmp_ps(err_dequan, rel_eb_avx_f, _CMP_LE_OQ)
            );

            quant_avx = _mm256_blendv_ps(nradius_avx_f, quant_avx, mask);
            // escape lanes = those whose FINAL code is the sentinel (-radius): out-of-range OR err-fail.
            // Decompress marks escapes with the same test (code == -32768), so COMP/DECOMP agree per-lane.
            __m256 esc_mask = _mm256_cmp_ps(quant_avx, nradius_avx_f, _CMP_EQ_OQ);
            __m256i quant_avx_i = _mm256_cvtps_epi32(quant_avx);

            size_t processed = 0;
            if constexpr (FullOnly) {
                processed = step;
                unsigned esc = static_cast<unsigned>(_mm256_movemask_ps(esc_mask)) & ((1u << step) - 1);
                if (offset == 1) {
                    __m256 out = _mm256_blendv_ps(decompressed, ori_avx, esc_mask);  // escape lane -> keep original
                    if constexpr (!SkipOverwrite) {
                        _mm256_storeu_ps(data + start, out);
                    }
                    if (esc) _mm256_storeu_ps(ori, ori_avx);   // contiguous read left ori[] unfilled
                    while (esc) { int k = __builtin_ctz(esc); quantizer.save_unpred2(ori[k], tid); esc &= esc - 1; }
                } else {
                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        if constexpr (!SkipOverwrite) data[(start + j) * offset] = tmp[j];
                    }
                    while (esc) {
                        int k = __builtin_ctz(esc);
                        if constexpr (!SkipOverwrite) data[(start + k) * offset] = ori[k];
                        quantizer.save_unpred2(ori[k], tid);
                        esc &= esc - 1;
                    }
                }
            } else {
                size_t j = 0;
                #pragma unroll
                for ( ; j < step && start + j < len; ++j)
                    if constexpr (!SkipOverwrite) data[(start + j) * offset] = tmp[j];
                processed = j;
                unsigned esc = static_cast<unsigned>(_mm256_movemask_ps(esc_mask)) & ((1u << processed) - 1);
                while (esc) {
                    int k = __builtin_ctz(esc);
                    if constexpr (!SkipOverwrite) data[(start + k) * offset] = ori[k];
                    quantizer.save_unpred2(ori[k], tid);
                    esc &= esc - 1;
                }
            }
            store_quant8(local_quant_inds[tid] + local_quant_index[tid].value, quant_avx_i);
            local_quant_index[tid].value += processed;
        }
        else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
            __m256i quant_avx_i = load_quant8(local_quant_inds[tid] + local_quant_index[tid].value);

            __m256d decompressed_low  = _mm256_cvtepi32_pd(_mm256_castsi256_si128(quant_avx_i));
            decompressed_low = _mm256_fmadd_pd(decompressed_low, ebx2_avx, _mm256_cvtps_pd(_mm256_castps256_ps128(sum)));

            __m256d decompressed_high = _mm256_cvtepi32_pd(_mm256_extracti128_si256(quant_avx_i, 1));
            decompressed_high = _mm256_fmadd_pd(decompressed_high, ebx2_avx, _mm256_cvtps_pd(_mm256_extractf128_ps(sum, 1)));

                __m256 decompressed = _mm256_insertf128_ps(
                _mm256_castps128_ps256(_mm256_cvtpd_ps(decompressed_low)),
                _mm256_cvtpd_ps(decompressed_high), 1);
            float tmp[8];
            _mm256_storeu_ps(tmp, decompressed);

            size_t processed = 0;
            unsigned esc_all = static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(
                                   _mm256_cmpeq_epi32(quant_avx_i, _mm256_set1_epi32(-32768)))));
            if constexpr (FullOnly) {
                processed = step;
                unsigned esc = esc_all & ((1u << step) - 1);
                if (offset == 1) {
                    _mm256_storeu_ps(data + start, decompressed);
                    while (esc) { int k = __builtin_ctz(esc); data[start + k] = quantizer.recover_unpred2(tid); esc &= esc - 1; }
                } else {
                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) data[(start + j) * offset] = tmp[j];
                    while (esc) { int k = __builtin_ctz(esc); data[(start + k) * offset] = quantizer.recover_unpred2(tid); esc &= esc - 1; }
                }
            } else {
                size_t j = 0;
                #pragma unroll
                for ( ; j < step && start + j < len; ++j)
                    data[(start + j) * offset] = tmp[j];
                processed = j;
                unsigned esc = esc_all & ((1u << processed) - 1);
                while (esc) {
                    int k = __builtin_ctz(esc);
                    data[(start + k) * offset] = quantizer.recover_unpred2(tid);
                    esc &= esc - 1;
                }
            }
            local_quant_index[tid].value += processed;
        }
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, int step, bool FullOnly, bool SkipOverwrite, typename U, typename>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::quantize_double (__m256d& sum, size_t& start, T*& data, size_t& offset, size_t& len, int tid) {
        if constexpr (CompMode == COMPMODE::COMP) {
            T ori[4];
            __m256d ori_avx;
            if constexpr (FullOnly) {
                // caller guarantees a full chunk: straight read, skip the per-lane tail routing
                if (offset == 1) ori_avx = _mm256_loadu_pd(data + start);
                else { for (size_t j = 0; j < step; ++j) ori[j] = data[(start + j) * offset]; ori_avx = _mm256_loadu_pd(ori); }
            } else {
                // tail: clamp past-end indices to the last valid element (avoid OOB read)
                for (size_t j = 0; j < step; ++j) {
                    size_t idx = (start + j < len) ? (start + j) : (len - 1);
                    ori[j] = data[idx * offset];
                }
                ori_avx = _mm256_loadu_pd(ori);
            }
            __m256d quant_avx = _mm256_sub_pd(ori_avx, sum); // prediction error
            quant_avx = _mm256_round_pd(_mm256_mul_pd(quant_avx,  ebx2_r_avx),  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

            __m256d mask = _mm256_and_pd(
                _mm256_cmp_pd(quant_avx, nradius_avx, _CMP_GT_OQ),
                _mm256_cmp_pd(quant_avx, radius_avx, _CMP_LT_OQ)
            );
            quant_avx = _mm256_blendv_pd(nradius_avx, quant_avx, mask);  // out-of-range -> sentinel placeholder

            __m256d decompressed = _mm256_fmadd_pd(quant_avx, ebx2_avx, sum);
            T tmp[4];
            _mm256_storeu_pd(tmp, decompressed);
            __m256d err_dequan = _mm256_sub_pd(decompressed, ori_avx);

            mask = _mm256_and_pd(
                    _mm256_cmp_pd(err_dequan, nrel_eb_avx_d, _CMP_GE_OQ),
                    _mm256_cmp_pd(err_dequan, rel_eb_avx_d, _CMP_LE_OQ)
            );
            quant_avx = _mm256_blendv_pd(nradius_avx, quant_avx, mask);
            // escape lanes = those whose FINAL code is the sentinel (-radius): out-of-range OR err-fail.
            // Decompress marks escapes with the same test (code == -32768), so COMP/DECOMP agree per-lane.
            __m256d esc_mask = _mm256_cmp_pd(quant_avx, nradius_avx, _CMP_EQ_OQ);

            __m128i quant_avx_i = _mm256_cvtpd_epi32(quant_avx);

            size_t processed = 0;
            if constexpr (FullOnly) {
                processed = step;
                unsigned esc = static_cast<unsigned>(_mm256_movemask_pd(esc_mask)) & ((1u << step) - 1);
                if (offset == 1) {
                    __m256d out = _mm256_blendv_pd(decompressed, ori_avx, esc_mask);  // escape lane -> keep original
                    if constexpr (!SkipOverwrite) {
                        _mm256_storeu_pd(data + start, out);
                    }
                    if (esc) _mm256_storeu_pd(ori, ori_avx);   // contiguous read left ori[] unfilled
                    while (esc) { int k = __builtin_ctz(esc); quantizer.save_unpred2(ori[k], tid); esc &= esc - 1; }
                } else {
                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        if constexpr (!SkipOverwrite) data[(start + j) * offset] = tmp[j];
                    }
                    while (esc) {
                        int k = __builtin_ctz(esc);
                        if constexpr (!SkipOverwrite) data[(start + k) * offset] = ori[k];
                        quantizer.save_unpred2(ori[k], tid);
                        esc &= esc - 1;
                    }
                }
            } else {
                size_t j = 0;
                #pragma unroll
                for ( ; j < step && start + j < len; ++j)
                    if constexpr (!SkipOverwrite) data[(start + j) * offset] = tmp[j];
                processed = j;
                unsigned esc = static_cast<unsigned>(_mm256_movemask_pd(esc_mask)) & ((1u << processed) - 1);
                while (esc) {
                    int k = __builtin_ctz(esc);
                    if constexpr (!SkipOverwrite) data[(start + k) * offset] = ori[k];
                    quantizer.save_unpred2(ori[k], tid);
                    esc &= esc - 1;
                }
            }
            store_quant4(local_quant_inds[tid] + local_quant_index[tid].value, quant_avx_i);
            local_quant_index[tid].value += processed;
        }
        else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
            __m128i quant_avx_i = load_quant4(local_quant_inds[tid] + local_quant_index[tid].value);

            __m256d decompressed = _mm256_fmadd_pd(_mm256_cvtepi32_pd(quant_avx_i),
                                    ebx2_avx, sum);
            T tmp[4];
            _mm256_storeu_pd(tmp, decompressed);

            size_t processed = 0;
            unsigned esc_all = static_cast<unsigned>(_mm_movemask_ps(_mm_castsi128_ps(
                                   _mm_cmpeq_epi32(quant_avx_i, _mm_set1_epi32(-32768)))));
            if constexpr (FullOnly) {
                processed = step;
                unsigned esc = esc_all & ((1u << step) - 1);
                if (offset == 1) {
                    _mm256_storeu_pd(data + start, decompressed);
                    while (esc) { int k = __builtin_ctz(esc); data[start + k] = quantizer.recover_unpred2(tid); esc &= esc - 1; }
                } else {
                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) data[(start + j) * offset] = tmp[j];
                    while (esc) { int k = __builtin_ctz(esc); data[(start + k) * offset] = quantizer.recover_unpred2(tid); esc &= esc - 1; }
                }
            } else {
                size_t j = 0;
                #pragma unroll
                for ( ; j < step && start + j < len; ++j)
                    data[(start + j) * offset] = tmp[j];
                processed = j;
                unsigned esc = esc_all & ((1u << processed) - 1);
                while (esc) {
                    int k = __builtin_ctz(esc);
                    data[(start + k) * offset] = quantizer.recover_unpred2(tid);
                    esc &= esc - 1;
                }
            }
            local_quant_index[tid].value += processed;
        }
    }
#elif defined(__ARM_FEATURE_SVE2)
    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool EnableInnerOmp, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_linear_and_quantize_1D_line(
        T *data, const size_t &len, size_t& offset, size_t& cur_ij_offset, int& tid, QuantizeFunc &&quantize_func) {
        if (len <= 1) {
            return;
        }
        auto odd_len = len / 2;
        auto even_len = len - odd_len;
        T *odd_data = data + cur_ij_offset + offset;
        size_t odd_offset = offset << 1;
        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static const size_t step = SVE2_parallelism;
            svbool_t pg = svptrue_b32();
            svbool_t pg64 = svptrue_b64();
            if constexpr (EnableInnerOmp) {
                const size_t vec_chunks = odd_len > step ? (odd_len - 1) / step : 0;
                #pragma omp parallel for
                for (size_t chunk = 0; chunk < vec_chunks; ++chunk) {
                    svbool_t pg = svptrue_b32(); svbool_t pg64 = svptrue_b64();  // private inside the region: a shared sizeless SVE predicate ICEs GCC's OpenMP outliner
                    size_t cur_i = chunk * step;
                    int cur_tid = omp_get_thread_num();
                    T a[step], b[step];
                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        size_t even = (cur_i + j) << 1;
                        a[j] = data[cur_ij_offset + even * offset];
                        b[j] = data[cur_ij_offset + (even + 2) * offset];
                    }
                    svfloat32_t sum = svmul_n_f32_x(pg, svadd_f32_x(pg, svld1(pg, a), svld1(pg, b)), 0.5f);
                    size_t start = cur_i;
                    quantize_float<CompMode, false, SkipOverwrite>(sum, start, odd_data, odd_offset, odd_len, step, pg, pg64, cur_tid);
                }
                i = vec_chunks * step;
            } else {
                for (; i + step < odd_len; i += step) {
                    T a[step], b[step];
                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        size_t even = (i + j) << 1;
                        a[j] = data[cur_ij_offset + even * offset];
                        b[j] = data[cur_ij_offset + (even + 2) * offset];
                    }
                    svfloat32_t sum = svmul_n_f32_x(pg, svadd_f32_x(pg, svld1(pg, a), svld1(pg, b)), 0.5f);
                    size_t start = i;
                    quantize_float<CompMode, false, SkipOverwrite>(sum, start, odd_data, odd_offset, odd_len, step, pg, pg64, tid);
                }
            }
        } else if constexpr (std::is_same_v<T, double>) {
            static const size_t step = SVE2_parallelism;
            svbool_t pg64 = svptrue_b64();
            if constexpr (EnableInnerOmp) {
                const size_t vec_chunks = odd_len > step ? (odd_len - 1) / step : 0;
                #pragma omp parallel for
                for (size_t chunk = 0; chunk < vec_chunks; ++chunk) {
                    svbool_t pg64 = svptrue_b64();  // private inside the region: a shared sizeless SVE predicate ICEs GCC's OpenMP outliner
                    size_t cur_i = chunk * step;
                    int cur_tid = omp_get_thread_num();
                    T a[step], b[step];
                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        size_t even = (cur_i + j) << 1;
                        a[j] = data[cur_ij_offset + even * offset];
                        b[j] = data[cur_ij_offset + (even + 2) * offset];
                    }
                    svfloat64_t sum = svmul_n_f64_x(pg64, svadd_f64_x(pg64, svld1(pg64, a), svld1(pg64, b)), 0.5);
                    size_t start = cur_i;
                    quantize_double<CompMode, false, SkipOverwrite>(sum, start, odd_data, odd_offset, odd_len, step, pg64, cur_tid);
                }
                i = vec_chunks * step;
            } else {
                for (; i + step < odd_len; i += step) {
                    T a[step], b[step];
                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        size_t even = (i + j) << 1;
                        a[j] = data[cur_ij_offset + even * offset];
                        b[j] = data[cur_ij_offset + (even + 2) * offset];
                    }
                    svfloat64_t sum = svmul_n_f64_x(pg64, svadd_f64_x(pg64, svld1(pg64, a), svld1(pg64, b)), 0.5);
                    size_t start = i;
                    quantize_double<CompMode, false, SkipOverwrite>(sum, start, odd_data, odd_offset, odd_len, step, pg64, tid);
                }
            }
        }
        for (; i + 1 < odd_len; ++i) {
            size_t start = ((i << 1) + 1) * offset;
            quantize_func(cur_ij_offset + start, data[cur_ij_offset + start],
                          interp_linear(data[cur_ij_offset + (i << 1) * offset],
                                        data[cur_ij_offset + ((i << 1) + 2) * offset]), tid);
        }
        T pred_edge;
        if (len < 3) {
            pred_edge = data[cur_ij_offset + ((even_len - 1) << 1) * offset];
        } else {
            pred_edge = interp_linear1(data[cur_ij_offset + ((even_len - 2) << 1) * offset],
                                       data[cur_ij_offset + ((even_len - 1) << 1) * offset]);
        }
        int last = 2 * odd_len - 1;
        quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[cur_ij_offset + last * offset], pred_edge, tid);
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool EnableInnerOmp, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_cubic_and_quantize_1D_line(
        T *data, const size_t &len, size_t& offset, size_t& cur_ij_offset, int& tid, QuantizeFunc &&quantize_func) {
        if (len <= 1) {
            return;
        }
        auto odd_len = len / 2;
        auto even_len = len - odd_len;
        T pred_first;
        if (even_len < 2) {
            pred_first = data[cur_ij_offset];
        } else if (even_len < 3) {
            pred_first = interp_linear(data[cur_ij_offset], data[cur_ij_offset + 2 * offset]);
        } else {
            pred_first = interp_quad_1(data[cur_ij_offset], data[cur_ij_offset + 2 * offset],
                                       data[cur_ij_offset + 4 * offset]);
        }
        quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + offset, data[cur_ij_offset + offset], pred_first, tid);

        T *odd_data = data + cur_ij_offset + offset;
        size_t odd_offset = offset << 1;
        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static const size_t step = SVE2_parallelism;
            svbool_t pg = svptrue_b32();
            svbool_t pg64 = svptrue_b64();
            if constexpr (EnableInnerOmp) {
                const size_t vec_chunks = even_len > step + 2 ? (even_len - 3) / step : 0;
                #pragma omp parallel for
                for (size_t chunk = 0; chunk < vec_chunks; ++chunk) {
                    svbool_t pg = svptrue_b32(); svbool_t pg64 = svptrue_b64();  // private inside the region: a shared sizeless SVE predicate ICEs GCC's OpenMP outliner
                    size_t cur_i = chunk * step;
                    int cur_tid = omp_get_thread_num();
                    T a[step], b[step], c[step], d[step];
                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        size_t even = (cur_i + j) << 1;
                        a[j] = data[cur_ij_offset + even * offset];
                        b[j] = data[cur_ij_offset + (even + 2) * offset];
                        c[j] = data[cur_ij_offset + (even + 4) * offset];
                        d[j] = data[cur_ij_offset + (even + 6) * offset];
                    }
                    svfloat32_t sum = svmul_n_f32_x(pg, svadd_f32_x(pg, svld1(pg, b), svld1(pg, c)), 9.0f);
                    sum = svsub_f32_x(pg, sum, svld1(pg, a));
                    sum = svsub_f32_x(pg, sum, svld1(pg, d));
                    sum = svmul_n_f32_x(pg, sum, 0.0625f);
                    size_t start = cur_i + 1;
                    quantize_float<CompMode, false, SkipOverwrite>(sum, start, odd_data, odd_offset, odd_len, step, pg, pg64, cur_tid);
                }
                i = vec_chunks * step;
            } else {
                for (; i + step + 2 < even_len; i += step) {
                    T a[step], b[step], c[step], d[step];
                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        size_t even = (i + j) << 1;
                        a[j] = data[cur_ij_offset + even * offset];
                        b[j] = data[cur_ij_offset + (even + 2) * offset];
                        c[j] = data[cur_ij_offset + (even + 4) * offset];
                        d[j] = data[cur_ij_offset + (even + 6) * offset];
                    }
                    svfloat32_t sum = svmul_n_f32_x(pg, svadd_f32_x(pg, svld1(pg, b), svld1(pg, c)), 9.0f);
                    sum = svsub_f32_x(pg, sum, svld1(pg, a));
                    sum = svsub_f32_x(pg, sum, svld1(pg, d));
                    sum = svmul_n_f32_x(pg, sum, 0.0625f);
                    size_t start = i + 1;
                    quantize_float<CompMode, false, SkipOverwrite>(sum, start, odd_data, odd_offset, odd_len, step, pg, pg64, tid);
                }
            }
        } else if constexpr (std::is_same_v<T, double>) {
            static const size_t step = SVE2_parallelism;
            svbool_t pg64 = svptrue_b64();
            if constexpr (EnableInnerOmp) {
                const size_t vec_chunks = even_len > step + 2 ? (even_len - 3) / step : 0;
                #pragma omp parallel for
                for (size_t chunk = 0; chunk < vec_chunks; ++chunk) {
                    svbool_t pg64 = svptrue_b64();  // private inside the region: a shared sizeless SVE predicate ICEs GCC's OpenMP outliner
                    size_t cur_i = chunk * step;
                    int cur_tid = omp_get_thread_num();
                    T a[step], b[step], c[step], d[step];
                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        size_t even = (cur_i + j) << 1;
                        a[j] = data[cur_ij_offset + even * offset];
                        b[j] = data[cur_ij_offset + (even + 2) * offset];
                        c[j] = data[cur_ij_offset + (even + 4) * offset];
                        d[j] = data[cur_ij_offset + (even + 6) * offset];
                    }
                    svfloat64_t sum = svmul_n_f64_x(pg64, svadd_f64_x(pg64, svld1(pg64, b), svld1(pg64, c)), 9.0);
                    sum = svsub_f64_x(pg64, sum, svld1(pg64, a));
                    sum = svsub_f64_x(pg64, sum, svld1(pg64, d));
                    sum = svmul_n_f64_x(pg64, sum, 0.0625);
                    size_t start = cur_i + 1;
                    quantize_double<CompMode, false, SkipOverwrite>(sum, start, odd_data, odd_offset, odd_len, step, pg64, cur_tid);
                }
                i = vec_chunks * step;
            } else {
                for (; i + step + 2 < even_len; i += step) {
                    T a[step], b[step], c[step], d[step];
                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        size_t even = (i + j) << 1;
                        a[j] = data[cur_ij_offset + even * offset];
                        b[j] = data[cur_ij_offset + (even + 2) * offset];
                        c[j] = data[cur_ij_offset + (even + 4) * offset];
                        d[j] = data[cur_ij_offset + (even + 6) * offset];
                    }
                    svfloat64_t sum = svmul_n_f64_x(pg64, svadd_f64_x(pg64, svld1(pg64, b), svld1(pg64, c)), 9.0);
                    sum = svsub_f64_x(pg64, sum, svld1(pg64, a));
                    sum = svsub_f64_x(pg64, sum, svld1(pg64, d));
                    sum = svmul_n_f64_x(pg64, sum, 0.0625);
                    size_t start = i + 1;
                    quantize_double<CompMode, false, SkipOverwrite>(sum, start, odd_data, odd_offset, odd_len, step, pg64, tid);
                }
            }
        }
        for (; i + 3 < even_len; ++i) {
            size_t start = ((i << 1) + 3) * offset;
            quantize_func(cur_ij_offset + start, data[cur_ij_offset + start],
                          interp_cubic(data[cur_ij_offset + (i << 1) * offset],
                                       data[cur_ij_offset + ((i << 1) + 2) * offset],
                                       data[cur_ij_offset + ((i << 1) + 4) * offset],
                                       data[cur_ij_offset + ((i << 1) + 6) * offset]), tid);
        }
        if (odd_len > 1) {
            if (odd_len < even_len) {
                T edge_pred = interp_quad_2(data[cur_ij_offset + ((even_len - 3) << 1) * offset],
                                            data[cur_ij_offset + ((even_len - 2) << 1) * offset],
                                            data[cur_ij_offset + ((even_len - 1) << 1) * offset]);
                int last = 2 * odd_len - 1;
                quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[cur_ij_offset + last * offset], edge_pred, tid);
            } else {
                if (odd_len > 2) {
                    T edge_pred = interp_quad_2(data[cur_ij_offset + ((even_len - 3) << 1) * offset],
                                                data[cur_ij_offset + ((even_len - 2) << 1) * offset],
                                                data[cur_ij_offset + ((even_len - 1) << 1) * offset]);
                    int last = 2 * odd_len - 3;
                    quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[cur_ij_offset + last * offset], edge_pred, tid);
                }
                T edge_pred = interp_linear1(data[cur_ij_offset + ((even_len - 2) << 1) * offset],
                                             data[cur_ij_offset + ((even_len - 1) << 1) * offset]);
                int last = 2 * odd_len - 1;
                quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[cur_ij_offset + last * offset], edge_pred, tid);
            }
        }
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_linear_and_quantize_1D(const T * buf, const size_t &len, T* data,
        size_t&  offset, size_t& cur_ij_offset, int& tid, QuantizeFunc &&quantize_func) {
        if(len == 1)
            return;

        auto odd_len = len / 2;
        auto even_len = len - odd_len;
        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static const size_t step = SVE2_parallelism;
            svbool_t pg = svptrue_b32();
            svbool_t pg64 = svptrue_b64();

            for (; i + 1  < even_len; i += step) { // 3 is not AVX_256_parallelism - 1 !!
                svfloat32_t va = svld1(pg, &buf[i]);
                svfloat32_t vb = svld1(pg, &buf[i + 1]);

                svfloat32_t sum = svadd_f32_x(pg, va, vb);
                sum = svmul_n_f32_x(pg, sum, 0.5f);
                // quantize
                size_t start = (i << 1) + 1;

                if constexpr (CompMode == COMPMODE::COMP) {
                    T ori[step];
                    size_t base = start * offset;
                    size_t offsetx2 = offset << 1;

                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        ori[j] = data[base + j * offsetx2];
                    }

                    svfloat32_t ori_sve = svld1(pg, ori);
                    svfloat32_t quant_sve = svsub_f32_x(pg, ori_sve, sum); // prediction error
                    T tmp[step];
                    int quant_vals[step];

                    quantize_1D_float(sum, ori_sve, quant_sve, tmp, pg, pg64);

                    svint32_t quant_sve_i = svcvt_s32_f32_z(pg, quant_sve);
                    svst1(pg, quant_vals, quant_sve_i);

                    size_t j = 0;
                    #pragma unroll
                    for ( ; j < step && i + j + 1 < odd_len; ++j) {
                        if (quant_vals[j] != -32768)
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else
                            quantizer.save_unpred2(ori[j], tid);
                    }
                    svst1h_s32(pg, local_quant_inds[tid] + local_quant_index[tid].value, quant_sve_i);
                    local_quant_index[tid].value += j;
                }
                else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
                    svint32_t quant_sve_i = svld1sh_s32(pg, local_quant_inds[tid] + local_quant_index[tid].value);
                    int quant_vals[step];
                    svst1(pg, quant_vals, quant_sve_i);

                    svfloat64_t decompressed_even_f64 = svcvt_f64_s32_x(pg64, quant_sve_i);
                    svfloat64_t decompressed_odd_f64  = svcvtlt_f64_f32_x(pg64, svcvt_f32_s32_x(pg, quant_sve_i));

                    decompressed_even_f64 = svmla_f64_x(pg64, svcvt_f64_f32_x(pg64, sum), decompressed_even_f64, svdup_f64(real_ebx2));
                    decompressed_odd_f64 = svmla_f64_x(pg64, svcvtlt_f64_f32_x(pg64, sum), decompressed_odd_f64, svdup_f64(real_ebx2));

                    svfloat32_t decompressed = svcvt_f32_f64_x(pg64, decompressed_even_f64);
                    decompressed = svcvtnt_f32_f64_x(decompressed, pg64, decompressed_odd_f64);

                    // sum already folded into svmla_f64 above (decode aligned with encode)

                    T tmp[step];
                    svst1_f32(pg, tmp, decompressed);

                    size_t j = 0;
                    for ( ; j < step && i + j + 1 < odd_len; ++j) {
                        if (quant_vals[j] != -32768)
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else
                            data[(start + (j << 1)) * offset] = quantizer.recover_unpred2(tid);
                    }
                    local_quant_index[tid].value  += j;
                }
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static const size_t step = SVE2_parallelism;
            svbool_t pg64 = svptrue_b64();

            for (; i + 1 < even_len; i += step) { // 3 is not AVX_256_parallelism - 1 !!
                svfloat64_t va = svld1(pg64, &buf[i]);
                svfloat64_t vb = svld1(pg64, &buf[i + 1]);

                svfloat64_t sum = svadd_f64_x(pg64, va, vb);
                sum = svmul_n_f64_x(pg64, sum, 0.5);

                size_t start = (i << 1) + 1;
                if constexpr (CompMode == COMPMODE::COMP) {
                    T ori[step];
                    size_t base = start * offset;
                    size_t offsetx2 = offset << 1;

                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        ori[j] = data[base + j * offsetx2];
                    }

                    svfloat64_t ori_sve = svld1(pg64, ori);
                    svfloat64_t quant_sve = svsub_f64_x(pg64, ori_sve, sum); // prediction error
                    T tmp[step];
                    int quant_vals[step];
                    quantize_1D_double(sum, ori_sve, quant_sve, tmp, pg64);

                    svint64_t quant_sve_i = svcvt_s64_f64_x(pg64, quant_sve);

                    svst1w_s64(pg64, quant_vals, quant_sve_i);

                    size_t j = 0;
                    #pragma unroll
                    for ( ; j < step && i + j + 1 < odd_len; ++j) {
                        if (quant_vals[j] != -32768)
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else
                            quantizer.save_unpred2(ori[j], tid);
                    }
                    svst1h_s64(pg64, local_quant_inds[tid] + local_quant_index[tid].value, quant_sve_i);
                    local_quant_index[tid].value += j;
                }
                else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
                    svint64_t quant_sve_i = svld1sh_s64(pg64, local_quant_inds[tid] + local_quant_index[tid].value);
                    int quant_vals[step];
                    svst1w_s64(pg64, quant_vals, quant_sve_i);

                    svfloat64_t decompressed = svmla_f64_x(pg64, sum,
                            svcvt_f64_s64_x(pg64, quant_sve_i), svdup_f64(real_ebx2));
                    T tmp[step];
                    svst1_f64(pg64, tmp, decompressed);
                    size_t j = 0;
                    for ( ; j < step && i + j + 1 < odd_len; ++j) {
                        if (quant_vals[j] != -32768)
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else
                            data[(start + (j << 1)) * offset] = quantizer.recover_unpred2(tid);
                    }
                    local_quant_index[tid].value += j;
                }

            }
        }
        T pred_edge;
        if(len < 3 )
            pred_edge = buf[even_len - 1];
        else
            pred_edge = interp_linear1(buf[even_len - 2], buf[even_len - 1]);
        int last = 2 * odd_len - 1;
        quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset , data[last * offset], pred_edge, tid);
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_cubic_and_quantize_1D(const T * buf, const size_t &len, T* data,
        size_t&  offset, size_t& cur_ij_offset, int& tid,  QuantizeFunc &&quantize_func) {
       // assert(len <= max_dim);
        if(len == 1)
            return;

        auto odd_len = len / 2;
        auto even_len = len - odd_len;

        T pred_first;
        if(even_len < 2)
            pred_first = (buf[0]);
        else if(even_len < 3)
            pred_first = interp_linear(buf[0], buf[1]);
        else
            pred_first = interp_quad_1(buf[0], buf[1], buf[2]);
        quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + offset , data[offset], pred_first, tid);

        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static const size_t step = SVE2_parallelism;
            svbool_t pg = svptrue_b32();
            svbool_t pg64 = svptrue_b64();
            for (; i + 3  < even_len; i += step) { // 3 is not AVX_256_parallelism - 1 !!

                svfloat32_t va = svld1(pg, &buf[i]);
                svfloat32_t vb = svld1(pg, &buf[i + 1]);
                svfloat32_t vc = svld1(pg, &buf[i + 2]);

                svfloat32_t sum = svadd_f32_x(pg, vb, vc);
                sum = svmul_n_f32_x(pg, sum, 9.0f);

                svfloat32_t vd = svld1(pg, &buf[i + 3]);
                sum = svsub_f32_x(pg, sum, va);
                sum = svsub_f32_x(pg, sum, vd);
                sum = svmul_n_f32_x(pg, sum, 0.0625f);

                size_t start = (i << 1) + 3;

                if constexpr (CompMode == COMPMODE::COMP) {
                     T ori[step];
                    size_t base = start * offset;
                    size_t offsetx2 = offset << 1;

                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        ori[j] = data[base + j * offsetx2];
                    }

                    svfloat32_t ori_sve = svld1(pg, ori);
                    svfloat32_t quant_sve = svsub_f32_x(pg, ori_sve, sum); // prediction error
                    T tmp[step];
                    int quant_vals[step];

                    quantize_1D_float (sum, ori_sve, quant_sve, tmp, pg, pg64);

                    svint32_t quant_sve_i = svcvt_s32_f32_z(pg, quant_sve);
                    svst1(pg, quant_vals, quant_sve_i);

                    size_t j = 0;
                    #pragma unroll
                    for ( ; j < step && i + j + 3 < even_len; ++j) {
                        if (quant_vals[j] != -32768)
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else
                            quantizer.save_unpred2(ori[j], tid);
                    }
                    svst1h_s32(pg, local_quant_inds[tid] + local_quant_index[tid].value, quant_sve_i);
                    local_quant_index[tid].value += j;
                }
                else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
                    svint32_t quant_sve_i = svld1sh_s32(pg, local_quant_inds[tid] + local_quant_index[tid].value);
                    int quant_vals[step];
                    svst1(pg, quant_vals, quant_sve_i);

                    svfloat64_t decompressed_even_f64 = svcvt_f64_s32_x(pg64, quant_sve_i);
                    svfloat64_t decompressed_odd_f64  = svcvtlt_f64_f32_x(pg64, svcvt_f32_s32_x(pg, quant_sve_i));

                    decompressed_even_f64 = svmla_f64_x(pg64, svcvt_f64_f32_x(pg64, sum), decompressed_even_f64, svdup_f64(real_ebx2));
                    decompressed_odd_f64 = svmla_f64_x(pg64, svcvtlt_f64_f32_x(pg64, sum), decompressed_odd_f64, svdup_f64(real_ebx2));

                    svfloat32_t decompressed = svcvt_f32_f64_x(pg64, decompressed_even_f64);
                    decompressed = svcvtnt_f32_f64_x(decompressed, pg64, decompressed_odd_f64);

                    // sum already folded into svmla_f64 above (decode aligned with encode)

                    T tmp[step];
                    svst1_f32(pg, tmp, decompressed);

                    size_t j = 0;
                    for ( ; j < step && i + j + 3 < even_len; ++j) {
                        if (quant_vals[j] != -32768)
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else
                            data[(start + (j << 1)) * offset] = quantizer.recover_unpred2(tid);
                    }
                    local_quant_index[tid].value += j;
                }
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static const size_t step = SVE2_parallelism;
            svbool_t pg64 = svptrue_b64();
            for (; i + 3 < even_len; i += step) { // 3 is not AVX_256_parallelism - 1 !!

                svfloat64_t va = svld1(pg64, &buf[i]);
                svfloat64_t vb = svld1(pg64, &buf[i + 1]);
                svfloat64_t vc = svld1(pg64, &buf[i + 2]);

                svfloat64_t sum = svadd_f64_x(pg64, vb, vc);
                sum = svmul_n_f64_x(pg64, sum, 9.0);

                svfloat64_t vd = svld1(pg64, &buf[i + 3]);
                sum = svsub_f64_x(pg64, sum, va);
                sum = svsub_f64_x(pg64, sum, vd);
                sum = svmul_n_f64_x(pg64, sum, 0.0625);

                size_t start = (i << 1) + 3;
                if constexpr (CompMode == COMPMODE::COMP) {
                    T ori[step];
                    size_t base = start * offset;
                    size_t offsetx2 = offset << 1;

                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) {
                        ori[j] = data[base + j * offsetx2];
                    }

                    svfloat64_t ori_sve = svld1(pg64, ori);
                    svfloat64_t quant_sve = svsub_f64_x(pg64, ori_sve, sum); // prediction error
                    T tmp[step];
                    int quant_vals[step];
                    quantize_1D_double(sum, ori_sve, quant_sve, tmp, pg64);

                    svint64_t quant_sve_i = svcvt_s64_f64_x(pg64, quant_sve);

                    svst1w_s64(pg64, quant_vals, quant_sve_i);

                    size_t j = 0;
                    #pragma unroll
                    for ( ; j < step && i + j + 3 < even_len; ++j) {
                        if (quant_vals[j] != -32768)
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else
                            quantizer.save_unpred2(ori[j], tid);
                    }
                    svst1h_s64(pg64, local_quant_inds[tid] + local_quant_index[tid].value, quant_sve_i);
                    local_quant_index[tid].value += j;
                }
                else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
                    svint64_t quant_sve_i = svld1sh_s64(pg64, local_quant_inds[tid] + local_quant_index[tid].value);
                    int quant_vals[step];
                    svst1w_s64(pg64, quant_vals, quant_sve_i);

                    svfloat64_t decompressed = svmla_f64_x(pg64, sum,
                            svcvt_f64_s64_x(pg64, quant_sve_i), svdup_f64(real_ebx2));
                    T tmp[step];
                    svst1_f64(pg64, tmp, decompressed);
                    size_t j = 0;
                    for ( ; j < step && i + j + 3 < even_len; ++j) {
                        if (quant_vals[j] != -32768)
                            data[(start + (j << 1)) * offset] = tmp[j];
                        else
                            data[(start + (j << 1)) * offset] = quantizer.recover_unpred2(tid);
                    }
                    local_quant_index[tid].value += j;
                }

            }
        }

        if(odd_len > 1){
            if(odd_len < even_len){//the only boundary is p[len- 1]
                //odd_len < even_len so even_len > 2
                T edge_pred;
                edge_pred = interp_quad_2(buf[even_len - 3], buf[even_len - 2], buf[even_len - 1]);
                int last = 2 * odd_len - 1;
                quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[last * offset], edge_pred, tid);

            }
            else{//the boundary points are is p[len -2 ] and p[len -1 ]
                T edge_pred;
                if(odd_len > 2){ //len - 2
                 //odd_len = even_len so even_len > 2
                    edge_pred = interp_quad_2(buf[even_len - 3],  buf[even_len - 2], buf[even_len - 1]);
                    int last = 2 * odd_len - 3;
                    quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[last * offset], edge_pred, tid);
                }
                //len -1
                //odd_len = even_len so even_len > 1
                    edge_pred = interp_linear1(buf[even_len - 2], buf[even_len - 1]);
                    int last = 2 * odd_len - 1;
                    quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[last * offset], edge_pred, tid);


            }
        }
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_linear_and_quantize(const T * a, const T* b, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func) {

        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static const size_t step = SVE2_parallelism;
            svbool_t pg = svptrue_b32();
            svbool_t pg64 = svptrue_b64();
            for (; i + step <= len; i += step) {
                svfloat32_t va = svld1(pg, &a[i]);
                svfloat32_t vb = svld1(pg, &b[i]);

                svfloat32_t sum = svadd_f32_x(pg, va, vb);
                sum = svmul_n_f32_x(pg, sum, 0.5f);

                quantize_float<CompMode, true, SkipOverwrite>(sum, i, data, offset, len, step, pg, pg64, tid);
            }
            if (i < len) {
                svfloat32_t va = svld1(pg, &a[i]);
                svfloat32_t vb = svld1(pg, &b[i]);

                svfloat32_t sum = svadd_f32_x(pg, va, vb);
                sum = svmul_n_f32_x(pg, sum, 0.5f);

                quantize_float<CompMode, false, SkipOverwrite>(sum, i, data, offset, len, step, pg, pg64, tid);
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static const size_t step = SVE2_parallelism;
            svbool_t pg64 = svptrue_b64();

            for (; i + step <= len; i += step) {
                svfloat64_t va = svld1(pg64, &a[i]);
                svfloat64_t vb = svld1(pg64, &b[i]);

                svfloat64_t sum = svadd_f64_x(pg64, va, vb);
                sum = svmul_n_f64_x(pg64, sum, 0.5);
                quantize_double<CompMode, true, SkipOverwrite>(sum, i, data, offset, len, step, pg64, tid);
            }
            if (i < len) {
                svfloat64_t va = svld1(pg64, &a[i]);
                svfloat64_t vb = svld1(pg64, &b[i]);

                svfloat64_t sum = svadd_f64_x(pg64, va, vb);
                sum = svmul_n_f64_x(pg64, sum, 0.5);
                quantize_double<CompMode, false, SkipOverwrite>(sum, i, data, offset, len, step, pg64, tid);
            }
        }
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_cubic_and_quantize(const T * a, const T* b, T* c, T*d, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func) {

        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static const size_t step = SVE2_parallelism;
            svbool_t pg = svptrue_b32();
            svbool_t pg64 = svptrue_b64();

            for (; i + step <= len; i += step) {
                svfloat32_t va = svld1(pg, &a[i]);
                svfloat32_t vb = svld1(pg, &b[i]);
                svfloat32_t vc = svld1(pg, &c[i]);

                svfloat32_t sum = svadd_f32_x(pg, vb, vc);
                sum = svmul_n_f32_x(pg, sum, 9.0f);

                svfloat32_t vd = svld1(pg, &d[i]);

                sum = svsub_f32_x(pg, sum, va);
                sum = svsub_f32_x(pg, sum, vd);
                sum = svmul_n_f32_x(pg, sum, 0.0625f);

                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode, true, SkipOverwrite>(sum, i, data, offset, len, step, pg, pg64, tid);
            }
            if (i < len) {
                svfloat32_t va = svld1(pg, &a[i]);
                svfloat32_t vb = svld1(pg, &b[i]);
                svfloat32_t vc = svld1(pg, &c[i]);

                svfloat32_t sum = svadd_f32_x(pg, vb, vc);
                sum = svmul_n_f32_x(pg, sum, 9.0f);

                svfloat32_t vd = svld1(pg, &d[i]);

                sum = svsub_f32_x(pg, sum, va);
                sum = svsub_f32_x(pg, sum, vd);
                sum = svmul_n_f32_x(pg, sum, 0.0625f);

                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode, false, SkipOverwrite>(sum, i, data, offset, len, step, pg, pg64, tid);
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static const size_t step = SVE2_parallelism;
            svbool_t pg64 = svptrue_b64();

            for (; i + step <= len; i += step) {
                svfloat64_t va = svld1(pg64, &a[i]);
                svfloat64_t vb = svld1(pg64, &b[i]);
                svfloat64_t vc = svld1(pg64, &c[i]);

                svfloat64_t sum = svadd_f64_x(pg64, vb, vc);
                sum = svmul_n_f64_x(pg64, sum, 9.0);

                svfloat64_t vd = svld1(pg64, &d[i]);

                sum = svsub_f64_x(pg64, sum, va);
                sum = svsub_f64_x(pg64, sum, vd);
                sum = svmul_n_f64_x(pg64, sum, 0.0625);

                quantize_double<CompMode, true, SkipOverwrite>(sum, i, data, offset, len, step, pg64, tid);
            }
            if (i < len) {
                svfloat64_t va = svld1(pg64, &a[i]);
                svfloat64_t vb = svld1(pg64, &b[i]);
                svfloat64_t vc = svld1(pg64, &c[i]);

                svfloat64_t sum = svadd_f64_x(pg64, vb, vc);
                sum = svmul_n_f64_x(pg64, sum, 9.0);

                svfloat64_t vd = svld1(pg64, &d[i]);

                sum = svsub_f64_x(pg64, sum, va);
                sum = svsub_f64_x(pg64, sum, vd);
                sum = svmul_n_f64_x(pg64, sum, 0.0625);

                quantize_double<CompMode, false, SkipOverwrite>(sum, i, data, offset, len, step, pg64, tid);
            }
        }
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_equal_and_quantize(const T * a, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func) {
        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static const size_t step = SVE2_parallelism;
            svbool_t pg = svptrue_b32();
            svbool_t pg64 = svptrue_b64();

            for (; i + step <= len; i += step) {
                svfloat32_t sum = svld1(pg, &a[i]);
                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode, true, SkipOverwrite>(sum, i, data, offset, len, step, pg, pg64, tid);
            }
            if (i < len) {
                svfloat32_t sum = svld1(pg, &a[i]);
                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode, false, SkipOverwrite>(sum, i, data, offset, len, step, pg, pg64, tid);
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static const size_t step = SVE2_parallelism;
            svbool_t pg64 = svptrue_b64();
            for (; i + step <= len; i += step) {
                svfloat64_t sum = svld1(pg64, &a[i]);
                quantize_double<CompMode, true, SkipOverwrite>(sum, i, data, offset, len, step, pg64, tid);
            }
            if (i < len) {
                svfloat64_t sum = svld1(pg64, &a[i]);
                quantize_double<CompMode, false, SkipOverwrite>(sum, i, data, offset, len, step, pg64, tid);
            }
        }
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_linear1_and_quantize(const T * a, const T* b, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func) {

        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static const size_t step = SVE2_parallelism;
            svbool_t pg = svptrue_b32();
            svbool_t pg64 = svptrue_b64();
            for (; i + step <= len; i += step) {
                svfloat32_t va = svld1(pg, &a[i]);
                svfloat32_t vb = svld1(pg, &b[i]);
                vb = svmul_n_f32_x(pg, vb, 1.5f);
                svfloat32_t sum = svmls_n_f32_x(pg, vb, va, 0.5f);
                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode, true, SkipOverwrite>(sum, i, data, offset, len, step, pg, pg64, tid);
            }
            if (i < len) {
                svfloat32_t va = svld1(pg, &a[i]);
                svfloat32_t vb = svld1(pg, &b[i]);
                vb = svmul_n_f32_x(pg, vb, 1.5f);
                svfloat32_t sum = svmls_n_f32_x(pg, vb, va, 0.5f);
                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode, false, SkipOverwrite>(sum, i, data, offset, len, step, pg, pg64, tid);
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static const size_t step = SVE2_parallelism;
            svbool_t pg64 = svptrue_b64();

            for (; i + step <= len; i += step) {
                svfloat64_t va = svld1(pg64, &a[i]);
                svfloat64_t vb = svld1(pg64, &b[i]);
                vb = svmul_n_f64_x(pg64, vb, 1.5);
                svfloat64_t sum = svmls_n_f64_x(pg64, vb, va, 0.5);
                quantize_double<CompMode, true, SkipOverwrite>(sum, i, data, offset, len, step, pg64, tid);
            }
            if (i < len) {
                svfloat64_t va = svld1(pg64, &a[i]);
                svfloat64_t vb = svld1(pg64, &b[i]);
                vb = svmul_n_f64_x(pg64, vb, 1.5);
                svfloat64_t sum = svmls_n_f64_x(pg64, vb, va, 0.5);
                quantize_double<CompMode, false, SkipOverwrite>(sum, i, data, offset, len, step, pg64, tid);
            }
        }
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_quad1_and_quantize(const T * a, const T* b, const T* c, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func) {

        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static const size_t step = SVE2_parallelism;
            svbool_t pg = svptrue_b32();
            svbool_t pg64 = svptrue_b64();

            for (; i + step <= len; i += step) {
                svfloat32_t vb = svld1(pg, &b[i]);
                svfloat32_t vc = svld1(pg, &c[i]);
                vb = svnmls_n_f32_x(pg, vc, vb, 6.0f);
                svfloat32_t va = svld1(pg, &a[i]);
                svfloat32_t sum = svmla_n_f32_x(pg, vb, va, 3.0f);
                sum = svmul_n_f32_x(pg, sum, 0.125f);
                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode, true, SkipOverwrite>(sum, i, data, offset, len, step, pg, pg64, tid);
            }
            if (i < len) {
                svfloat32_t vb = svld1(pg, &b[i]);
                svfloat32_t vc = svld1(pg, &c[i]);
                vb = svnmls_n_f32_x(pg, vc, vb, 6.0f);
                svfloat32_t va = svld1(pg, &a[i]);
                svfloat32_t sum = svmla_n_f32_x(pg, vb, va, 3.0f);
                sum = svmul_n_f32_x(pg, sum, 0.125f);
                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode, false, SkipOverwrite>(sum, i, data, offset, len, step, pg, pg64, tid);
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static const size_t step = SVE2_parallelism;
            svbool_t pg64 = svptrue_b64();

            for (; i + step <= len; i += step) {
                svfloat64_t vb = svld1(pg64, &b[i]);
                svfloat64_t vc = svld1(pg64, &c[i]);
                vb = svnmls_n_f64_x(pg64, vc, vb, 6.0);
                svfloat64_t va = svld1(pg64, &a[i]);
                svfloat64_t sum = svmla_n_f64_x(pg64, vb, va, 3.0);
                sum = svmul_n_f64_x(pg64, sum, 0.125);
                quantize_double<CompMode, true, SkipOverwrite>(sum, i, data, offset, len, step, pg64, tid);
            }
            if (i < len) {
                svfloat64_t vb = svld1(pg64, &b[i]);
                svfloat64_t vc = svld1(pg64, &c[i]);
                vb = svnmls_n_f64_x(pg64, vc, vb, 6.0);
                svfloat64_t va = svld1(pg64, &a[i]);
                svfloat64_t sum = svmla_n_f64_x(pg64, vb, va, 3.0);
                sum = svmul_n_f64_x(pg64, sum, 0.125);
                quantize_double<CompMode, false, SkipOverwrite>(sum, i, data, offset, len, step, pg64, tid);
            }

        }
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_quad2_and_quantize (const T * a, const T* b, const T* c, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func) {
        size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            static const size_t step = SVE2_parallelism;
            svbool_t pg = svptrue_b32();
            svbool_t pg64 = svptrue_b64();

            for (; i + step <= len; i += step) {
                svfloat32_t va = svld1(pg, &a[i]);
                svfloat32_t vb = svld1(pg, &b[i]);
                vb = svnmls_n_f32_x(pg, va, vb, 6.0f);

                svfloat32_t vc = svld1(pg, &c[i]);
                svfloat32_t sum = svmla_n_f32_x(pg, vb, vc, 3.0f);
                sum = svmul_n_f32_x(pg, sum, 0.125f);
                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode, true, SkipOverwrite>(sum, i, data, offset, len, step, pg, pg64, tid);
            }
            if (i < len) {
                svfloat32_t va = svld1(pg, &a[i]);
                svfloat32_t vb = svld1(pg, &b[i]);
                vb = svnmls_n_f32_x(pg, va, vb, 6.0f);

                svfloat32_t vc = svld1(pg, &c[i]);
                svfloat32_t sum = svmla_n_f32_x(pg, vb, vc, 3.0f);
                sum = svmul_n_f32_x(pg, sum, 0.125f);
                // _mm256_storeu_ps(p + i, sum);
                quantize_float<CompMode, false, SkipOverwrite>(sum, i, data, offset, len, step, pg, pg64, tid);
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            static const size_t step = SVE2_parallelism;
            svbool_t pg64 = svptrue_b64();

            for (; i + step <= len; i += step) {
                svfloat64_t va = svld1(pg64, &a[i]);
                svfloat64_t vb = svld1(pg64, &b[i]);
                vb = svnmls_n_f64_x(pg64, va, vb, 6.0);

                svfloat64_t vc = svld1(pg64, &c[i]);
                svfloat64_t sum = svmla_n_f64_x(pg64, vb, vc, 3.0);
                sum = svmul_n_f64_x(pg64, sum, 0.125);
                // _mm256_storeu_ps(p + i, sum);
                quantize_double<CompMode, true, SkipOverwrite>(sum, i, data, offset, len, step, pg64, tid);
            }
            if (i < len) {
                svfloat64_t va = svld1(pg64, &a[i]);
                svfloat64_t vb = svld1(pg64, &b[i]);
                vb = svnmls_n_f64_x(pg64, va, vb, 6.0);

                svfloat64_t vc = svld1(pg64, &c[i]);
                svfloat64_t sum = svmla_n_f64_x(pg64, vb, vc, 3.0);
                sum = svmul_n_f64_x(pg64, sum, 0.125);
                // _mm256_storeu_ps(p + i, sum);
                quantize_double<CompMode, false, SkipOverwrite>(sum, i, data, offset, len, step, pg64, tid);
            }
        }

    }

    template <class T, uint N, class QuantizerOMP>
    template<typename U, typename>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::quantize_1D_float (
        svfloat32_t& sum, svfloat32_t& ori_sve, svfloat32_t& quant_sve, T* tmp, svbool_t& pg, svbool_t& pg64) {

            svfloat64_t quant_even_f64 = svcvt_f64_f32_x(pg64, quant_sve);
            svfloat64_t quant_odd_f64  = svcvtlt_f64_f32_x(pg64, quant_sve);
            quant_even_f64 = svrintn_f64_x(pg64, svmul_n_f64_x(pg64, quant_even_f64, real_ebx2_r));
            quant_odd_f64  = svrintn_f64_x(pg64, svmul_n_f64_x(pg64, quant_odd_f64, real_ebx2_r));

            svbool_t pg_gt_neg = svcmpgt_n_f64(pg64, quant_even_f64, -radius); // val > -radius
            svbool_t pg_lt_pos = svcmplt_n_f64(pg64, quant_even_f64,  radius); // val < +radius
            svbool_t pg_in_range = svand_b_z(pg64, pg_gt_neg, pg_lt_pos);
            quant_even_f64 = svsel_f64(pg_in_range, quant_even_f64, svdup_n_f64(0.0));

            svbool_t pg_gt_neg_o = svcmpgt_n_f64(pg64, quant_odd_f64, -radius);
            svbool_t pg_lt_pos_o = svcmplt_n_f64(pg64, quant_odd_f64,  radius);
            svbool_t pg_in_range_o = svand_b_z(pg64, pg_gt_neg_o, pg_lt_pos_o);
            quant_odd_f64 = svsel_f64(pg_in_range_o, quant_odd_f64, svdup_n_f64(0.0));

            // dequantization for decompression

            svfloat64_t decompressed_even_f64 = svmla_f64_x(pg64, svcvt_f64_f32_x(pg64, sum),
                    quant_even_f64, svdup_f64(real_ebx2));
            svfloat64_t decompressed_odd_f64  = svmla_f64_x(pg64, svcvtlt_f64_f32_x(pg64, sum),
                    quant_odd_f64, svdup_f64(real_ebx2));

            // svfloat32_t even_f32 = svcvt_f32_f64_x(pg64, decompressed_even_f64);
            // svfloat32_t odd_f32  = svcvtlt_f32_f64_x(pg64, decompressed_odd_f64);
            svfloat32_t decompressed = svcvt_f32_f64_x(pg64, decompressed_even_f64);
            decompressed = svcvtnt_f32_f64_x(decompressed, pg64, decompressed_odd_f64);

            svst1_f32(pg, tmp, decompressed);

            // even_f32 = svcvt_f32_f64_x(pg64, quant_even_f64);
            // odd_f32  = svcvt_f32_f64_x(pg64, quant_odd_f64);
            // quant_sve = svzip1_f32(svuzp1_f32(even_f32, even_f32), svuzp1_f32(odd_f32, odd_f32));

            quant_sve = svcvt_f32_f64_x(pg64, quant_even_f64);
            quant_sve = svcvtnt_f32_f64_x(quant_sve, pg64, quant_odd_f64);

            svfloat32_t err_dequan = svsub_f32_x(pg, decompressed, ori_sve);

            pg_in_range = svand_b_z(pg, svcmpge_n_f32(pg, err_dequan, -real_eb), svcmple_n_f32(pg, err_dequan, real_eb));
            quant_sve = svsel_f32(pg_in_range, quant_sve, svdup_n_f32(-static_cast<float>(radius)));
    }

    template <class T, uint N, class QuantizerOMP>
    template<typename U, typename>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::quantize_1D_double (
        svfloat64_t& sum, svfloat64_t& ori_sve, svfloat64_t& quant_sve, T* tmp, svbool_t& pg64) {

        quant_sve = svrintn_f64_x(pg64, svmul_n_f64_x(pg64, quant_sve, real_ebx2_r));

        svbool_t pg_gt_neg = svcmpgt_n_f64(pg64, quant_sve, -radius);
        svbool_t pg_lt_pos = svcmplt_n_f64(pg64, quant_sve,  radius);
        svbool_t pg_in_range = svand_b_z(pg64, pg_gt_neg, pg_lt_pos);
        quant_sve = svsel_f64(pg_in_range, quant_sve, svdup_n_f64(0.0));

        svfloat64_t decompressed = svmla_f64_x(pg64, sum, quant_sve, svdup_f64(real_ebx2));
        svst1_f64(pg64, tmp, decompressed);
        svfloat64_t err_dequan = svsub_f64_x(pg64, decompressed, ori_sve);

        pg_in_range = svand_b_z(pg64, svcmpge_n_f64(pg64, err_dequan, -real_eb), svcmple_n_f64(pg64, err_dequan, real_eb));
        quant_sve = svsel_f64(pg_in_range, quant_sve, svdup_n_f64(-static_cast<double>(radius)));
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool FullOnly, bool SkipOverwrite, typename U, typename>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::quantize_float (svfloat32_t& sum, size_t& start, T*& data, size_t& offset,
        size_t& len, const size_t& step, svbool_t& pg, svbool_t& pg64, int tid) {
        if constexpr (CompMode == COMPMODE::COMP) {
            T ori[step];
            size_t base = start * offset;

            svfloat32_t ori_sve;
            if constexpr (FullOnly) {
                if (offset == 1) {
                    ori_sve = svld1(pg, data + start);   // contiguous: SIMD load straight (ori[] filled lazily on escape)
                } else {
                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) ori[j] = data[base + j * offset];
                    ori_sve = svld1(pg, ori);
                }
            } else {
                // tail: clamp past-end indices to the last valid element (avoid OOB read)
                for (size_t j = 0; j < step; ++j) {
                    size_t idx = (start + j < len) ? (start + j) : (len - 1);
                    ori[j] = data[idx * offset];
                }
                ori_sve = svld1(pg, ori);
            }
            svfloat32_t quant_sve = svsub_f32_x(pg, ori_sve, sum); // prediction error

            T tmp[step];
            int quant_vals[step];

            // calculate quantization code
            svfloat64_t quant_even_f64 = svcvt_f64_f32_x(pg64, quant_sve);
            svfloat64_t quant_odd_f64  = svcvtlt_f64_f32_x(pg64, quant_sve);
            quant_even_f64 = svrintn_f64_x(pg64, svmul_n_f64_x(pg64, quant_even_f64, real_ebx2_r));
            quant_odd_f64  = svrintn_f64_x(pg64, svmul_n_f64_x(pg64, quant_odd_f64, real_ebx2_r));

            svbool_t pg_gt_neg = svcmpgt_n_f64(pg64, quant_even_f64, -radius); // val > -radius
            svbool_t pg_lt_pos = svcmplt_n_f64(pg64, quant_even_f64,  radius); // val < +radius
            svbool_t pg_in_range = svand_b_z(pg64, pg_gt_neg, pg_lt_pos);
            quant_even_f64 = svsel_f64(pg_in_range, quant_even_f64, svdup_n_f64(0.0));

            svbool_t pg_gt_neg_o = svcmpgt_n_f64(pg64, quant_odd_f64, -radius);
            svbool_t pg_lt_pos_o = svcmplt_n_f64(pg64, quant_odd_f64,  radius);
            svbool_t pg_in_range_o = svand_b_z(pg64, pg_gt_neg_o, pg_lt_pos_o);
            quant_odd_f64 = svsel_f64(pg_in_range_o, quant_odd_f64, svdup_n_f64(0.0));

            // dequantization for decompression
            svfloat64_t decompressed_even_f64 = svmla_f64_x(pg64, svcvt_f64_f32_x(pg64, sum),
                    quant_even_f64, svdup_f64(real_ebx2));
            svfloat64_t decompressed_odd_f64  = svmla_f64_x(pg64, svcvtlt_f64_f32_x(pg64, sum),
                    quant_odd_f64, svdup_f64(real_ebx2));

            svfloat32_t decompressed = svcvt_f32_f64_x(pg64, decompressed_even_f64);
            decompressed = svcvtnt_f32_f64_x(decompressed, pg64, decompressed_odd_f64);

            svst1_f32(pg, tmp, decompressed);

            quant_sve = svcvt_f32_f64_x(pg64, quant_even_f64);
            quant_sve = svcvtnt_f32_f64_x(quant_sve, pg64, quant_odd_f64);

            svfloat32_t err_dequan = svsub_f32_x(pg, decompressed, ori_sve);

            pg_in_range = svand_b_z(pg, svcmpge_n_f32(pg, err_dequan, -real_eb), svcmple_n_f32(pg, err_dequan, real_eb));
            quant_sve = svsel_f32(pg_in_range, quant_sve, svdup_n_f32(-static_cast<float>(radius)));

            svint32_t quant_sve_i = svcvt_s32_f32_z(pg, quant_sve);
            svst1(pg, quant_vals, quant_sve_i);
            size_t j;
            if constexpr (FullOnly) {
                if (offset == 1) {
                    svbool_t esc_pred = svcmpeq_n_s32(pg, quant_sve_i, -32768);
                    if constexpr (!SkipOverwrite) {
                        svst1_f32(pg, data + start, svsel_f32(esc_pred, ori_sve, decompressed));
                    }
                    if (svptest_any(pg, esc_pred)) {
                        svst1_f32(pg, ori, ori_sve);   // contiguous read left ori[] unfilled
                        #pragma unroll
                        for (size_t k = 0; k < step; ++k)
                            if (quant_vals[k] == -32768) quantizer.save_unpred2(ori[k], tid);
                    }
                } else {
                    #pragma unroll
                    for (size_t k = 0; k < step; ++k) {
                        if (quant_vals[k] != -32768) {
                            if constexpr (!SkipOverwrite) data[(start + k) * offset] = tmp[k];
                        }
                        else quantizer.save_unpred2(ori[k], tid);
                    }
                }
                j = step;
            } else {
                j = 0;
                #pragma unroll
                for ( ; j < step && start + j < len; ++j) {
                    if (quant_vals[j] != -32768) {
                        if constexpr (!SkipOverwrite) data[(start + j) * offset] = tmp[j];
                    }
                    else quantizer.save_unpred2(ori[j], tid);
                }
            }

            svst1h_s32(pg, local_quant_inds[tid] + local_quant_index[tid].value, quant_sve_i);
            local_quant_index[tid].value += j;
        }
        else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
            svint32_t quant_sve_i = svld1sh_s32(pg, local_quant_inds[tid] + local_quant_index[tid].value);
            int quant_vals[step];
            svst1(pg, quant_vals, quant_sve_i);

            svfloat64_t decompressed_even_f64 = svcvt_f64_s32_x(pg64, quant_sve_i);
            svfloat64_t decompressed_odd_f64  = svcvtlt_f64_f32_x(pg64, svcvt_f32_s32_x(pg, quant_sve_i));

            decompressed_even_f64 = svmla_f64_x(pg64, svcvt_f64_f32_x(pg64, sum), decompressed_even_f64, svdup_f64(real_ebx2));
            decompressed_odd_f64 = svmla_f64_x(pg64, svcvtlt_f64_f32_x(pg64, sum), decompressed_odd_f64, svdup_f64(real_ebx2));

            svfloat32_t decompressed = svcvt_f32_f64_x(pg64, decompressed_even_f64);
            decompressed = svcvtnt_f32_f64_x(decompressed, pg64, decompressed_odd_f64);

            // sum already folded into svmla_f64 above (decode aligned with encode)

            T tmp[step];
            svst1_f32(pg, tmp, decompressed);

            size_t j;
            if constexpr (FullOnly) {
                if (offset == 1) {
                    svst1_f32(pg, data + start, decompressed);
                    svbool_t esc_pred = svcmpeq_n_s32(pg, quant_sve_i, -32768);
                    if (svptest_any(pg, esc_pred)) {
                        #pragma unroll
                        for (size_t k = 0; k < step; ++k)
                            if (quant_vals[k] == -32768) data[start + k] = quantizer.recover_unpred2(tid);
                    }
                } else {
                    #pragma unroll
                    for (size_t k = 0; k < step; ++k) {
                        if (quant_vals[k] != -32768) data[(start + k) * offset] = tmp[k];
                        else data[(start + k) * offset] = quantizer.recover_unpred2(tid);
                    }
                }
                j = step;
            } else {
                j = 0;
                #pragma unroll
                for ( ; j < step && start + j < len; ++j) {
                    if (quant_vals[j] != -32768) data[(start + j) * offset] = tmp[j];
                    else data[(start + j) * offset] = quantizer.recover_unpred2(tid);
                }
            }
            local_quant_index[tid].value += j;
        }
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool FullOnly, bool SkipOverwrite, typename U, typename>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::quantize_double (svfloat64_t& sum, size_t& start, T*& data, size_t& offset,
        size_t& len, const size_t& step, svbool_t& pg64, int tid) {
        if constexpr (CompMode == COMPMODE::COMP) {
            T ori[step];
            size_t base = start * offset;

            svfloat64_t ori_sve;
            if constexpr (FullOnly) {
                if (offset == 1) {
                    ori_sve = svld1(pg64, data + start);   // contiguous: SIMD load straight (ori[] filled lazily on escape)
                } else {
                    #pragma unroll
                    for (size_t j = 0; j < step; ++j) ori[j] = data[base + j * offset];
                    ori_sve = svld1(pg64, ori);
                }
            } else {
                // tail: clamp past-end indices to the last valid element (avoid OOB read)
                for (size_t j = 0; j < step; ++j) {
                    size_t idx = (start + j < len) ? (start + j) : (len - 1);
                    ori[j] = data[idx * offset];
                }
                ori_sve = svld1(pg64, ori);
            }
            svfloat64_t quant_sve = svsub_f64_x(pg64, ori_sve, sum); // prediction error
            T tmp[step];
            int quant_vals[step];

            quant_sve = svrintn_f64_x(pg64, svmul_n_f64_x(pg64, quant_sve, real_ebx2_r));

            svbool_t pg_gt_neg = svcmpgt_n_f64(pg64, quant_sve, -radius);
            svbool_t pg_lt_pos = svcmplt_n_f64(pg64, quant_sve,  radius);
            svbool_t pg_in_range = svand_b_z(pg64, pg_gt_neg, pg_lt_pos);
            quant_sve = svsel_f64(pg_in_range, quant_sve, svdup_n_f64(0.0));

            svfloat64_t decompressed = svmla_f64_x(pg64, sum, quant_sve, svdup_f64(real_ebx2));
            svst1_f64(pg64, tmp, decompressed);
            svfloat64_t err_dequan = svsub_f64_x(pg64, decompressed, ori_sve);

            pg_in_range = svand_b_z(pg64, svcmpge_n_f64(pg64, err_dequan, -real_eb), svcmple_n_f64(pg64, err_dequan, real_eb));
            quant_sve = svsel_f64(pg_in_range, quant_sve, svdup_n_f64(-static_cast<double>(radius)));


            svint64_t quant_sve_i = svcvt_s64_f64_x(pg64, quant_sve);

            svst1w_s64(pg64, quant_vals, quant_sve_i);

            size_t j;
            if constexpr (FullOnly) {
                if (offset == 1) {
                    svbool_t esc_pred = svcmpeq_n_s64(pg64, quant_sve_i, -32768);
                    if constexpr (!SkipOverwrite) {
                        svst1_f64(pg64, data + start, svsel_f64(esc_pred, ori_sve, decompressed));
                    }
                    if (svptest_any(pg64, esc_pred)) {
                        svst1_f64(pg64, ori, ori_sve);   // contiguous read left ori[] unfilled
                        #pragma unroll
                        for (size_t k = 0; k < step; ++k)
                            if (quant_vals[k] == -32768) quantizer.save_unpred2(ori[k], tid);
                    }
                } else {
                    #pragma unroll
                    for (size_t k = 0; k < step; ++k) {
                        if (quant_vals[k] != -32768) {
                            if constexpr (!SkipOverwrite) data[(start + k) * offset] = tmp[k];
                        }
                        else quantizer.save_unpred2(ori[k], tid);
                    }
                }
                j = step;
            } else {
                j = 0;
                #pragma unroll
                for ( ; j < step && start + j < len; ++j) {
                    if (quant_vals[j] != -32768) {
                        if constexpr (!SkipOverwrite) data[(start + j) * offset] = tmp[j];
                    }
                    else quantizer.save_unpred2(ori[j], tid);
                }
            }
            svst1h_s64(pg64, local_quant_inds[tid] + local_quant_index[tid].value, quant_sve_i);
            local_quant_index[tid].value += j;
        }
        else if constexpr (CompMode == COMPMODE::DECOMP) {
            svint64_t quant_sve_i = svld1sh_s64(pg64, local_quant_inds[tid] + local_quant_index[tid].value);
            int quant_vals[step];
            svst1w_s64(pg64, quant_vals, quant_sve_i);
            // codes are stored signed (no +radius offset), so use them directly — matches
            // the COMP store and the quantize_float path. (Removed a leftover -radius shift.)

            svfloat64_t decompressed = svmla_f64_x(pg64, sum,
                    svcvt_f64_s64_x(pg64, quant_sve_i), svdup_f64(real_ebx2));
            T tmp[step];
            svst1_f64(pg64, tmp, decompressed);
            size_t j;
            if constexpr (FullOnly) {
                if (offset == 1) {
                    svst1_f64(pg64, data + start, decompressed);
                    svbool_t esc_pred = svcmpeq_n_s64(pg64, quant_sve_i, -32768);
                    if (svptest_any(pg64, esc_pred)) {
                        #pragma unroll
                        for (size_t k = 0; k < step; ++k)
                            if (quant_vals[k] == -32768) data[start + k] = quantizer.recover_unpred2(tid);
                    }
                } else {
                    #pragma unroll
                    for (size_t k = 0; k < step; ++k) {
                        if (quant_vals[k] != -32768) data[(start + k) * offset] = tmp[k];
                        else data[(start + k) * offset] = quantizer.recover_unpred2(tid);
                    }
                }
                j = step;
            } else {
                j = 0;
                #pragma unroll
                for ( ; j < step && start + j < len; ++j) {
                    if (quant_vals[j] != -32768) data[(start + j) * offset] = tmp[j];
                    else data[(start + j) * offset] = quantizer.recover_unpred2(tid);
                }
            }
            local_quant_index[tid].value += j;
        }
    }

#else
    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool EnableInnerOmp, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_linear_and_quantize_1D_line(
        T *data, const size_t &len, size_t& offset, size_t& cur_ij_offset, int& tid, QuantizeFunc &&quantize_func) {
        if (len <= 1) {
            return;
        }
        auto odd_len = len / 2;
        auto even_len = len - odd_len;
        size_t i = 0;
        if constexpr (EnableInnerOmp) {
            if (!legacy_stream) {  // same chunk-to-thread layout as the AVX lanes, so the streams interoperate
                const size_t step = AVX_256_parallelism;
                const size_t vec_chunks = odd_len > step ? (odd_len - 1) / step : 0;
                #pragma omp parallel for
                for (size_t chunk = 0; chunk < vec_chunks; ++chunk) {
                    int cur_tid = omp_get_thread_num();
                    for (size_t j = chunk * step; j < (chunk + 1) * step; ++j) {
                        size_t start = ((j << 1) + 1) * offset;
                        quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + start, data[cur_ij_offset + start],
                                      interp_linear(data[cur_ij_offset + (j << 1) * offset],
                                                    data[cur_ij_offset + ((j << 1) + 2) * offset]), cur_tid);
                    }
                }
                i = vec_chunks * step;
            }
        }
        for (; i + 1 < odd_len; ++i) {
            size_t start = ((i << 1) + 1) * offset;
            quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + start, data[cur_ij_offset + start],
                          interp_linear(data[cur_ij_offset + (i << 1) * offset],
                                        data[cur_ij_offset + ((i << 1) + 2) * offset]), tid);
        }
        T pred_edge;
        if (len < 3) {
            pred_edge = data[cur_ij_offset + ((even_len - 1) << 1) * offset];
        } else {
            pred_edge = interp_linear1(data[cur_ij_offset + ((even_len - 2) << 1) * offset],
                                       data[cur_ij_offset + ((even_len - 1) << 1) * offset]);
        }
        int last = 2 * odd_len - 1;
        quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[cur_ij_offset + last * offset], pred_edge, tid);
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool EnableInnerOmp, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_cubic_and_quantize_1D_line(
        T *data, const size_t &len, size_t& offset, size_t& cur_ij_offset, int& tid, QuantizeFunc &&quantize_func) {
        if (len <= 1) {
            return;
        }
        auto odd_len = len / 2;
        auto even_len = len - odd_len;
        T pred_first;
        if (even_len < 2) {
            pred_first = data[cur_ij_offset];
        } else if (even_len < 3) {
            pred_first = interp_linear(data[cur_ij_offset], data[cur_ij_offset + 2 * offset]);
        } else {
            pred_first = interp_quad_1(data[cur_ij_offset], data[cur_ij_offset + 2 * offset],
                                       data[cur_ij_offset + 4 * offset]);
        }
        quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + offset, data[cur_ij_offset + offset], pred_first, tid);

        size_t i = 0;
        if constexpr (EnableInnerOmp) {
            if (!legacy_stream) {  // same chunk-to-thread layout as the AVX lanes, so the streams interoperate
                const size_t step = AVX_256_parallelism;
                const size_t vec_chunks = even_len > step + 2 ? (even_len - 3) / step : 0;
                #pragma omp parallel for
                for (size_t chunk = 0; chunk < vec_chunks; ++chunk) {
                    int cur_tid = omp_get_thread_num();
                    for (size_t j = chunk * step; j < (chunk + 1) * step; ++j) {
                        size_t start = ((j << 1) + 3) * offset;
                        quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + start, data[cur_ij_offset + start],
                                      interp_cubic(data[cur_ij_offset + (j << 1) * offset],
                                                   data[cur_ij_offset + ((j << 1) + 2) * offset],
                                                   data[cur_ij_offset + ((j << 1) + 4) * offset],
                                                   data[cur_ij_offset + ((j << 1) + 6) * offset]), cur_tid);
                    }
                }
                i = vec_chunks * step;
            }
        }
        for (; i + 3 < even_len; ++i) {
            size_t start = ((i << 1) + 3) * offset;
            quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + start, data[cur_ij_offset + start],
                          interp_cubic(data[cur_ij_offset + (i << 1) * offset],
                                       data[cur_ij_offset + ((i << 1) + 2) * offset],
                                       data[cur_ij_offset + ((i << 1) + 4) * offset],
                                       data[cur_ij_offset + ((i << 1) + 6) * offset]), tid);
        }
        if (odd_len > 1) {
            if (odd_len < even_len) {
                T edge_pred = interp_quad_2(data[cur_ij_offset + ((even_len - 3) << 1) * offset],
                                            data[cur_ij_offset + ((even_len - 2) << 1) * offset],
                                            data[cur_ij_offset + ((even_len - 1) << 1) * offset]);
                int last = 2 * odd_len - 1;
                quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[cur_ij_offset + last * offset], edge_pred, tid);
            } else {
                if (odd_len > 2) {
                    T edge_pred = interp_quad_2(data[cur_ij_offset + ((even_len - 3) << 1) * offset],
                                                data[cur_ij_offset + ((even_len - 2) << 1) * offset],
                                                data[cur_ij_offset + ((even_len - 1) << 1) * offset]);
                    int last = 2 * odd_len - 3;
                    quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[cur_ij_offset + last * offset], edge_pred, tid);
                }
                T edge_pred = interp_linear1(data[cur_ij_offset + ((even_len - 2) << 1) * offset],
                                             data[cur_ij_offset + ((even_len - 1) << 1) * offset]);
                int last = 2 * odd_len - 1;
                quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[cur_ij_offset + last * offset], edge_pred, tid);
            }
        }
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_linear_and_quantize_1D(const T * buf, const size_t &len, T* data,
        size_t&  offset, size_t& cur_ij_offset, int& tid, QuantizeFunc &&quantize_func) {

        if(len == 1)
            return;

        auto odd_len = len / 2;
        auto even_len = len - odd_len;
        size_t i = 0;

        for (; i + 1  < odd_len; ++i) {
            size_t start = ((i << 1) + 1) * offset;
            quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + start,  data[start], interp_linear(buf[i], buf[i + 1]), tid);
        }
        T pred_edge;
        if(len < 3 )
            pred_edge = buf[even_len - 1];
        else
            pred_edge = interp_linear1(buf[even_len - 2], buf[even_len - 1]);
        int last = 2 * odd_len - 1;
        quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset , data[last * offset], pred_edge, tid);
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_cubic_and_quantize_1D(const T * buf, const size_t &len, T* data,
        size_t&  offset, size_t& cur_ij_offset, int& tid, QuantizeFunc &&quantize_func) {
       // assert(len <= max_dim);
        if(len == 1)
            return;

        auto odd_len = len / 2;
        auto even_len = len - odd_len;

        T pred_first;
        if(even_len < 2)
            pred_first = (buf[0]);
        else if(even_len < 3)
            pred_first = interp_linear(buf[0], buf[1]);
        else
            pred_first = interp_quad_1(buf[0], buf[1], buf[2]);
        quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + offset , data[offset], pred_first, tid);

        size_t i = 0;
        for (; i + 3  < even_len; ++i) {
            size_t start = ((i << 1) + 3) * offset;
            quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + start,  data[start], interp_cubic(buf[i], buf[i + 1], buf[i + 2], buf[i + 3]), tid);
        }

        if(odd_len > 1){
            if(odd_len < even_len){//the only boundary is p[len- 1]
                //odd_len < even_len so even_len > 2
                T edge_pred;
                edge_pred = interp_quad_2(buf[even_len - 3], buf[even_len - 2], buf[even_len - 1]);
                int last = 2 * odd_len - 1;
                quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[last * offset], edge_pred, tid);

            }
            else{//the boundary points are is p[len -2 ] and p[len -1 ]
                T edge_pred;
                if(odd_len > 2){ //len - 2
                 //odd_len = even_len so even_len > 2
                    edge_pred = interp_quad_2(buf[even_len - 3],  buf[even_len - 2], buf[even_len - 1]);
                    int last = 2 * odd_len - 3;
                    quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[last * offset], edge_pred, tid);
                }
                //len -1
                //odd_len = even_len so even_len > 1
                    edge_pred = interp_linear1(buf[even_len - 2], buf[even_len - 1]);
                    int last = 2 * odd_len - 1;
                    quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[last * offset], edge_pred, tid);
            }
        }
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_linear_and_quantize(const T * a, const T* b, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func) {
        size_t i = 0;
        for (; i < len; ++i) {
            size_t start = i * offset;
            quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + start,  data[start], interp_linear(a[i], b[i]), tid);
        }

    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_cubic_and_quantize(const T * a, const T* b, T* c, T*d, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func) {

        size_t i = 0;
        for (; i < len; ++i) {
            size_t start = i * offset;
            quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + start,  data[start], interp_cubic(a[i], b[i], c[i], d[i]), tid);
        }
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_equal_and_quantize(const T * a, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func) {
        size_t i = 0;
        for (; i < len; ++i) {
            size_t start = i * offset;
            quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + start,  data[start], a[i], tid);
        }
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_linear1_and_quantize(const T * a, const T* b, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func) {
        size_t i = 0;
        for (; i < len; ++i) {
            size_t start = i * offset;
            quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + start,  data[start], interp_linear1(a[i], b[i]), tid);
        }
    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_quad1_and_quantize(const T * a, const T* b, const T* c, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func) {
        size_t i = 0;
        for (; i < len; ++i) {
            size_t start = i * offset;
            quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + start,  data[start], interp_quad_1(a[i], b[i], c[i]), tid);
        }

    }

    template <class T, uint N, class QuantizerOMP>
    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void InterpolationDecomposition_OMP<T, N, QuantizerOMP>::interp_quad2_and_quantize(const T * a, const T* b, const T* c, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func) {
        size_t i = 0;
        for (; i < len; ++i) {
            size_t start = i * offset;
            quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + start,  data[start], interp_quad_2(a[i], b[i], c[i]), tid);
        }

    }

#endif

}
