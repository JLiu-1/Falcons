/**
 * @file InterpolationDecomposition.hpp
 * @author Bing Lu
 * @brief Falcons interpolation decomposition and SIMD integration.
 *
 * Copyright (c) 2026 Bing Lu.
 */

#ifndef Falcons_INTERPOLATION_DECOMPOSITION_HPP
#define Falcons_INTERPOLATION_DECOMPOSITION_HPP

#include <cmath>

#if defined(__linux__)
#include <sys/mman.h>
#endif
#include <cstring>
#include <cstdlib>
#include <type_traits>
#include "Decomposition.hpp"
#include "Falcons/def.hpp"
#include "Falcons/quantizer/Quantizer.hpp"
#include "Falcons/utils/Config.hpp"
#include "Falcons/utils/FileUtil.hpp"
#include "Falcons/utils/Interpolators.hpp"
#include "Falcons/utils/Iterator.hpp"
#include "Falcons/utils/MemoryUtil.hpp"
#include "Falcons/utils/BlockwiseIterator.hpp"

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __ARM_FEATURE_SVE2
#include <arm_sve.h>
#endif

namespace Falcons {
template <TUNING Tuning, class T, uint N, class Quantizer>
class InterpolationDecomposition : public concepts::DecompositionInterface<T, int16_t, N> {
   public:
    InterpolationDecomposition(const Config &conf, Quantizer quantizer) : quantizer(quantizer) {
        static_assert(std::is_base_of<concepts::QuantizerInterface<T, int>, Quantizer>::value,
                      "must implement the quantizer interface");
    }


    T *decompress(const Config &conf, int16_t* quant_inds_vec, T *dec_data) override {
        init();
        if constexpr (N != 1) {
#ifdef __ARM_FEATURE_SVE2
            buffer_len = max_dim +  2 * SVE2_parallelism - max_dim % SVE2_parallelism;
#else
            buffer_len = max_dim +  2 * AVX_256_parallelism - max_dim % AVX_256_parallelism;
#endif
            interp_buffer_1 = new T[buffer_len];
            interp_buffer_2 = new T[buffer_len];
            interp_buffer_3 = new T[buffer_len];
            interp_buffer_4 = new T[buffer_len];
        }
        // pred_buffer = new T[buffer_len];
        // for(size_t i =0;i<buffer_len;i++)   // zero-fill unnecessary: AVX tail over-reads are bounded out of stored output (bit-exact verified)
            // interp_buffer_1[i] = interp_buffer_2[i] = interp_buffer_3[i] = interp_buffer_4[i] = T(0);
        this->quant_inds = quant_inds_vec;
        double eb = quantizer.get_eb();

        if (anchor_stride == 0) {                                               // check whether used anchor points
            *dec_data = quantizer.recover(0, this->quant_inds[quant_index++]);  // no anchor points
        } else {
            recover_anchor_grid(dec_data);  // recover anchor points
            interp_level--;
        }

        for (int level = interp_level; level > 0 && level <= interp_level; level--) {
            // set level-wise error bound
            if (eb_alpha < 0) {
                if (level >= 3) {
                    quantizer.set_eb(eb * eb_ratio);
                } else {
                    quantizer.set_eb(eb);
                }
            } else if (eb_alpha >= 1) {
                double cur_ratio = pow(eb_alpha, level - 1);
                if (cur_ratio > eb_beta) {
                    cur_ratio = eb_beta;
                }
                quantizer.set_eb(eb / cur_ratio);
            }
            size_t stride = 1U << (level - 1);
            auto interp_block_size = blocksize * stride;
            auto inter_block_range = std::make_shared<multi_dimensional_range<T, N>>(
                dec_data, std::begin(original_dimensions), std::end(original_dimensions), interp_block_size, 0);
            auto inter_begin = inter_block_range->begin();
            auto inter_end = inter_block_range->end();
            for (auto block = inter_begin; block != inter_end; ++block) {
                auto end_idx = block.get_global_index();
                for (uint i = 0; i < N; i++) {
                    end_idx[i] += interp_block_size;
                    if (end_idx[i] > original_dimensions[i] - 1) {
                        end_idx[i] = original_dimensions[i] - 1;
                    }
                }
                auto recover_and_save = [&](size_t idx, T &d, T pred) {
                    d = quantizer.recover(pred, quant_inds[quant_index++]);
                };
                if (level == 1) {
                    interpolation<COMPMODE::DECOMP, true>(
                        dec_data, block.get_global_index(), end_idx, interpolators[interp_id],
                        recover_and_save, direction_sequence_id, stride);
                } else {
                    interpolation<COMPMODE::DECOMP, false>(
                        dec_data, block.get_global_index(), end_idx, interpolators[interp_id],
                        recover_and_save, direction_sequence_id, stride);
                }
            }
        }
        quantizer.postdecompress_data();

        if constexpr (N != 1) {
            delete [] interp_buffer_1;
            delete [] interp_buffer_2;
            delete [] interp_buffer_3;
            delete [] interp_buffer_4;
            delete [] z_batch_buf; z_batch_buf = nullptr; z_batch_cap = 0;
        }
        // delete [] pred_buffer;
        return dec_data;
    }

    // compress given the error bound
    std::tuple<int16_t*, uint64_t> compress(const Config &conf, T *data) override {
        std::copy_n(conf.dims.begin(), N, original_dimensions.begin());

        interp_id = conf.interpAlgo;
        direction_sequence_id = conf.interpDirection;
        anchor_stride = conf.interpAnchorStride;
        //blocksize = 102400;  // a empirical value. Can be very large but not helpful
        eb_alpha = conf.interpAlpha;
        eb_beta = conf.interpBeta;

        init();
        blocksize = max_dim;
        if constexpr (N != 1) {
#ifdef __ARM_FEATURE_SVE2
            buffer_len = max_dim +  2 * SVE2_parallelism - max_dim % SVE2_parallelism;
#else
            buffer_len = max_dim +  2 * AVX_256_parallelism - max_dim % AVX_256_parallelism;
#endif
            interp_buffer_1 = new T[buffer_len];
            interp_buffer_2 = new T[buffer_len];
            interp_buffer_3 = new T[buffer_len];
            interp_buffer_4 = new T[buffer_len];
        }
        // pred_buffer = new T[buffer_len];

        // for(size_t i =0;i<buffer_len;i++)
            // pred_buffer[i] =
            // interp_buffer_1[i] = interp_buffer_2[i] = interp_buffer_3[i] = interp_buffer_4[i] = T(0);

        double eb = quantizer.get_eb();
        quant_inds = new int16_t[num_elements + 16];  // +16 padding for SIMD chunk stores
#if defined(__linux__)
        {   // hint THP for this large hot array (works under madvise mode; noop if unsupported)
            uintptr_t _mb = reinterpret_cast<uintptr_t>(quant_inds) & ~static_cast<uintptr_t>(4095);
            madvise(reinterpret_cast<void *>(_mb),
                    (num_elements + 16) * sizeof(int16_t) + (reinterpret_cast<uintptr_t>(quant_inds) - _mb),
                    MADV_HUGEPAGE);
        }
#endif

        if (anchor_stride == 0) {  // check whether to use anchor points
            quant_inds[quant_index++] = quantizer.quantize_and_overwrite(*data, 0);  // no
        } else {
            build_anchor_grid(data);  // losslessly saving anchor points
            interp_level--;
        }
        for (int level = interp_level; level > 0 && level <= interp_level; level--) {
            double cur_eb = eb;
            // set level-wise error bound
            if (eb_alpha < 0) {
                if (level >= 3) {
                    cur_eb = eb * eb_ratio;
                } else {
                    cur_eb = eb;
                }
            } else if (eb_alpha >= 1) {
                double cur_ratio = pow(eb_alpha, level - 1);
                if (cur_ratio > eb_beta) {
                    cur_ratio = eb_beta;
                }
                cur_eb = eb / cur_ratio;
            }
            quantizer.set_eb(cur_eb);
            size_t stride = 1U << (level - 1);

            auto interp_block_size = blocksize * stride;

            auto inter_block_range = std::make_shared<multi_dimensional_range<T, N>>(
                data, std::begin(original_dimensions), std::end(original_dimensions), interp_block_size, 0);

            auto inter_begin = inter_block_range->begin();
            auto inter_end = inter_block_range->end();

            for (auto block = inter_begin; block != inter_end; ++block) {
                auto end_idx = block.get_global_index();
                for (uint i = 0; i < N; i++) {
                    end_idx[i] += interp_block_size;
                    if (end_idx[i] > original_dimensions[i] - 1) {
                        end_idx[i] = original_dimensions[i] - 1;
                    }
                }

                auto quantize_and_save = [&](size_t idx, T &d, T pred) {
                    quant_inds[quant_index++] = quantizer.quantize_and_overwrite(d, pred);
                };
                if (level == 1) {
                    interpolation<COMPMODE::COMP, true>(
                        data, block.get_global_index(), end_idx, interpolators[interp_id],
                        quantize_and_save, direction_sequence_id, stride);
                } else {
                    interpolation<COMPMODE::COMP, false>(
                        data, block.get_global_index(), end_idx, interpolators[interp_id],
                        quantize_and_save, direction_sequence_id, stride);
                }
            }
        }
        quantizer.set_eb(eb);
        quantizer.postcompress_data();
        if constexpr (N != 1) {
            delete [] interp_buffer_1;
            delete [] interp_buffer_2;
            delete [] interp_buffer_3;
            delete [] interp_buffer_4;
            delete [] z_batch_buf; z_batch_buf = nullptr; z_batch_cap = 0;
        }
        // delete [] pred_buffer;

        return {quant_inds, quant_index};
    }

    void save(uchar *&c) override {
        write(original_dimensions.data(), N, c);
        write(blocksize, c);
        write(interp_id, c);
        write(direction_sequence_id, c);
        write(anchor_stride, c);
        write(eb_alpha, c);
        write(eb_beta, c);


        quantizer.save(c);
    }
    std::vector<T>& test_unpred() {
        return quantizer.test_unpred();
    }
    void load(const uchar *&c, size_t &remaining_length) override {
        read(original_dimensions.data(), N, c, remaining_length);
        read(blocksize, c, remaining_length);
        read(interp_id, c, remaining_length);
        read(direction_sequence_id, c, remaining_length);
        read(anchor_stride, c, remaining_length);
        read(eb_alpha, c, remaining_length);
        read(eb_beta, c, remaining_length);


        quantizer.load(c, remaining_length);
    }

    std::pair<int, int> get_out_range() override { return quantizer.get_out_range(); }

   private:
    template <COMPMODE CompMode, bool SkipOverwrite>
    ALWAYS_INLINE void quantize_point(size_t idx, T &d, T pred) {
        (void)idx;
        if constexpr (CompMode == COMPMODE::COMP) {
            if constexpr (SkipOverwrite) {
                quant_inds[quant_index++] = quantizer.quantize_without_overwrite(d, pred);
            } else {
                quant_inds[quant_index++] = quantizer.quantize_and_overwrite(d, pred);
            }
        } else {
            d = quantizer.recover(pred, quant_inds[quant_index++]);
        }
    }

    void init() {
        quant_index = 0;
        //assert(blocksize % 2 == 0 && "Interpolation block size should be even numbers");
        assert((anchor_stride & anchor_stride - 1) == 0 && "Anchor stride should be 0 or 2's exponentials");
        num_elements = 1;
        interp_level = -1;
	    bool use_anchor = false;
        max_dim = 1;
        for (uint i = 0; i < N; i++) {
            if (interp_level < ceil(log2(original_dimensions[i]))) {
                interp_level = static_cast<int>(ceil(log2(original_dimensions[i])));
            }
    	    if (original_dimensions[i] > anchor_stride)
    	        use_anchor = true;
            num_elements *= original_dimensions[i];
            max_dim = std::max(max_dim,original_dimensions[i]);
        }

        if (!use_anchor)
            anchor_stride = 0;
        if (anchor_stride > 0) {
            int max_interpolation_level = static_cast<int>(log2(anchor_stride)) + 1;
            if (max_interpolation_level <= interp_level) {
                interp_level = max_interpolation_level;
            }
        }
        radius = quantizer.get_out_range().second >> 1;
#ifdef __AVX2__
        radius_avx = _mm256_set1_pd(radius);
        nradius_avx = _mm256_set1_pd(-radius);
        zero_avx_d = _mm256_set1_pd(0);


        if constexpr (std::is_same_v<T, float>) {
            nradius_avx_f = _mm256_set1_ps(-radius);   // int16 escape sentinel = -radius = -32768
        }
#endif
#ifdef __ARM_FEATURE_SVE2
        SVE2_parallelism = svcntb() / sizeof(T);
#endif
        original_dim_offsets[N - 1] = 1;
        for (int i = N - 2; i >= 0; i--) {
            original_dim_offsets[i] = original_dim_offsets[i + 1] * original_dimensions[i + 1];
        }

        dim_sequences = std::vector<std::array<int, N>>();
        auto sequence = std::array<int, N>();
        for (uint i = 0; i < N; i++) {
            sequence[i] = i;
        }
        do {
            dim_sequences.push_back(sequence);
        } while (std::next_permutation(sequence.begin(), sequence.end()));
    }

    void build_anchor_grid(T *data) {  // store anchor points. steplength: anchor_stride on each dimension
        std::array<size_t, N> strides;
        std::array<size_t, N> begins{0};
        std::fill(strides.begin(), strides.end(), anchor_stride);
        foreach
            <T, N>(data, 0, begins, original_dimensions, strides, original_dim_offsets,
                   [&](T *d) {quantizer.force_save_unpred(*d); });
    }

    void recover_anchor_grid(T *data) {  // recover anchor points. steplength: anchor_stride on each dimension
        std::array<size_t, N> strides;
        std::array<size_t, N> begins{0};
        std::fill(strides.begin(), strides.end(), anchor_stride);
        foreach
            <T, N>(data, 0, begins, original_dimensions, strides, original_dim_offsets, [&](T *d) {
                *d = quantizer.recover_unpred();
                // quant_index++;
            });
    }

    /**
     * Do interpolations along a certain dimension, and move through that dimension only.
     * This is the original API, described in the ICDE'21 paper.
     * @tparam QuantizeFunc
     * @param data
     * @param begin
     * @param end
     * @param stride
     * @param interp_func
     * @param quantize_func
     * @return
     */
    template <class QuantizeFunc>
    double interpolation_1d(T *data, size_t begin, size_t end, size_t stride, const std::string &interp_func,
                            QuantizeFunc &&quantize_func) {
        size_t n = (end - begin) / stride + 1;
        if (n <= 1) {
            return 0;
        }
        double predict_error = 0;

        size_t stride3x = 3 * stride;
        size_t stride5x = 5 * stride;
        if (interp_func == "linear" || n < 5) {
            // if (pb == PB_predict_overwrite) {
            for (size_t i = 1; i + 1 < n; i += 2) {
                T *d = data + begin + i * stride;
                quantize_func(d - data, *d, interp_linear(*(d - stride), *(d + stride)));
            }
            if (n % 2 == 0) {
                T *d = data + begin + (n - 1) * stride;
                if (n < 4) {
                    quantize_func(d - data, *d, *(d - stride));
                } else {
                    quantize_func(d - data, *d, interp_linear1(*(d - stride3x), *(d - stride)));
                }
            }
            // }
        } else {
            T *d;
            size_t i;
            for (i = 3; i + 3 < n; i += 2) {
                d = data + begin + i * stride;
                quantize_func(d - data, *d,
                              interp_cubic(*(d - stride3x), *(d - stride), *(d + stride), *(d + stride3x)));
            }
            d = data + begin + stride;
            quantize_func(d - data, *d, interp_quad_1(*(d - stride), *(d + stride), *(d + stride3x)));

            d = data + begin + i * stride;
            quantize_func(d - data, *d, interp_quad_2(*(d - stride3x), *(d - stride), *(d + stride)));
            if (n % 2 == 0) {
                d = data + begin + (n - 1) * stride;
                quantize_func(d - data, *d, interp_quad_3(*(d - stride5x), *(d - stride3x), *(d - stride)));
            }
        }

        return predict_error;
    }

    /**
     * Do all interpolations along a certain dimension on the full data grid. Moving on the fastest-dim.
     * This is the new API, described in the SIGMOD'24 paper.
     * @tparam QuantizeFunc
     * @param data
     * @param begin_idx
     * @param end_idx
     * @param direction
     * @param strides
     * @param math_stride
     * @param interp_func
     * @param quantize_func
     * @return
     */
    template <class QuantizeFunc>
    double interpolation_1d_fastest_dim_first(T *data, const std::array<size_t, N> &begin_idx,
                                              const std::array<size_t, N> &end_idx, const size_t &direction,
                                              std::array<size_t, N> &strides, const size_t &math_stride,
                                              const std::string &interp_func, QuantizeFunc &&quantize_func) {
        for (size_t i = 0; i < N; i++) {
            if (end_idx[i] < begin_idx[i]) return 0;
        }
        size_t math_begin_idx = begin_idx[direction], math_end_idx = end_idx[direction];
        size_t n = (math_end_idx - math_begin_idx) / math_stride + 1;
        if (n <= 1) {
            return 0;
        }
        double predict_error = 0.0;
        size_t offset = 0;
        size_t stride = math_stride * original_dim_offsets[direction];
        std::array<size_t, N> begins, ends, dim_offsets;
        for (size_t i = 0; i < N; i++) {
            begins[i] = 0;
            ends[i] = end_idx[i] - begin_idx[i] + 1;
            dim_offsets[i] = original_dim_offsets[i];
            offset += original_dim_offsets[i] * begin_idx[i];
        }
        dim_offsets[direction] = stride;
        size_t stride2x = 2 * stride;
        if (interp_func == "linear") {
            begins[direction] = 1;
            ends[direction] = n - 1;
            strides[direction] = 2;
            foreach
                <T, N>(data, offset, begins, ends, strides, dim_offsets,
                       [&](T *d) { quantize_func(d - data, *d, interp_linear(*(d - stride), *(d + stride))); });
            if (n % 2 == 0) {
                begins[direction] = n - 1;
                ends[direction] = n;
                foreach
                    <T, N>(data, offset, begins, ends, strides, dim_offsets, [&](T *d) {
                        if (n < 3)
                            quantize_func(d - data, *d, *(d - stride));
                        else
                            quantize_func(d - data, *d, interp_linear1(*(d - stride2x), *(d - stride)));
                    });
            }
        } else {
            size_t stride3x = 3 * stride;
            size_t i_start = 3;
            begins[direction] = i_start;
            ends[direction] = (n >= 3) ? (n - 3) : 0;
            strides[direction] = 2;
            foreach
                <T, N>(data, offset, begins, ends, strides, dim_offsets, [&](T *d) {
                    quantize_func(d - data, *d,
                                  interp_cubic(*(d - stride3x), *(d - stride), *(d + stride), *(d + stride3x)));
                });
            std::vector<size_t> boundaries;
            boundaries.push_back(1);
            if (n % 2 == 1 && n > 3) {
                boundaries.push_back(n - 2);
            }
            if (n % 2 == 0 && n > 4) {
                boundaries.push_back(n - 3);
            }
            if (n % 2 == 0 && n > 2) {
                boundaries.push_back(n - 1);
            }
            for (auto boundary : boundaries) {
                begins[direction] = boundary;
                ends[direction] = boundary + 1;
                foreach
                    <T, N>(data, offset, begins, ends, strides, dim_offsets, [&](T *d) {
                        if (boundary >= 3) {
                            if (boundary + 3 < n)
                                quantize_func(
                                    d - data, *d,
                                    interp_cubic(*(d - stride3x), *(d - stride), *(d + stride), *(d + stride3x)));
                            else if (boundary + 1 < n)
                                quantize_func(d - data, *d,
                                              interp_quad_2(*(d - stride3x), *(d - stride), *(d + stride)));
                            else
                                quantize_func(d - data, *d, interp_linear1(*(d - stride3x), *(d - stride)));
                        } else {
                            if (boundary + 3 < n)
                                quantize_func(d - data, *d,
                                              interp_quad_1(*(d - stride), *(d + stride), *(d + stride3x)));
                            else if (boundary + 1 < n)
                                quantize_func(d - data, *d, interp_linear(*(d - stride), *(d + stride)));
                            else
                                quantize_func(d - data, *d, *(d - stride));
                        }
                    });
            }
        }
        return predict_error;
    }

    template <COMPMODE CompMode, bool SkipOverwrite = false, class QuantizeFunc>
    ALWAYS_INLINE void interp_linear_and_quantize_1D(const T * buf, const size_t &len, T* data,
        size_t&  offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func);

    template <COMPMODE CompMode, bool SkipOverwrite = false, class QuantizeFunc>
    ALWAYS_INLINE void interp_cubic_and_quantize_1D(const T * buf, const size_t &len, T* data,
        size_t&  offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func);

    template <COMPMODE CompMode, bool SkipOverwrite = false, class QuantizeFunc>
    ALWAYS_INLINE void interp_linear_and_quantize(const T * a, const T* b, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func);

    template <COMPMODE CompMode, bool SkipOverwrite = false, class QuantizeFunc>
    ALWAYS_INLINE void interp_cubic_and_quantize(const T * a, const T* b, T* c, T*d, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func);

    template <COMPMODE CompMode, bool SkipOverwrite = false, class QuantizeFunc>
    ALWAYS_INLINE void interp_equal_and_quantize(const T * a, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func);

    template <COMPMODE CompMode, bool SkipOverwrite = false, class QuantizeFunc>
    ALWAYS_INLINE void interp_linear1_and_quantize(const T * a, const T* b, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func);

    template <COMPMODE CompMode, bool SkipOverwrite = false, class QuantizeFunc>
    ALWAYS_INLINE void interp_quad1_and_quantize(const T * a, const T* b, const T* c, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func);

    template <COMPMODE CompMode, bool SkipOverwrite = false, class QuantizeFunc>
    ALWAYS_INLINE void interp_quad2_and_quantize(const T * a, const T* b, const T* c, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func);

    template <COMPMODE CompMode, bool SkipOverwrite = false, class QuantizeFunc>
    ALWAYS_INLINE void interp_linear_and_quantize_1D_line(T *data, const size_t &len,
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func);

    template <COMPMODE CompMode, bool SkipOverwrite = false, class QuantizeFunc>
    ALWAYS_INLINE void interp_cubic_and_quantize_1D_line(T *data, const size_t &len,
        size_t& offset, size_t& cur_ij_offset, QuantizeFunc &&quantize_func);

#ifdef __AVX2__
    template<typename U = T, typename = std::enable_if_t<std::is_same_v<U, float>>>
    ALWAYS_INLINE void quantize_1D_float (__m256& sum, __m256& ori_avx, __m256& quant_avx, T tmp[8]);

    template<typename U = T, typename = std::enable_if_t<std::is_same_v<U, double>>>
    ALWAYS_INLINE void quantize_1D_double (__m256d& sum, __m256d& ori_avx, __m256d& quant_avx, T tmp[4]);

    template <COMPMODE CompMode, int step, bool FullOnly = false, bool SkipOverwrite = false, typename U = T, typename = std::enable_if_t<std::is_same_v<U, float>>>
    ALWAYS_INLINE void quantize_float (__m256& sum, size_t& start, T*& data, size_t& offset, size_t& len);

    template <COMPMODE CompMode, int step, bool FullOnly = false, bool SkipOverwrite = false, typename U = T, typename = std::enable_if_t<std::is_same_v<U, double>>>
    ALWAYS_INLINE void quantize_double (__m256d& sum, size_t& start, T*& data, size_t& offset, size_t& len);

#endif
#ifdef __ARM_FEATURE_SVE2
    template<typename U = T, typename = std::enable_if_t<std::is_same_v<U, float>>>
    ALWAYS_INLINE void quantize_1D_float (svfloat32_t& sum, svfloat32_t& ori_sve, svfloat32_t& quant_sve, T* tmp,
        svbool_t& pg, svbool_t& pg64);

    template<typename U = T, typename = std::enable_if_t<std::is_same_v<U, double>>>
    ALWAYS_INLINE void quantize_1D_double (svfloat64_t& sum, svfloat64_t& ori_sve, svfloat64_t& quant_sve, T* tmp,
        svbool_t& pg64);

    template <COMPMODE CompMode, bool FullOnly = false, bool SkipOverwrite = false, typename U = T, typename = std::enable_if_t<std::is_same_v<U, float>>>
    ALWAYS_INLINE void quantize_float (svfloat32_t& sum, size_t& start, T*& data, size_t& offset, size_t& len,
        const size_t& step, svbool_t& pg, svbool_t& pg64);

    template <COMPMODE CompMode, bool FullOnly = false, bool SkipOverwrite = false, typename U = T, typename = std::enable_if_t<std::is_same_v<U, double>>>
    ALWAYS_INLINE void quantize_double (svfloat64_t& sum, size_t& start, T*& data, size_t& offset, size_t& len,
        const size_t& step, svbool_t& pg64);

#endif

    template <COMPMODE CompMode, bool SkipOverwrite = false, class QuantizeFunc>
    double interpolation_1d_simd_2d_y(T *data, const std::array<size_t, N> &begin_idx,
                                              const std::array<size_t, N> &end_idx, const size_t &direction,
                                              std::array<size_t, N> &strides, const size_t &math_stride,
                                              const std::string &interp_func, QuantizeFunc &&quantize_func) {
        assert(direction == 0  && N == 2);
        for (size_t i = 0; i < N; ++i) {
            if (end_idx[i] < begin_idx[i]) return 0;
        }
        size_t math_begin_idx = begin_idx[direction], math_end_idx = end_idx[direction];
        size_t n = (math_end_idx - math_begin_idx) / math_stride + 1;
        if (n <= 1) {
            return 0;
        }
        double predict_error = 0.0;
        size_t offset = 0;
        size_t stride = math_stride * original_dim_offsets[direction];
        std::array<size_t, N> begins, ends, dim_offsets;
        for (size_t i = 0; i < N; i++) {
            begins[i] = 0;
            ends[i] = end_idx[i] - begin_idx[i] + 1;
            dim_offsets[i] = original_dim_offsets[i];
            offset += original_dim_offsets[i] * begin_idx[i];
        }
        dim_offsets[direction] = stride;
        size_t last_linear_stride = SkipOverwrite ? 3 * stride : 2 * stride;
        if (interp_func == "linear") {
            begins[direction] = 1;
            ends[direction] = (n >= 1) ? (n - 1) : 0;
            strides[direction] = 2;
            size_t vector_len = ends[1] > begins[1] ? (ends[1]-begins[1]-1)/strides[1] + 1 : 0;
            auto cur_buffer_1 = interp_buffer_1;
            auto cur_buffer_2 = interp_buffer_2;

            size_t buffer_idx = 0;

            if (begins[0] < ends[0]) {
                auto cur_ij_offset = offset + dim_offsets[0];
                for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                        auto cur_offset =  cur_ij_offset + k;
                        cur_buffer_1[buffer_idx] = data[cur_offset - stride];
                        cur_buffer_2[buffer_idx] = data[cur_offset + stride];
                        ++buffer_idx;
                }
                interp_linear_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1,cur_buffer_2, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
            }
            for(size_t i = begins[0] + strides[0]; i < ends[0]; i += strides[0]){
                auto cur_ij_offset = offset + i * dim_offsets[0];
                auto temp_buffer = cur_buffer_1;
                cur_buffer_1 = cur_buffer_2;
                cur_buffer_2 = temp_buffer;
                buffer_idx = 0;
                for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                    auto cur_offset =  cur_ij_offset + stride + k;
                    cur_buffer_2[buffer_idx++] = data[cur_offset];
                }
                interp_linear_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1,cur_buffer_2, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
            }

            if (n % 2 == 0) {
                auto cur_ij_offset = offset + (n - 1) * dim_offsets[0];
                if(n < 3) {
                    buffer_idx = 0;
                    auto cur_buffer_2 = interp_buffer_2;
                    for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                        auto cur_offset =  cur_ij_offset + k;
                        cur_buffer_2[buffer_idx] = data[cur_offset - stride];  // fix: was cur_ij_offset-stride+2k (double k), read OOB
                        ++buffer_idx;
                    }
                    interp_equal_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
                }
                else {
                    buffer_idx = 0;
                    for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                        auto cur_offset =  cur_ij_offset + k - last_linear_stride;
                        cur_buffer_1[buffer_idx] = data[cur_offset];
                        ++buffer_idx;
                    }
                    interp_linear1_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1, cur_buffer_2, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
                }
            }
        } else {
            size_t stride3x = 3 * stride;
            begins[direction] = 3;
            ends[direction] = (n >= 3) ? (n - 3) : 0;
            strides[direction] = 2;
            size_t vector_len = ends[1] > begins[1] ? (ends[1]-begins[1]-1) / strides[1] + 1 : 0;

            auto cur_buffer_1 = interp_buffer_1;
            auto cur_buffer_2 = interp_buffer_2;
            auto cur_buffer_3 = interp_buffer_3;
            auto cur_buffer_4 = interp_buffer_4;
            size_t buffer_idx = 0;
            auto cur_ij_offset = offset + dim_offsets[0];
            if (n >= 5) {
                for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                    auto cur_offset =  cur_ij_offset + k;
                    cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                    cur_buffer_3[buffer_idx] = data[cur_offset + stride];
                    cur_buffer_4[buffer_idx] = data[cur_offset + stride3x];
                    ++buffer_idx;
                }
                interp_quad1_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
            }
            else if (n >= 3) {
                for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                    auto cur_offset =  cur_ij_offset + k;
                    cur_buffer_3[buffer_idx] = data[cur_offset - stride];
                    cur_buffer_4[buffer_idx] = data[cur_offset + stride];
                    ++buffer_idx;
                }
                interp_linear_and_quantize<CompMode, SkipOverwrite>(cur_buffer_3, cur_buffer_4, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
            }
            else if (n >= 1) {
                for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                    auto cur_offset =  cur_ij_offset + k;
                    cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                    ++buffer_idx;
                }
                interp_equal_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
            }
            for(size_t i = begins[0]; i < ends[0]; i += strides[0]){
                auto temp_buffer = cur_buffer_1;
                cur_buffer_1 = cur_buffer_2;
                cur_buffer_2 = cur_buffer_3;
                cur_buffer_3 = cur_buffer_4;
                cur_buffer_4 = temp_buffer;

                buffer_idx = 0;
                cur_ij_offset = offset + i * dim_offsets[0];
                for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                    auto cur_offset =  cur_ij_offset + stride3x + k;
                    cur_buffer_4[buffer_idx++] = data[cur_offset];
                }
                interp_cubic_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1,cur_buffer_2,cur_buffer_3,cur_buffer_4, vector_len,
                        data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
            }
            if (n % 2 == 1) {
                if (n > 3) {
                    cur_ij_offset = offset + (n - 2) * dim_offsets[0];
                    interp_quad2_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                        data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
                }
            }
            else {
                if (n > 4) {
                    cur_ij_offset = offset + (n - 3) * dim_offsets[0];
                    interp_quad2_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                        data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
                }
                if (n > 2) {
                    cur_ij_offset = offset + (n - 1) * dim_offsets[0];
                    interp_linear1_and_quantize<CompMode, SkipOverwrite>(cur_buffer_3, cur_buffer_4, vector_len,
                        data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
                }
            }
        }
        return predict_error;
    }

    template <COMPMODE CompMode, bool SkipOverwrite = false, class QuantizeFunc>
    double interpolation_1d_simd_2d_x(T *data, const std::array<size_t, N> &begin_idx,
                                              const std::array<size_t, N> &end_idx, const size_t &direction,
                                              std::array<size_t, N> &strides, const size_t &math_stride,
                                              const std::string &interp_func, QuantizeFunc &&quantize_func) {
        // assert(direction==1 && N==2);
        for (size_t i = 0; i < N; ++i) {
            if (end_idx[i] < begin_idx[i]) return 0;
        }
        size_t math_begin_idx = begin_idx[direction], math_end_idx = end_idx[direction];
        size_t n = (math_end_idx - math_begin_idx) / math_stride + 1;
        if (n <= 1) {
            return 0;
        }
        double predict_error = 0.0;
        size_t offset = 0;
        size_t stride = math_stride * original_dim_offsets[direction];
        std::array<size_t, N> begins, ends, dim_offsets;
        for (size_t i = 0; i < N; ++i) {
            begins[i] = 0;
            ends[i] = end_idx[i] - begin_idx[i] + 1;
            dim_offsets[i] = original_dim_offsets[i];
            offset += original_dim_offsets[i] * begin_idx[i];
        }
        dim_offsets[direction] = stride;

        if (interp_func == "linear") {
            begins[direction] = 1;
            ends[direction] = n;
            strides[direction] = 2;
            for (size_t i = begins[0]; i < ends[0]; i += strides[0]) {
                auto cur_ij_offset = offset + i * dim_offsets[0];
                interp_linear_and_quantize_1D_line<CompMode, SkipOverwrite>(
                    data, n, dim_offsets[direction], cur_ij_offset, quantize_func);
            }
        } else {
            begins[direction] = 1;
            ends[direction] = n;
            strides[direction] = 2;
            for (size_t i = begins[0]; i < ends[0]; i += strides[0]) {
                auto cur_ij_offset = offset + i * dim_offsets[0];
                interp_cubic_and_quantize_1D_line<CompMode, SkipOverwrite>(
                    data, n, dim_offsets[direction], cur_ij_offset, quantize_func);
            }
        }
        return predict_error;
    }

    template <COMPMODE CompMode, bool SkipOverwrite = false, class QuantizeFunc>
    double interpolation_1d_simd_3d_z(T *data, const std::array<size_t, N> &begin_idx,
                                              const std::array<size_t, N> &end_idx, const size_t &direction,
                                              std::array<size_t, N> &strides, const size_t &math_stride,
                                              const std::string &interp_func, QuantizeFunc &&quantize_func,
                                              size_t gi_begin = 0, size_t gi_end = ~size_t(0)) {
        assert(direction==0  && N==3);
        for (size_t i = 0; i < N; i++) {
            if (end_idx[i] < begin_idx[i]) return 0;
        }
        size_t math_begin_idx = begin_idx[direction], math_end_idx = end_idx[direction];
        size_t n = (math_end_idx - math_begin_idx) / math_stride + 1;
        if (n <= 1) {
            return 0;
        }
        double predict_error = 0.0;
        // Plane-major multi-row gather for single-plane windows. Measured SLOWER inside the xyz
        // wavefront (sources are already L3-hot, so the scratch round-trip costs more than the
        // gather locality wins: off 0.454 < NB2 0.457 < NB4 0.461 < NB8 0.465 s on dir5). Kept for
        // reference; would only help a standalone cold-miss z sweep.
        constexpr bool _z_batch = false;
        constexpr size_t _z_nb = 8;       // rows per batch
        size_t offset = 0;
        size_t stride = math_stride * original_dim_offsets[direction];
        std::array<size_t, N> begins, ends, dim_offsets;
        for (size_t i = 0; i < N; i++) {
            begins[i] = 0;
            ends[i] = end_idx[i] - begin_idx[i] + 1;
            dim_offsets[i] = original_dim_offsets[i];
            offset += original_dim_offsets[i] * begin_idx[i];
        }
        dim_offsets[direction] = stride;

        if (interp_func == "linear") {
            size_t last_linear_stride = SkipOverwrite ? 3 * stride : 2 * stride;
            begins[direction] = 1;
            ends[direction] = (n >= 1) ? (n - 1) : 0;
            strides[direction] = 2;
            size_t vector_len = ends[2] > begins[2] ? (ends[2]-begins[2]-1)/strides[2] + 1 : 0;


            size_t li0 = (gi_begin > begins[0]) ? gi_begin : begins[0];       /* window clamp (grid units) */
            size_t liE = (gi_end < ends[0]) ? gi_end : ends[0];
            if constexpr (_z_batch) if (gi_end == gi_begin + 2 && gi_begin >= begins[0] && gi_begin < ends[0]) {
                // single interior plane: gather NB rows per source plane (contiguous sweeps), then quantize
                const size_t zi_ = gi_begin;
                const size_t need = 2 * _z_nb * buffer_len;
                if (z_batch_cap < need) { delete[] z_batch_buf; z_batch_buf = new T[need]; z_batch_cap = need; }
                T *zs = z_batch_buf;
                for (size_t j0 = begins[1]; j0 < ends[1]; j0 += strides[1] * _z_nb) {
                    size_t nb = 0;
                    for (size_t jj = j0; jj < ends[1] && nb < _z_nb; jj += strides[1]) ++nb;
                    for (int p = 0; p < 2; ++p) {
                        const size_t g = zi_ + (p == 0 ? size_t(-1) : size_t(1));
                        for (size_t b = 0; b < nb; ++b) {
                            const size_t j = j0 + b * strides[1];
                            T *buf = zs + (p * _z_nb + b) * buffer_len;
                            size_t bi = 0;
                            const size_t base = offset + g * dim_offsets[0] + j * dim_offsets[1];
                            for (size_t k = begins[2]; k < ends[2]; k += strides[2]) buf[bi++] = data[base + k];
                        }
                    }
                    for (size_t b = 0; b < nb; ++b) {
                        const size_t j = j0 + b * strides[1];
                        auto cur_ij_offset = offset + zi_ * dim_offsets[0] + j * dim_offsets[1];
                        interp_linear_and_quantize<CompMode, SkipOverwrite>(
                            zs + (0 * _z_nb + b) * buffer_len, zs + (1 * _z_nb + b) * buffer_len, vector_len,
                            data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                    }
                }
                return predict_error;
            }
            for (size_t j = begins[1]; j < ends[1]; j += strides[1]) {
                auto cur_buffer_1 = interp_buffer_1;
                auto cur_buffer_2 = interp_buffer_2;
                bool ran_main = false;
                for(size_t i = li0; i < liE; i += strides[0]){
                    ran_main = true;
                    auto cur_ij_offset = offset + i * dim_offsets[0] + j * dim_offsets[1];
                    size_t buffer_idx = 0;
                    if( i == li0) {
                        for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                            auto cur_offset =  cur_ij_offset + k;
                            cur_buffer_1[buffer_idx] = data[cur_offset - stride];
                            cur_buffer_2[buffer_idx] = data[cur_offset + stride];
                            ++buffer_idx;
                        }
                    }
                    else{
                        auto temp_buffer = cur_buffer_1;
                        cur_buffer_1 = cur_buffer_2;
                        cur_buffer_2 = temp_buffer;
                        buffer_idx = 0;
                        for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                            auto cur_offset =  cur_ij_offset + stride + k;
                            cur_buffer_2[buffer_idx++] = data[cur_offset];
                        }

                    }
                    interp_linear_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1,cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);

                }
                if (n % 2 == 0 && gi_begin <= n - 1 && n - 1 < gi_end) {
                    auto cur_ij_offset = offset + (n - 1) * dim_offsets[0] + j * dim_offsets[1];
                    if (!ran_main && n >= 3) {           /* windowed call: b2 (row below) is stale */
                        size_t buffer_idx = 0;
                        for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                            cur_buffer_2[buffer_idx++] = data[cur_ij_offset - stride + k];
                        }
                    }
                    if(n < 3) {
                        // n==2: interior loop skipped, so cur_buffer_2 holds stale rows.
                        // Fill it with the single available neighbor row (below) before predicting.
                        size_t buffer_idx = 0;
                        for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                            cur_buffer_2[buffer_idx++] = data[cur_ij_offset - stride + k];
                        }
                        interp_equal_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                    }
                    else {
                        size_t buffer_idx = 0;
                        for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                            auto cur_offset =  cur_ij_offset - last_linear_stride + k;
                           cur_buffer_1[buffer_idx++] = data[cur_offset];
                        }
                        interp_linear1_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1, cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                    }
                }
            }

        } else {
            size_t stride3x = 3 * stride;
            size_t i_start = 3;
            begins[direction] = i_start;
            ends[direction] = (n >= 3) ? (n - 3) : 0;
            strides[direction] = 2;
            size_t vector_len = ends[2] > begins[2] ? (ends[2]-begins[2]-1)/strides[2] + 1 : 0;
            size_t ci0 = (gi_begin > begins[0]) ? gi_begin : begins[0];       /* window clamp (grid units) */
            size_t ciE = (gi_end < ends[0]) ? gi_end : ends[0];
            if constexpr (_z_batch) if (gi_end == gi_begin + 2 && gi_begin >= begins[0] && gi_begin < ends[0]) {
                // single interior plane: gather NB rows per source plane (contiguous sweeps), then quantize
                const size_t zi_ = gi_begin;
                const size_t need = 4 * _z_nb * buffer_len;
                if (z_batch_cap < need) { delete[] z_batch_buf; z_batch_buf = new T[need]; z_batch_cap = need; }
                T *zs = z_batch_buf;
                static const long gof[4] = {-3, -1, 1, 3};
                for (size_t j0 = begins[1]; j0 < ends[1]; j0 += strides[1] * _z_nb) {
                    size_t nb = 0;
                    for (size_t jj = j0; jj < ends[1] && nb < _z_nb; jj += strides[1]) ++nb;
                    for (int p = 0; p < 4; ++p) {
                        const size_t g = (zi_ + gof[p]);
                        for (size_t b = 0; b < nb; ++b) {
                            const size_t j = j0 + b * strides[1];
                            T *buf = zs + (p * _z_nb + b) * buffer_len;
                            size_t bi = 0;
                            const size_t base = offset + g * dim_offsets[0] + j * dim_offsets[1];
                            for (size_t k = begins[2]; k < ends[2]; k += strides[2]) buf[bi++] = data[base + k];
                        }
                    }
                    for (size_t b = 0; b < nb; ++b) {
                        const size_t j = j0 + b * strides[1];
                        auto cur_ij_offset = offset + zi_ * dim_offsets[0] + j * dim_offsets[1];
                        interp_cubic_and_quantize<CompMode, SkipOverwrite>(
                            zs + (0 * _z_nb + b) * buffer_len, zs + (1 * _z_nb + b) * buffer_len,
                            zs + (2 * _z_nb + b) * buffer_len, zs + (3 * _z_nb + b) * buffer_len, vector_len,
                            data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                    }
                }
                return predict_error;
            }

            for (size_t j = begins[1]; j < ends[1]; j += strides[1]) {
                auto cur_buffer_1 = interp_buffer_1;
                auto cur_buffer_2 = interp_buffer_2;
                auto cur_buffer_3 = interp_buffer_3;
                auto cur_buffer_4 = interp_buffer_4;
                bool primed = false;   /* buffers hold valid sliding-window planes */
                auto zgather = [&](T *buf, size_t g) {                        /* gather grid-plane g, row j */
                    size_t bi = 0;
                    auto base = offset + g * dim_offsets[0] + j * dim_offsets[1];
                    for (size_t k = begins[2]; k < ends[2]; k += strides[2]) buf[bi++] = data[base + k];
                };
                if ((gi_begin <= 1 && 1 < gi_end) && n >= 5) {
                    size_t buffer_idx = 0;
                    auto cur_ij_offset = offset + dim_offsets[0] + j * dim_offsets[1];
                    for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                        auto cur_offset =  cur_ij_offset + k;
                        cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                        cur_buffer_3[buffer_idx] = data[cur_offset + stride];
                        cur_buffer_4[buffer_idx] = data[cur_offset + stride3x];
                        ++buffer_idx;
                    }
                    interp_quad1_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                    primed = true;
                }
                else if ((gi_begin <= 1 && 1 < gi_end) && n >= 3) {
                    size_t buffer_idx = 0;
                    auto cur_ij_offset = offset + dim_offsets[0] + j * dim_offsets[1];
                    for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                        auto cur_offset =  cur_ij_offset + k;
                        cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                        cur_buffer_3[buffer_idx] = data[cur_offset + stride];
                        ++buffer_idx;
                    }
                    interp_linear_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, cur_buffer_3, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                }
                else if ((gi_begin <= 1 && 1 < gi_end) && n >= 1) {
                    size_t buffer_idx = 0;
                    auto cur_ij_offset = offset + dim_offsets[0] + j * dim_offsets[1];
                    for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                        auto cur_offset =  cur_ij_offset + k;
                        cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                        ++buffer_idx;
                    }
                    interp_equal_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                }
                for(size_t i = ci0; i < ciE; i += strides[0]){
                    auto cur_ij_offset = offset + i * dim_offsets[0] + j * dim_offsets[1];
                    if (primed) {
                        auto temp_buffer = cur_buffer_1;
                        cur_buffer_1 = cur_buffer_2;
                        cur_buffer_2 = cur_buffer_3;
                        cur_buffer_3 = cur_buffer_4;
                        cur_buffer_4 = temp_buffer;
                        size_t buffer_idx = 0;
                        for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                            auto cur_offset =  cur_ij_offset + stride3x + k;
                           cur_buffer_4[buffer_idx++] = data[cur_offset];
                        }
                    } else {                       /* windowed call: prime all four predictor planes */
                        zgather(cur_buffer_1, i - 3);
                        zgather(cur_buffer_2, i - 1);
                        zgather(cur_buffer_3, i + 1);
                        zgather(cur_buffer_4, i + 3);
                        primed = true;
                    }
                    interp_cubic_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1,cur_buffer_2,cur_buffer_3,cur_buffer_4, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);

                }
                if (n % 2 == 1 && gi_begin <= n - 2 && n - 2 < gi_end) {
                    if (n > 3) {
                        auto cur_ij_offset = offset + (n - 2) * dim_offsets[0] + j * dim_offsets[1];
                        if (!primed) {             /* windowed call: fetch the planes quad2 reads */
                            zgather(cur_buffer_2, n - 5);
                            zgather(cur_buffer_3, n - 3);
                            zgather(cur_buffer_4, n - 1);
                            primed = true;
                        }
                        cur_buffer_1 = cur_buffer_2;
                        cur_buffer_2 = cur_buffer_3;
                        cur_buffer_3 = cur_buffer_4;
                        interp_quad2_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1, cur_buffer_2, cur_buffer_3, vector_len,
                            data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);

                    }
                }
                else if (n % 2 == 0 && gi_begin <= n - 3 && n - 3 < gi_end) {
                    if (n > 4) {
                        auto cur_ij_offset = offset + (n - 3) * dim_offsets[0] + j * dim_offsets[1];
                        if (!primed) {             /* windowed call: fetch the planes quad2 reads */
                            zgather(cur_buffer_2, n - 6);
                            zgather(cur_buffer_3, n - 4);
                            zgather(cur_buffer_4, n - 2);
                            primed = true;
                        }
                        interp_quad2_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                            data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                    }
                }
            }
            std::vector<size_t> boundaries;
            // boundaries.push_back(1);
            // if (n % 2 == 1 && n > 3) {
            //     boundaries.push_back(n - 2);
            // }
            // if (n % 2 == 0 && n > 4) {
            //     boundaries.push_back(n - 3);
            // }
            if (n % 2 == 0 && n > 2 && gi_begin <= n - 1 && n - 1 < gi_end) {
                boundaries.push_back(n - 1);
            }

            for (auto boundary : boundaries) {
                begins[direction] = boundary;
                ends[direction] = boundary + 1;
                foreach
                    <T, N>(data, offset, begins, ends, strides, dim_offsets, [&](T *d) {
                        if (boundary >= 3) {
                            if (boundary + 3 < n)
                                quantize_point<CompMode, SkipOverwrite>(
                                    d - data, *d,
                                    interp_cubic(*(d - stride3x), *(d - stride), *(d + stride), *(d + stride3x)));
                            else if (boundary + 1 < n)
                                quantize_point<CompMode, SkipOverwrite>(d - data, *d,
                                              interp_quad_2(*(d - stride3x), *(d - stride), *(d + stride)));
                            else
                                quantize_point<CompMode, SkipOverwrite>(d - data, *d, interp_linear1(*(d - stride3x), *(d - stride)));
                        } else {
                            if (boundary + 3 < n)
                                quantize_point<CompMode, SkipOverwrite>(d - data, *d,
                                              interp_quad_1(*(d - stride), *(d + stride), *(d + stride3x)));
                            else if (boundary + 1 < n)
                                quantize_point<CompMode, SkipOverwrite>(d - data, *d, interp_linear(*(d - stride), *(d + stride)));
                            else
                                quantize_point<CompMode, SkipOverwrite>(d - data, *d, *(d - stride));
                        }
                    });
            }
        }
        return predict_error;
    }

    template <COMPMODE CompMode, bool SkipOverwrite = false, class QuantizeFunc>
    double interpolation_1d_simd_3d_y(T *data, const std::array<size_t, N> &begin_idx,
                                              const std::array<size_t, N> &end_idx, const size_t &direction,
                                              std::array<size_t, N> &strides, const size_t &math_stride,
                                              const std::string &interp_func, QuantizeFunc &&quantize_func, size_t zi_begin=0, size_t zi_end=~size_t(0)) {
        assert(direction==1  && N==3);
        for (size_t i = 0; i < N; i++) {
            if (end_idx[i] < begin_idx[i]) return 0;
        }
        size_t math_begin_idx = begin_idx[direction], math_end_idx = end_idx[direction];
        size_t n = (math_end_idx - math_begin_idx) / math_stride + 1;
        if (n <= 1) {
            return 0;
        }
        double predict_error = 0.0;
        size_t offset = 0;
        size_t stride = math_stride * original_dim_offsets[direction];
        std::array<size_t, N> begins, ends, dim_offsets;
        for (size_t i = 0; i < N; i++) {
            begins[i] = 0;
            ends[i] = end_idx[i] - begin_idx[i] + 1;
            dim_offsets[i] = original_dim_offsets[i];
            offset += original_dim_offsets[i] * begin_idx[i];
        }
        dim_offsets[direction] = stride;
        if (zi_begin > begins[0]) begins[0] = zi_begin;   /* fuse tiling: restrict z-outer (covers boundary) */
        if (zi_end < ends[0]) ends[0] = zi_end;
        size_t last_linear_stride = SkipOverwrite ? 3 * stride : 2 * stride;
        if (interp_func == "linear") {
            begins[direction] = 1;
            ends[direction] = (n >= 1) ? (n - 1) : 0;
            strides[direction] = 2;
            size_t vector_len = ends[2] > begins[2] ? (ends[2]-begins[2]-1)/strides[2] + 1 : 0;

            for (size_t i = (zi_begin>begins[0]?zi_begin:begins[0]); i < (zi_end<ends[0]?zi_end:ends[0]); i += strides[0]) {
                auto cur_buffer_1 = interp_buffer_1;
                auto cur_buffer_2 = interp_buffer_2;
                // auto cur_pred_buffer = pred_buffer + buffer_offset;
                for(size_t j = begins[1]; j < ends[1]; j += strides[1]){
                    auto cur_ij_offset = offset + i * dim_offsets[0] + j * dim_offsets[1];
                    size_t buffer_idx = 0;
                    if( j == begins[1]) {
                        for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                            auto cur_offset =  cur_ij_offset + k;
                            //if (visited[cur_offset- stride3x]==0 or visited[cur_offset+ stride3x]==0)
                            //   std::cout<<"e1 "<<i<<" "<<j<<" "<<k<<" "<<stride3x<<std::endl;
                           cur_buffer_1[buffer_idx] = data[cur_offset - stride];
                           //cur_buffer_1[buffer_idx] = data[0];
                            cur_buffer_2[buffer_idx] = data[cur_offset + stride];
                           // cur_buffer_2[buffer_idx] = data[0];
                            ++buffer_idx;

                        }
                    }
                    else{
                        auto temp_buffer = cur_buffer_1;
                        cur_buffer_1 = cur_buffer_2;
                        cur_buffer_2 = temp_buffer;

                        buffer_idx = 0;
                        for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                            auto cur_offset =  cur_ij_offset + stride + k;
                           // if (visited[cur_offset]==0 )
                            //   std::cout<<"e2 "<<i<<" "<<j<<" "<<k<<" "<<cur_offset<<std::endl;

                           cur_buffer_2[buffer_idx++] = data[cur_offset];

                            //cur_buffer_4[buffer_idx++] = data[0];

                        }
                    }
                    interp_linear_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1, cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);

                }
                if (n % 2 == 0) {
                    auto cur_ij_offset = offset + i * dim_offsets[0] + (n - 1) * dim_offsets[1];
                    if(n < 3) {
                        // n==2: interior loop skipped, so cur_buffer_2 holds stale rows.
                        // Fill it with the single available neighbor row (below) before predicting.
                        size_t buffer_idx = 0;
                        for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                            cur_buffer_2[buffer_idx++] = data[cur_ij_offset - stride + k];
                        }
                        interp_equal_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                    }
                    else {
                        size_t buffer_idx = 0;
                        for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                            auto cur_offset =  cur_ij_offset - last_linear_stride + k;
                           cur_buffer_1[buffer_idx++] = data[cur_offset];
                        }
                        interp_linear1_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1, cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                    }
                }

            }
        } else {
            size_t stride3x = 3 * stride;
            size_t i_start = 3;
            begins[direction] = i_start;
            ends[direction] = (n >= 3) ? (n - 3) : 0;
            strides[direction] = 2;
            size_t vector_len = ends[2] > begins[2] ? (ends[2]-begins[2]-1)/strides[2] + 1 : 0;

            for (size_t i = (zi_begin>begins[0]?zi_begin:begins[0]); i < (zi_end<ends[0]?zi_end:ends[0]); i += strides[0]) {
                auto cur_buffer_1 = interp_buffer_1;
                auto cur_buffer_2 = interp_buffer_2;
                auto cur_buffer_3 = interp_buffer_3;
                auto cur_buffer_4 = interp_buffer_4;
                // auto cur_pred_buffer = pred_buffer + buffer_offset;
                if (n >= 5) {
                    size_t buffer_idx = 0;
                    auto cur_ij_offset = offset + i * dim_offsets[0] + dim_offsets[1];
                    for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                        auto cur_offset =  cur_ij_offset + k;
                        cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                        cur_buffer_3[buffer_idx] = data[cur_offset + stride];
                        cur_buffer_4[buffer_idx] = data[cur_offset + stride3x];
                        ++buffer_idx;
                    }
                    interp_quad1_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                }
                else if (n >= 3) {
                    size_t buffer_idx = 0;
                    auto cur_ij_offset = offset + i * dim_offsets[0] + dim_offsets[1];
                    for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                        auto cur_offset =  cur_ij_offset + k;
                        cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                        cur_buffer_3[buffer_idx] = data[cur_offset + stride];
                        ++buffer_idx;
                    }
                    interp_linear_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, cur_buffer_3, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                }

                else if (n >= 1) {
                    size_t buffer_idx = 0;
                    auto cur_ij_offset = offset + i * dim_offsets[0] + dim_offsets[1];
                    for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                        auto cur_offset =  cur_ij_offset + k;
                        cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                        ++buffer_idx;
                    }
                    interp_equal_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                }
                for(size_t j = begins[1]; j < ends[1]; j += strides[1]){
                    auto cur_ij_offset = offset + i * dim_offsets[0] + j * dim_offsets[1];
                    size_t buffer_idx = 0;
                        auto temp_buffer = cur_buffer_1;
                        cur_buffer_1 = cur_buffer_2;
                        cur_buffer_2 = cur_buffer_3;
                        cur_buffer_3 = cur_buffer_4;
                        cur_buffer_4 = temp_buffer;

                        buffer_idx = 0;
                        for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                            auto cur_offset =  cur_ij_offset + stride3x + k;
                           // if (visited[cur_offset]==0 )
                            //   std::cout<<"e2 "<<i<<" "<<j<<" "<<k<<" "<<cur_offset<<std::endl;
                           cur_buffer_4[buffer_idx++] = data[cur_offset];
                            //cur_buffer_4[buffer_idx++] = data[0];
                        }
                    interp_cubic_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1, cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                }
                if (n % 2 == 1) {
                    if (n > 3) {
                        // size_t buffer_idx = 0;
                        auto cur_ij_offset = offset + i * dim_offsets[0] + (n - 2) * dim_offsets[1];
                        interp_quad2_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                            data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);

                    }
                }
                else {
                    if (n > 4) {
                        // size_t buffer_idx = 0;
                        auto cur_ij_offset = offset + i * dim_offsets[0] + (n - 3) * dim_offsets[1];
                        cur_buffer_1 = cur_buffer_2;
                        cur_buffer_2 = cur_buffer_3;
                        cur_buffer_3 = cur_buffer_4;
                        interp_quad2_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1, cur_buffer_2, cur_buffer_3, vector_len,
                            data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                    }
                    // else if (n > 2) {
                        // size_t buffer_idx = 0;
                        // auto cur_ij_offset = offset + i * dim_offsets[0] + (n - 1) * dim_offsets[1];
                        // interp_linear1_and_quantize<CompMode>(cur_buffer_2, cur_buffer_3, vector_len,
                        //     data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                    // }
                }
            }

            std::vector<size_t> boundaries;
            // boundaries.push_back(1);
            // if (n % 2 == 1 && n > 3) {
            //     boundaries.push_back(n - 2);
            // }
            // if (n % 2 == 0 && n > 4) {
            //     boundaries.push_back(n - 3);
            // }
            if (n % 2 == 0 && n > 2) {
                boundaries.push_back(n - 1);
            }

            for (auto boundary : boundaries) {
                begins[direction] = boundary;
                ends[direction] = boundary + 1;
                foreach
                    <T, N>(data, offset, begins, ends, strides, dim_offsets, [&](T *d) {
                        if (boundary >= 3) {
                            if (boundary + 3 < n)
                                quantize_point<CompMode, SkipOverwrite>(
                                    d - data, *d,
                                    interp_cubic(*(d - stride3x), *(d - stride), *(d + stride), *(d + stride3x)));
                            else if (boundary + 1 < n)
                                quantize_point<CompMode, SkipOverwrite>(d - data, *d,
                                              interp_quad_2(*(d - stride3x), *(d - stride), *(d + stride)));
                            else
                                quantize_point<CompMode, SkipOverwrite>(d - data, *d, interp_linear1(*(d - stride3x), *(d - stride)));
                        } else {
                            if (boundary + 3 < n)
                                quantize_point<CompMode, SkipOverwrite>(d - data, *d,
                                              interp_quad_1(*(d - stride), *(d + stride), *(d + stride3x)));
                            else if (boundary + 1 < n)
                                quantize_point<CompMode, SkipOverwrite>(d - data, *d, interp_linear(*(d - stride), *(d + stride)));
                            else
                                quantize_point<CompMode, SkipOverwrite>(d - data, *d, *(d - stride));
                        }
                    });
            }
        }
        return predict_error;
    }


    template <COMPMODE CompMode, bool SkipOverwrite = false, class QuantizeFunc>
    double interpolation_1d_simd_3d_x(T *data, const std::array<size_t, N> &begin_idx,
                                              const std::array<size_t, N> &end_idx, const size_t &direction,
                                              std::array<size_t, N> &strides, const size_t &math_stride,
                                              const std::string &interp_func, QuantizeFunc &&quantize_func, size_t zi_begin=0, size_t zi_end=~size_t(0)) {
        assert(direction==2 && N==3);
        for (size_t i = 0; i < N; i++) {
            if (end_idx[i] < begin_idx[i]) return 0;
        }
        size_t math_begin_idx = begin_idx[direction], math_end_idx = end_idx[direction];
        size_t n = (math_end_idx - math_begin_idx) / math_stride + 1;
        if (n <= 1) {
            return 0;
        }
        double predict_error = 0.0;
        size_t offset = 0;
        size_t stride = math_stride * original_dim_offsets[direction];
        std::array<size_t, N> begins, ends, dim_offsets;
        for (size_t i = 0; i < N; i++) {
            begins[i] = 0;
            ends[i] = end_idx[i] - begin_idx[i] + 1;
            dim_offsets[i] = original_dim_offsets[i];
            offset += original_dim_offsets[i] * begin_idx[i];
        }
        dim_offsets[direction] = stride;

        if (interp_func == "linear") {
            begins[direction] = 1;
            ends[direction] = n;
            strides[direction] = 2;
            for (size_t i = (zi_begin>begins[0]?zi_begin:begins[0]); i < (zi_end<ends[0]?zi_end:ends[0]); i += strides[0]) {
                for(size_t j = begins[1]; j < ends[1]; j += strides[1]){
                    auto cur_ij_offset = offset + i * dim_offsets[0] + j * dim_offsets[1];
                    interp_linear_and_quantize_1D_line<CompMode, SkipOverwrite>(
                        data, n, dim_offsets[direction], cur_ij_offset, quantize_func);
                }
            }
        } else {
            begins[direction] = 1;
            ends[direction] = n;
            strides[direction] = 2;
            for (size_t i = (zi_begin>begins[0]?zi_begin:begins[0]); i < (zi_end<ends[0]?zi_end:ends[0]); i += strides[0]) {
                for(size_t j = begins[1]; j < ends[1]; j += strides[1]){
                    auto cur_ij_offset = offset + i * dim_offsets[0] + j * dim_offsets[1];
                    interp_cubic_and_quantize_1D_line<CompMode, SkipOverwrite>(
                        data, n, dim_offsets[direction], cur_ij_offset, quantize_func);
                }
            }
        }
        return predict_error;
    }

    // Row-level fused y+x wavefront within a z-slab (direction 0 only).
    // Mirrors interpolation_1d_simd_3d_y (begin_idx = the y-pass config), but as soon as a row
    // becomes final it is immediately x-interpolated while still cache-hot: after y produces odd
    // row j, the x cursor sweeps every pending row (element index <= that row). The y predictor
    // buffers keep their sliding reuse. SkipX applies to the x lines (the final pass); y rows
    // always write back. x_row_begin / x_col_begin are the x-pass config (bx[1] / bx[2]).
    template <COMPMODE CompMode, bool SkipX = false, class QuantizeFunc>
    double interpolation_1d_simd_3d_yx(T *data, const std::array<size_t, N> &begin_idx,
                                       const std::array<size_t, N> &end_idx,
                                       std::array<size_t, N> &strides, const size_t &math_stride,
                                       const std::string &interp_func, QuantizeFunc &&quantize_func,
                                       size_t zi_begin, size_t zi_end,
                                       size_t x_row_begin, size_t x_col_begin) {
        constexpr size_t direction = 1;
        assert(N == 3);
        for (size_t i = 0; i < N; i++) {
            if (end_idx[i] < begin_idx[i]) return 0;
        }
        size_t math_begin_idx = begin_idx[direction], math_end_idx = end_idx[direction];
        size_t n = (math_end_idx - math_begin_idx) / math_stride + 1;
        double predict_error = 0.0;
        size_t offset = 0;
        size_t stride = math_stride * original_dim_offsets[direction];
        std::array<size_t, N> begins, ends, dim_offsets;
        for (size_t i = 0; i < N; i++) {
            begins[i] = 0;
            ends[i] = end_idx[i] - begin_idx[i] + 1;
            dim_offsets[i] = original_dim_offsets[i];
            offset += original_dim_offsets[i] * begin_idx[i];
        }
        dim_offsets[direction] = stride;
        if (zi_begin > begins[0]) begins[0] = zi_begin;   /* slab tiling (covers boundary too) */
        if (zi_end < ends[0]) ends[0] = zi_end;

        // x-line setup: exactly the rows/columns the separate x pass would process
        const bool x_linear = (interp_func == "linear");
        const size_t x_n = (end_idx[2] >= x_col_begin) ? (end_idx[2] - x_col_begin) / math_stride + 1 : 0;
        const size_t x_stride_e = math_stride * original_dim_offsets[2];
        const size_t x_base = original_dim_offsets[0] * begin_idx[0] + original_dim_offsets[2] * x_col_begin;

        if (n <= 1) {   // no y targets in this block: still emit all x lines of the slab
            if (x_n > 1) {
                for (size_t i = begins[0]; i < ends[0]; i += strides[0]) {
                    for (size_t r = x_row_begin; r <= end_idx[1]; r += math_stride) {
                        size_t xs = x_stride_e;
                        size_t ro = x_base + i * original_dim_offsets[0] + r * original_dim_offsets[1];
                        if (x_linear) interp_linear_and_quantize_1D_line<CompMode, SkipX>(data, x_n, xs, ro, quantize_func);
                        else          interp_cubic_and_quantize_1D_line<CompMode, SkipX>(data, x_n, xs, ro, quantize_func);
                    }
                }
            }
            return 0;
        }

        if (interp_func == "linear") {
            size_t last_linear_stride = 2 * stride;   // y never skips (not the final pass)
            begins[direction] = 1;
            ends[direction] = (n >= 1) ? (n - 1) : 0;
            strides[direction] = 2;
            size_t vector_len = ends[2] > begins[2] ? (ends[2]-begins[2]-1)/strides[2] + 1 : 0;

            for (size_t i = begins[0]; i < ends[0]; i += strides[0]) {
                size_t next_x = x_row_begin;
                auto x_upto = [&](size_t r_last) {   // x-interp every pending row <= r_last (element units)
                    if (x_n <= 1) return;
                    for (; next_x <= r_last && next_x <= end_idx[1]; next_x += math_stride) {
                        size_t xs = x_stride_e;
                        size_t ro = x_base + i * original_dim_offsets[0] + next_x * original_dim_offsets[1];
                        interp_linear_and_quantize_1D_line<CompMode, SkipX>(data, x_n, xs, ro, quantize_func);
                    }
                };
                auto cur_buffer_1 = interp_buffer_1;
                auto cur_buffer_2 = interp_buffer_2;
                for(size_t j = begins[1]; j < ends[1]; j += strides[1]){
                    auto cur_ij_offset = offset + i * dim_offsets[0] + j * dim_offsets[1];
                    size_t buffer_idx = 0;
                    if( j == begins[1]) {
                        for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                            auto cur_offset =  cur_ij_offset + k;
                            cur_buffer_1[buffer_idx] = data[cur_offset - stride];
                            cur_buffer_2[buffer_idx] = data[cur_offset + stride];
                            ++buffer_idx;
                        }
                    }
                    else{
                        auto temp_buffer = cur_buffer_1;
                        cur_buffer_1 = cur_buffer_2;
                        cur_buffer_2 = temp_buffer;
                        buffer_idx = 0;
                        for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                            auto cur_offset =  cur_ij_offset + stride + k;
                            cur_buffer_2[buffer_idx++] = data[cur_offset];
                        }
                    }
                    interp_linear_and_quantize<CompMode, false>(cur_buffer_1, cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                    x_upto(begin_idx[1] + j * math_stride);
                }
                if (n % 2 == 0) {
                    auto cur_ij_offset = offset + i * dim_offsets[0] + (n - 1) * dim_offsets[1];
                    if(n < 3) {
                        size_t buffer_idx = 0;
                        for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                            cur_buffer_2[buffer_idx++] = data[cur_ij_offset - stride + k];
                        }
                        interp_equal_and_quantize<CompMode, false>(cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                    }
                    else {
                        size_t buffer_idx = 0;
                        for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                            auto cur_offset =  cur_ij_offset - last_linear_stride + k;
                            cur_buffer_1[buffer_idx++] = data[cur_offset];
                        }
                        interp_linear1_and_quantize<CompMode, false>(cur_buffer_1, cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                    }
                    x_upto(begin_idx[1] + (n - 1) * math_stride);
                }
                x_upto(end_idx[1]);   // remaining even-row tail of the plane
            }
        } else {
            size_t stride3x = 3 * stride;
            begins[direction] = 3;
            ends[direction] = (n >= 3) ? (n - 3) : 0;
            strides[direction] = 2;
            size_t vector_len = ends[2] > begins[2] ? (ends[2]-begins[2]-1)/strides[2] + 1 : 0;

            for (size_t i = begins[0]; i < ends[0]; i += strides[0]) {
                size_t next_x = x_row_begin;
                auto x_upto = [&](size_t r_last) {
                    if (x_n <= 1) return;
                    for (; next_x <= r_last && next_x <= end_idx[1]; next_x += math_stride) {
                        size_t xs = x_stride_e;
                        size_t ro = x_base + i * original_dim_offsets[0] + next_x * original_dim_offsets[1];
                        interp_cubic_and_quantize_1D_line<CompMode, SkipX>(data, x_n, xs, ro, quantize_func);
                    }
                };
                auto cur_buffer_1 = interp_buffer_1;
                auto cur_buffer_2 = interp_buffer_2;
                auto cur_buffer_3 = interp_buffer_3;
                auto cur_buffer_4 = interp_buffer_4;
                if (n >= 5) {
                    size_t buffer_idx = 0;
                    auto cur_ij_offset = offset + i * dim_offsets[0] + dim_offsets[1];
                    for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                        auto cur_offset =  cur_ij_offset + k;
                        cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                        cur_buffer_3[buffer_idx] = data[cur_offset + stride];
                        cur_buffer_4[buffer_idx] = data[cur_offset + stride3x];
                        ++buffer_idx;
                    }
                    interp_quad1_and_quantize<CompMode, false>(cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                    x_upto(begin_idx[1] + math_stride);
                }
                else if (n >= 3) {
                    size_t buffer_idx = 0;
                    auto cur_ij_offset = offset + i * dim_offsets[0] + dim_offsets[1];
                    for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                        auto cur_offset =  cur_ij_offset + k;
                        cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                        cur_buffer_3[buffer_idx] = data[cur_offset + stride];
                        ++buffer_idx;
                    }
                    interp_linear_and_quantize<CompMode, false>(cur_buffer_2, cur_buffer_3, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                    x_upto(begin_idx[1] + math_stride);
                }
                else if (n >= 1) {
                    size_t buffer_idx = 0;
                    auto cur_ij_offset = offset + i * dim_offsets[0] + dim_offsets[1];
                    for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                        auto cur_offset =  cur_ij_offset + k;
                        cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                        ++buffer_idx;
                    }
                    interp_equal_and_quantize<CompMode, false>(cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                    x_upto(begin_idx[1] + math_stride);
                }
                for(size_t j = begins[1]; j < ends[1]; j += strides[1]){
                    auto cur_ij_offset = offset + i * dim_offsets[0] + j * dim_offsets[1];
                    size_t buffer_idx = 0;
                    auto temp_buffer = cur_buffer_1;
                    cur_buffer_1 = cur_buffer_2;
                    cur_buffer_2 = cur_buffer_3;
                    cur_buffer_3 = cur_buffer_4;
                    cur_buffer_4 = temp_buffer;
                    buffer_idx = 0;
                    for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                        auto cur_offset =  cur_ij_offset + stride3x + k;
                        cur_buffer_4[buffer_idx++] = data[cur_offset];
                    }
                    interp_cubic_and_quantize<CompMode, false>(cur_buffer_1, cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                    x_upto(begin_idx[1] + j * math_stride);
                }
                if (n % 2 == 1) {
                    if (n > 3) {
                        auto cur_ij_offset = offset + i * dim_offsets[0] + (n - 2) * dim_offsets[1];
                        interp_quad2_and_quantize<CompMode, false>(cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                            data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                        x_upto(begin_idx[1] + (n - 2) * math_stride);
                    }
                }
                else {
                    if (n > 4) {
                        auto cur_ij_offset = offset + i * dim_offsets[0] + (n - 3) * dim_offsets[1];
                        cur_buffer_1 = cur_buffer_2;
                        cur_buffer_2 = cur_buffer_3;
                        cur_buffer_3 = cur_buffer_4;
                        interp_quad2_and_quantize<CompMode, false>(cur_buffer_1, cur_buffer_2, cur_buffer_3, vector_len,
                            data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                        x_upto(begin_idx[1] + (n - 3) * math_stride);
                    }
                }
                if (n % 2 == 0 && n > 2) {
                    // edge row j = n-1 of THIS plane (scalar), then its x line via the cursor
                    const size_t boundary = n - 1;
                    std::array<size_t, N> eb = begins, ee = ends;
                    eb[direction] = boundary; ee[direction] = boundary + 1;
                    eb[0] = i; ee[0] = i + 1;
                    foreach
                        <T, N>(data, offset, eb, ee, strides, dim_offsets, [&](T *d) {
                            if (boundary >= 3) {
                                if (boundary + 3 < n)
                                    quantize_point<CompMode, false>(
                                        d - data, *d,
                                        interp_cubic(*(d - stride3x), *(d - stride), *(d + stride), *(d + stride3x)));
                                else if (boundary + 1 < n)
                                    quantize_point<CompMode, false>(d - data, *d,
                                                  interp_quad_2(*(d - stride3x), *(d - stride), *(d + stride)));
                                else
                                    quantize_point<CompMode, false>(d - data, *d, interp_linear1(*(d - stride3x), *(d - stride)));
                            } else {
                                if (boundary + 3 < n)
                                    quantize_point<CompMode, false>(d - data, *d,
                                                  interp_quad_1(*(d - stride), *(d + stride), *(d + stride3x)));
                                else if (boundary + 1 < n)
                                    quantize_point<CompMode, false>(d - data, *d, interp_linear(*(d - stride), *(d + stride)));
                                else
                                    quantize_point<CompMode, false>(d - data, *d, *(d - stride));
                            }
                        });
                    x_upto(begin_idx[1] + boundary * math_stride);
                }
                x_upto(end_idx[1]);
            }
        }
        return predict_error;
    }

    // Row-level fused x+y wavefront within a z-slab (direction 5 only).
    // Mirrors interpolation_1d_simd_3d_y (begin_idx = the y-pass config), but the x cursor LEADS:
    // before y interpolates odd row j, every even row its predictors read (j+1 linear, j+3 cubic)
    // has already been x-interpolated -- x rows are produced right before y consumes them,
    // cache-hot. dir5 x rows sit on the 2s grid (cursor step = 2*math_stride); odd rows are
    // filled completely by y itself. Both passes are non-final here (no skip).
    template <COMPMODE CompMode, class QuantizeFunc>
    double interpolation_1d_simd_3d_xy(T *data, const std::array<size_t, N> &begin_idx,
                                       const std::array<size_t, N> &end_idx,
                                       std::array<size_t, N> &strides, const size_t &math_stride,
                                       const std::string &interp_func, QuantizeFunc &&quantize_func,
                                       size_t zi_begin, size_t zi_end,
                                       size_t x_row_begin, size_t x_col_begin) {
        constexpr size_t direction = 1;
        assert(N == 3);
        for (size_t i = 0; i < N; i++) {
            if (end_idx[i] < begin_idx[i]) return 0;
        }
        size_t math_begin_idx = begin_idx[direction], math_end_idx = end_idx[direction];
        size_t n = (math_end_idx - math_begin_idx) / math_stride + 1;
        double predict_error = 0.0;
        size_t offset = 0;
        size_t stride = math_stride * original_dim_offsets[direction];
        std::array<size_t, N> begins, ends, dim_offsets;
        for (size_t i = 0; i < N; i++) {
            begins[i] = 0;
            ends[i] = end_idx[i] - begin_idx[i] + 1;
            dim_offsets[i] = original_dim_offsets[i];
            offset += original_dim_offsets[i] * begin_idx[i];
        }
        dim_offsets[direction] = stride;
        if (zi_begin > begins[0]) begins[0] = zi_begin;   /* slab tiling (covers boundary too) */
        if (zi_end < ends[0]) ends[0] = zi_end;

        const bool x_linear = (interp_func == "linear");
        const size_t x_n = (end_idx[2] >= x_col_begin) ? (end_idx[2] - x_col_begin) / math_stride + 1 : 0;
        const size_t x_stride_e = math_stride * original_dim_offsets[2];
        const size_t x_base = original_dim_offsets[0] * begin_idx[0] + original_dim_offsets[2] * x_col_begin;
        const size_t x_step = 2 * math_stride;   // dir5 x rows live on the 2s grid

        if (n <= 1) {   // no y targets: still emit the x lines of the slab
            if (x_n > 1) {
                for (size_t i = begins[0]; i < ends[0]; i += strides[0]) {
                    for (size_t r = x_row_begin; r <= end_idx[1]; r += x_step) {
                        size_t xs = x_stride_e;
                        size_t ro = x_base + i * original_dim_offsets[0] + r * original_dim_offsets[1];
                        if (x_linear) interp_linear_and_quantize_1D_line<CompMode, false>(data, x_n, xs, ro, quantize_func);
                        else          interp_cubic_and_quantize_1D_line<CompMode, false>(data, x_n, xs, ro, quantize_func);
                    }
                }
            }
            return 0;
        }

        if (interp_func == "linear") {
            size_t last_linear_stride = 2 * stride;   // y is not the final pass here
            begins[direction] = 1;
            ends[direction] = (n >= 1) ? (n - 1) : 0;
            strides[direction] = 2;
            size_t vector_len = ends[2] > begins[2] ? (ends[2]-begins[2]-1)/strides[2] + 1 : 0;

            for (size_t i = begins[0]; i < ends[0]; i += strides[0]) {
                size_t next_x = x_row_begin;
                auto x_upto = [&](size_t r_last) {
                    if (x_n <= 1) return;
                    for (; next_x <= r_last && next_x <= end_idx[1]; next_x += x_step) {
                        size_t xs = x_stride_e;
                        size_t ro = x_base + i * original_dim_offsets[0] + next_x * original_dim_offsets[1];
                        if (x_linear) interp_linear_and_quantize_1D_line<CompMode, false>(data, x_n, xs, ro, quantize_func);
                        else          interp_cubic_and_quantize_1D_line<CompMode, false>(data, x_n, xs, ro, quantize_func);
                    }
                };
                auto cur_buffer_1 = interp_buffer_1;
                auto cur_buffer_2 = interp_buffer_2;
                for(size_t j = begins[1]; j < ends[1]; j += strides[1]){
                    x_upto(begin_idx[1] + (j + 1) * math_stride);   // y reads even rows j-1, j+1
                    auto cur_ij_offset = offset + i * dim_offsets[0] + j * dim_offsets[1];
                    size_t buffer_idx = 0;
                    if( j == begins[1]) {
                        for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                            auto cur_offset =  cur_ij_offset + k;
                            cur_buffer_1[buffer_idx] = data[cur_offset - stride];
                            cur_buffer_2[buffer_idx] = data[cur_offset + stride];
                            ++buffer_idx;
                        }
                    }
                    else{
                        auto temp_buffer = cur_buffer_1;
                        cur_buffer_1 = cur_buffer_2;
                        cur_buffer_2 = temp_buffer;
                        buffer_idx = 0;
                        for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                            auto cur_offset =  cur_ij_offset + stride + k;
                            cur_buffer_2[buffer_idx++] = data[cur_offset];
                        }
                    }
                    interp_linear_and_quantize<CompMode, false>(cur_buffer_1, cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                }
                if (n % 2 == 0) {
                    auto cur_ij_offset = offset + i * dim_offsets[0] + (n - 1) * dim_offsets[1];
                    if(n < 3) {
                        size_t buffer_idx = 0;
                        for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                            cur_buffer_2[buffer_idx++] = data[cur_ij_offset - stride + k];
                        }
                        interp_equal_and_quantize<CompMode, false>(cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                    }
                    else {
                        size_t buffer_idx = 0;
                        for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                            auto cur_offset =  cur_ij_offset - last_linear_stride + k;
                            cur_buffer_1[buffer_idx++] = data[cur_offset];
                        }
                        interp_linear1_and_quantize<CompMode, false>(cur_buffer_1, cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                    }
                }
                x_upto(end_idx[1]);
            }
        } else {
            size_t stride3x = 3 * stride;
            begins[direction] = 3;
            ends[direction] = (n >= 3) ? (n - 3) : 0;
            strides[direction] = 2;
            size_t vector_len = ends[2] > begins[2] ? (ends[2]-begins[2]-1)/strides[2] + 1 : 0;

            for (size_t i = begins[0]; i < ends[0]; i += strides[0]) {
                size_t next_x = x_row_begin;
                auto x_upto = [&](size_t r_last) {
                    if (x_n <= 1) return;
                    for (; next_x <= r_last && next_x <= end_idx[1]; next_x += x_step) {
                        size_t xs = x_stride_e;
                        size_t ro = x_base + i * original_dim_offsets[0] + next_x * original_dim_offsets[1];
                        if (x_linear) interp_linear_and_quantize_1D_line<CompMode, false>(data, x_n, xs, ro, quantize_func);
                        else          interp_cubic_and_quantize_1D_line<CompMode, false>(data, x_n, xs, ro, quantize_func);
                    }
                };
                auto cur_buffer_1 = interp_buffer_1;
                auto cur_buffer_2 = interp_buffer_2;
                auto cur_buffer_3 = interp_buffer_3;
                auto cur_buffer_4 = interp_buffer_4;
                if (n >= 5) {
                    x_upto(begin_idx[1] + 4 * math_stride);   // quad1 reads rows 0,2,4
                    size_t buffer_idx = 0;
                    auto cur_ij_offset = offset + i * dim_offsets[0] + dim_offsets[1];
                    for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                        auto cur_offset =  cur_ij_offset + k;
                        cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                        cur_buffer_3[buffer_idx] = data[cur_offset + stride];
                        cur_buffer_4[buffer_idx] = data[cur_offset + stride3x];
                        ++buffer_idx;
                    }
                    interp_quad1_and_quantize<CompMode, false>(cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                }
                else if (n >= 3) {
                    x_upto(begin_idx[1] + 2 * math_stride);
                    size_t buffer_idx = 0;
                    auto cur_ij_offset = offset + i * dim_offsets[0] + dim_offsets[1];
                    for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                        auto cur_offset =  cur_ij_offset + k;
                        cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                        cur_buffer_3[buffer_idx] = data[cur_offset + stride];
                        ++buffer_idx;
                    }
                    interp_linear_and_quantize<CompMode, false>(cur_buffer_2, cur_buffer_3, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                }
                else if (n >= 1) {
                    x_upto(begin_idx[1]);
                    size_t buffer_idx = 0;
                    auto cur_ij_offset = offset + i * dim_offsets[0] + dim_offsets[1];
                    for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                        auto cur_offset =  cur_ij_offset + k;
                        cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                        ++buffer_idx;
                    }
                    interp_equal_and_quantize<CompMode, false>(cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                }
                for(size_t j = begins[1]; j < ends[1]; j += strides[1]){
                    x_upto(begin_idx[1] + (j + 3) * math_stride);   // cubic y reads even rows j-3..j+3
                    auto cur_ij_offset = offset + i * dim_offsets[0] + j * dim_offsets[1];
                    size_t buffer_idx = 0;
                    auto temp_buffer = cur_buffer_1;
                    cur_buffer_1 = cur_buffer_2;
                    cur_buffer_2 = cur_buffer_3;
                    cur_buffer_3 = cur_buffer_4;
                    cur_buffer_4 = temp_buffer;
                    buffer_idx = 0;
                    for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                        auto cur_offset =  cur_ij_offset + stride3x + k;
                        cur_buffer_4[buffer_idx++] = data[cur_offset];
                    }
                    interp_cubic_and_quantize<CompMode, false>(cur_buffer_1, cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                }
                if (n % 2 == 1) {
                    if (n > 3) {
                        x_upto(begin_idx[1] + (n - 1) * math_stride);   // quad2 reads n-5, n-3, n-1
                        auto cur_ij_offset = offset + i * dim_offsets[0] + (n - 2) * dim_offsets[1];
                        interp_quad2_and_quantize<CompMode, false>(cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                            data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                    }
                }
                else {
                    if (n > 4) {
                        x_upto(begin_idx[1] + (n - 2) * math_stride);   // quad2 reads n-6, n-4, n-2
                        auto cur_ij_offset = offset + i * dim_offsets[0] + (n - 3) * dim_offsets[1];
                        cur_buffer_1 = cur_buffer_2;
                        cur_buffer_2 = cur_buffer_3;
                        cur_buffer_3 = cur_buffer_4;
                        interp_quad2_and_quantize<CompMode, false>(cur_buffer_1, cur_buffer_2, cur_buffer_3, vector_len,
                            data + cur_ij_offset, strides[2], cur_ij_offset, quantize_func);
                    }
                }
                if (n % 2 == 0 && n > 2) {
                    const size_t boundary = n - 1;
                    x_upto(begin_idx[1] + (n - 2) * math_stride);   // edge reads n-4, n-2
                    std::array<size_t, N> eb = begins, ee = ends;
                    eb[direction] = boundary; ee[direction] = boundary + 1;
                    eb[0] = i; ee[0] = i + 1;
                    foreach
                        <T, N>(data, offset, eb, ee, strides, dim_offsets, [&](T *d) {
                            if (boundary >= 3) {
                                if (boundary + 3 < n)
                                    quantize_point<CompMode, false>(
                                        d - data, *d,
                                        interp_cubic(*(d - stride3x), *(d - stride), *(d + stride), *(d + stride3x)));
                                else if (boundary + 1 < n)
                                    quantize_point<CompMode, false>(d - data, *d,
                                                  interp_quad_2(*(d - stride3x), *(d - stride), *(d + stride)));
                                else
                                    quantize_point<CompMode, false>(d - data, *d, interp_linear1(*(d - stride3x), *(d - stride)));
                            } else {
                                if (boundary + 3 < n)
                                    quantize_point<CompMode, false>(d - data, *d,
                                                  interp_quad_1(*(d - stride), *(d + stride), *(d + stride3x)));
                                else if (boundary + 1 < n)
                                    quantize_point<CompMode, false>(d - data, *d, interp_linear(*(d - stride), *(d + stride)));
                                else
                                    quantize_point<CompMode, false>(d - data, *d, *(d - stride));
                            }
                        });
                }
                x_upto(end_idx[1]);
            }
        }
        return predict_error;
    }

    // Row-level fused y+x for the 2D dir0 case (dim0 pass leads, dim1 x lines lag). Mirrors
    // interpolation_1d_simd_2d_y (dim0 buffer pass, never skips) with the dim1 x lines emitted
    // cache-hot right after each dim0 row is final. SkipX applies to the x lines (the final pass).
    template <COMPMODE CompMode, bool SkipX = false, class QuantizeFunc>
    double interpolation_1d_simd_2d_yx(T *data, const std::array<size_t, N> &begin_idx,
                                       const std::array<size_t, N> &end_idx, std::array<size_t, N> &strides,
                                       const size_t &math_stride, const std::string &interp_func,
                                       QuantizeFunc &&quantize_func, size_t x_row_begin, size_t x_col_begin) {
        constexpr size_t direction = 0;
        assert(N == 2);
        for (size_t i = 0; i < N; ++i) if (end_idx[i] < begin_idx[i]) return 0;
        size_t n = (end_idx[direction] - begin_idx[direction]) / math_stride + 1;
        double predict_error = 0.0;
        size_t offset = 0;
        size_t stride = math_stride * original_dim_offsets[direction];
        std::array<size_t, N> begins, ends, dim_offsets;
        for (size_t i = 0; i < N; i++) {
            begins[i] = 0; ends[i] = end_idx[i] - begin_idx[i] + 1;
            dim_offsets[i] = original_dim_offsets[i]; offset += original_dim_offsets[i] * begin_idx[i];
        }
        dim_offsets[direction] = stride;
        const bool x_linear = (interp_func == "linear");
        const size_t x_n = (end_idx[1] >= x_col_begin) ? (end_idx[1] - x_col_begin) / math_stride + 1 : 0;
        const size_t x_stride_e = math_stride * original_dim_offsets[1];
        const size_t x_base = original_dim_offsets[1] * x_col_begin;
        size_t next_x = x_row_begin;
        auto x_upto = [&](size_t r_last) {   // dim1 x lines for every dim0 row <= r_last
            if (x_n <= 1) return;
            for (; next_x <= r_last && next_x <= end_idx[0]; next_x += math_stride) {
                size_t xs = x_stride_e;
                size_t ro = x_base + next_x * original_dim_offsets[0];
                if (x_linear) interp_linear_and_quantize_1D_line<CompMode, SkipX>(data, x_n, xs, ro, quantize_func);
                else          interp_cubic_and_quantize_1D_line<CompMode, SkipX>(data, x_n, xs, ro, quantize_func);
            }
        };
        if (n <= 1) { x_upto(end_idx[0]); return predict_error; }
        size_t last_linear_stride = 2 * stride;   // dim0 pass is not final here -> never skips
        if (interp_func == "linear") {
            begins[direction] = 1; ends[direction] = (n >= 1) ? (n - 1) : 0; strides[direction] = 2;
            size_t vector_len = ends[1] > begins[1] ? (ends[1]-begins[1]-1)/strides[1] + 1 : 0;
            auto cur_buffer_1 = interp_buffer_1;
            auto cur_buffer_2 = interp_buffer_2;
            size_t buffer_idx = 0;
            if (begins[0] < ends[0]) {
                auto cur_ij_offset = offset + dim_offsets[0];
                for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                    auto cur_offset = cur_ij_offset + k;
                    cur_buffer_1[buffer_idx] = data[cur_offset - stride];
                    cur_buffer_2[buffer_idx] = data[cur_offset + stride];
                    ++buffer_idx;
                }
                interp_linear_and_quantize<CompMode, false>(cur_buffer_1, cur_buffer_2, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
                x_upto(begin_idx[0] + math_stride);
            }
            for (size_t i = begins[0] + strides[0]; i < ends[0]; i += strides[0]) {
                auto cur_ij_offset = offset + i * dim_offsets[0];
                auto temp_buffer = cur_buffer_1; cur_buffer_1 = cur_buffer_2; cur_buffer_2 = temp_buffer;
                buffer_idx = 0;
                for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                    auto cur_offset = cur_ij_offset + stride + k;
                    cur_buffer_2[buffer_idx++] = data[cur_offset];
                }
                interp_linear_and_quantize<CompMode, false>(cur_buffer_1, cur_buffer_2, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
                x_upto(begin_idx[0] + i * math_stride);
            }
            if (n % 2 == 0) {
                auto cur_ij_offset = offset + (n - 1) * dim_offsets[0];
                if (n < 3) {
                    buffer_idx = 0;
                    auto cb2 = interp_buffer_2;
                    for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                        auto cur_offset = cur_ij_offset + k;
                        cb2[buffer_idx] = data[cur_offset - stride];
                        ++buffer_idx;
                    }
                    interp_equal_and_quantize<CompMode, false>(cb2, vector_len,
                        data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
                } else {
                    buffer_idx = 0;
                    for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                        auto cur_offset = cur_ij_offset + k - last_linear_stride;
                        cur_buffer_1[buffer_idx] = data[cur_offset];
                        ++buffer_idx;
                    }
                    interp_linear1_and_quantize<CompMode, false>(cur_buffer_1, cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
                }
                x_upto(begin_idx[0] + (n - 1) * math_stride);
            }
            x_upto(end_idx[0]);
        } else {
            size_t stride3x = 3 * stride;
            begins[direction] = 3; ends[direction] = (n >= 3) ? (n - 3) : 0; strides[direction] = 2;
            size_t vector_len = ends[1] > begins[1] ? (ends[1]-begins[1]-1)/strides[1] + 1 : 0;
            auto cur_buffer_1 = interp_buffer_1;
            auto cur_buffer_2 = interp_buffer_2;
            auto cur_buffer_3 = interp_buffer_3;
            auto cur_buffer_4 = interp_buffer_4;
            size_t buffer_idx = 0;
            auto cur_ij_offset = offset + dim_offsets[0];
            if (n >= 5) {
                for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                    auto cur_offset = cur_ij_offset + k;
                    cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                    cur_buffer_3[buffer_idx] = data[cur_offset + stride];
                    cur_buffer_4[buffer_idx] = data[cur_offset + stride3x];
                    ++buffer_idx;
                }
                interp_quad1_and_quantize<CompMode, false>(cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
            } else if (n >= 3) {
                for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                    auto cur_offset = cur_ij_offset + k;
                    cur_buffer_3[buffer_idx] = data[cur_offset - stride];
                    cur_buffer_4[buffer_idx] = data[cur_offset + stride];
                    ++buffer_idx;
                }
                interp_linear_and_quantize<CompMode, false>(cur_buffer_3, cur_buffer_4, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
            } else if (n >= 1) {
                for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                    auto cur_offset = cur_ij_offset + k;
                    cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                    ++buffer_idx;
                }
                interp_equal_and_quantize<CompMode, false>(cur_buffer_2, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
            }
            x_upto(begin_idx[0] + math_stride);
            for (size_t i = begins[0]; i < ends[0]; i += strides[0]) {
                auto temp_buffer = cur_buffer_1;
                cur_buffer_1 = cur_buffer_2; cur_buffer_2 = cur_buffer_3; cur_buffer_3 = cur_buffer_4; cur_buffer_4 = temp_buffer;
                buffer_idx = 0;
                cur_ij_offset = offset + i * dim_offsets[0];
                for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                    auto cur_offset = cur_ij_offset + stride3x + k;
                    cur_buffer_4[buffer_idx++] = data[cur_offset];
                }
                interp_cubic_and_quantize<CompMode, false>(cur_buffer_1, cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
                x_upto(begin_idx[0] + i * math_stride);
            }
            if (n % 2 == 1) {
                if (n > 3) {
                    cur_ij_offset = offset + (n - 2) * dim_offsets[0];
                    interp_quad2_and_quantize<CompMode, false>(cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                        data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
                    x_upto(begin_idx[0] + (n - 2) * math_stride);
                }
            } else {
                if (n > 4) {
                    cur_ij_offset = offset + (n - 3) * dim_offsets[0];
                    interp_quad2_and_quantize<CompMode, false>(cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                        data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
                    x_upto(begin_idx[0] + (n - 3) * math_stride);
                }
                if (n > 2) {
                    cur_ij_offset = offset + (n - 1) * dim_offsets[0];
                    interp_linear1_and_quantize<CompMode, false>(cur_buffer_3, cur_buffer_4, vector_len,
                        data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
                    x_upto(begin_idx[0] + (n - 1) * math_stride);
                }
            }
            x_upto(end_idx[0]);
        }
        return predict_error;
    }

    // Row-level fused x+y for the 2D else case (dim1 x lines lead, dim0 pass lags and is final).
    // Mirrors interpolation_1d_simd_2d_y (dim0 buffer pass, which SKIPS since it is the final pass)
    // but each even-dim0 row's dim1 x line is produced just before the dim0 pass reads it. x lines
    // sit on the 2s grid (cursor step 2*math_stride) and never skip.
    template <COMPMODE CompMode, bool SkipOverwrite = false, class QuantizeFunc>
    double interpolation_1d_simd_2d_xy(T *data, const std::array<size_t, N> &begin_idx,
                                       const std::array<size_t, N> &end_idx, std::array<size_t, N> &strides,
                                       const size_t &math_stride, const std::string &interp_func,
                                       QuantizeFunc &&quantize_func, size_t x_row_begin, size_t x_col_begin) {
        constexpr size_t direction = 0;
        assert(N == 2);
        for (size_t i = 0; i < N; ++i) if (end_idx[i] < begin_idx[i]) return 0;
        size_t n = (end_idx[direction] - begin_idx[direction]) / math_stride + 1;
        double predict_error = 0.0;
        size_t offset = 0;
        size_t stride = math_stride * original_dim_offsets[direction];
        std::array<size_t, N> begins, ends, dim_offsets;
        for (size_t i = 0; i < N; i++) {
            begins[i] = 0; ends[i] = end_idx[i] - begin_idx[i] + 1;
            dim_offsets[i] = original_dim_offsets[i]; offset += original_dim_offsets[i] * begin_idx[i];
        }
        dim_offsets[direction] = stride;
        const bool x_linear = (interp_func == "linear");
        const size_t x_n = (end_idx[1] >= x_col_begin) ? (end_idx[1] - x_col_begin) / math_stride + 1 : 0;
        const size_t x_stride_e = math_stride * original_dim_offsets[1];
        const size_t x_base = original_dim_offsets[1] * x_col_begin;
        const size_t x_step = 2 * math_stride;   // dim1 x lines live on even dim0 rows
        size_t next_x = x_row_begin;
        auto x_upto = [&](size_t r_last) {   // dim1 x lines for even dim0 rows <= r_last (produced ahead of the dim0 pass)
            if (x_n <= 1) return;
            for (; next_x <= r_last && next_x <= end_idx[0]; next_x += x_step) {
                size_t xs = x_stride_e;
                size_t ro = x_base + next_x * original_dim_offsets[0];
                if (x_linear) interp_linear_and_quantize_1D_line<CompMode, false>(data, x_n, xs, ro, quantize_func);
                else          interp_cubic_and_quantize_1D_line<CompMode, false>(data, x_n, xs, ro, quantize_func);
            }
        };
        if (n <= 1) { x_upto(end_idx[0]); return predict_error; }
        size_t last_linear_stride = SkipOverwrite ? 3 * stride : 2 * stride;   // dim0 pass is final -> skips
        if (interp_func == "linear") {
            begins[direction] = 1; ends[direction] = (n >= 1) ? (n - 1) : 0; strides[direction] = 2;
            size_t vector_len = ends[1] > begins[1] ? (ends[1]-begins[1]-1)/strides[1] + 1 : 0;
            auto cur_buffer_1 = interp_buffer_1;
            auto cur_buffer_2 = interp_buffer_2;
            size_t buffer_idx = 0;
            if (begins[0] < ends[0]) {
                x_upto(begin_idx[0] + 2 * math_stride);   // dim0 row 1 reads even rows 0, 2
                auto cur_ij_offset = offset + dim_offsets[0];
                for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                    auto cur_offset = cur_ij_offset + k;
                    cur_buffer_1[buffer_idx] = data[cur_offset - stride];
                    cur_buffer_2[buffer_idx] = data[cur_offset + stride];
                    ++buffer_idx;
                }
                interp_linear_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1, cur_buffer_2, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
            }
            for (size_t i = begins[0] + strides[0]; i < ends[0]; i += strides[0]) {
                x_upto(begin_idx[0] + (i + 1) * math_stride);
                auto cur_ij_offset = offset + i * dim_offsets[0];
                auto temp_buffer = cur_buffer_1; cur_buffer_1 = cur_buffer_2; cur_buffer_2 = temp_buffer;
                buffer_idx = 0;
                for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                    auto cur_offset = cur_ij_offset + stride + k;
                    cur_buffer_2[buffer_idx++] = data[cur_offset];
                }
                interp_linear_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1, cur_buffer_2, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
            }
            if (n % 2 == 0) {
                x_upto(begin_idx[0] + (n - 1) * math_stride);
                auto cur_ij_offset = offset + (n - 1) * dim_offsets[0];
                if (n < 3) {
                    buffer_idx = 0;
                    auto cb2 = interp_buffer_2;
                    for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                        auto cur_offset = cur_ij_offset + k;
                        cb2[buffer_idx] = data[cur_offset - stride];
                        ++buffer_idx;
                    }
                    interp_equal_and_quantize<CompMode, SkipOverwrite>(cb2, vector_len,
                        data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
                } else {
                    buffer_idx = 0;
                    for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                        auto cur_offset = cur_ij_offset + k - last_linear_stride;
                        cur_buffer_1[buffer_idx] = data[cur_offset];
                        ++buffer_idx;
                    }
                    interp_linear1_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1, cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
                }
            }
            x_upto(end_idx[0]);
        } else {
            size_t stride3x = 3 * stride;
            begins[direction] = 3; ends[direction] = (n >= 3) ? (n - 3) : 0; strides[direction] = 2;
            size_t vector_len = ends[1] > begins[1] ? (ends[1]-begins[1]-1)/strides[1] + 1 : 0;
            auto cur_buffer_1 = interp_buffer_1;
            auto cur_buffer_2 = interp_buffer_2;
            auto cur_buffer_3 = interp_buffer_3;
            auto cur_buffer_4 = interp_buffer_4;
            size_t buffer_idx = 0;
            auto cur_ij_offset = offset + dim_offsets[0];
            if (n >= 5) {
                x_upto(begin_idx[0] + 4 * math_stride);   // dim0 row 1 reads even rows 0,2,4
                for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                    auto cur_offset = cur_ij_offset + k;
                    cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                    cur_buffer_3[buffer_idx] = data[cur_offset + stride];
                    cur_buffer_4[buffer_idx] = data[cur_offset + stride3x];
                    ++buffer_idx;
                }
                interp_quad1_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
            } else if (n >= 3) {
                x_upto(begin_idx[0] + 2 * math_stride);
                for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                    auto cur_offset = cur_ij_offset + k;
                    cur_buffer_3[buffer_idx] = data[cur_offset - stride];
                    cur_buffer_4[buffer_idx] = data[cur_offset + stride];
                    ++buffer_idx;
                }
                interp_linear_and_quantize<CompMode, SkipOverwrite>(cur_buffer_3, cur_buffer_4, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
            } else if (n >= 1) {
                x_upto(begin_idx[0]);
                for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                    auto cur_offset = cur_ij_offset + k;
                    cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                    ++buffer_idx;
                }
                interp_equal_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
            }
            for (size_t i = begins[0]; i < ends[0]; i += strides[0]) {
                x_upto(begin_idx[0] + (i + 3) * math_stride);   // cubic dim0 row reads even rows i-3..i+3
                auto temp_buffer = cur_buffer_1;
                cur_buffer_1 = cur_buffer_2; cur_buffer_2 = cur_buffer_3; cur_buffer_3 = cur_buffer_4; cur_buffer_4 = temp_buffer;
                buffer_idx = 0;
                cur_ij_offset = offset + i * dim_offsets[0];
                for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                    auto cur_offset = cur_ij_offset + stride3x + k;
                    cur_buffer_4[buffer_idx++] = data[cur_offset];
                }
                interp_cubic_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1, cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
            }
            if (n % 2 == 1) {
                if (n > 3) {
                    x_upto(begin_idx[0] + (n - 1) * math_stride);
                    cur_ij_offset = offset + (n - 2) * dim_offsets[0];
                    interp_quad2_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                        data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
                }
            } else {
                if (n > 4) {
                    x_upto(begin_idx[0] + (n - 2) * math_stride);
                    cur_ij_offset = offset + (n - 3) * dim_offsets[0];
                    interp_quad2_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                        data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
                }
                if (n > 2) {
                    x_upto(begin_idx[0] + (n - 1) * math_stride);
                    cur_ij_offset = offset + (n - 1) * dim_offsets[0];
                    interp_linear1_and_quantize<CompMode, SkipOverwrite>(cur_buffer_3, cur_buffer_4, vector_len,
                        data + cur_ij_offset, strides[1], cur_ij_offset, quantize_func);
                }
            }
            x_upto(end_idx[0]);
        }
        return predict_error;
    }

    template <COMPMODE CompMode, bool SkipOverwrite = false, class QuantizeFunc>
    double interpolation_1d_simd(T *data, size_t begin, size_t end, size_t math_stride,
                                 const std::string &interp_func, QuantizeFunc &&quantize_func) {
        size_t n = (end - begin) / math_stride + 1;
        size_t stride = math_stride * original_dim_offsets[0];   // element stride (== math_stride for N==1)
        size_t cur_ij_offset = begin * original_dim_offsets[0];  // base offset of the line
        if (interp_func == "linear") {
            interp_linear_and_quantize_1D_line<CompMode, SkipOverwrite>(data, n, stride, cur_ij_offset, quantize_func);
        } else {
            interp_cubic_and_quantize_1D_line<CompMode, SkipOverwrite>(data, n, stride, cur_ij_offset, quantize_func);
        }
        return 0;
    }
    template <COMPMODE CompMode, bool SkipOverwrite = false, class QuantizeFunc>
    double interpolation(T *data, std::array<size_t, N> begin, std::array<size_t, N> end,
                         const std::string &interp_func, QuantizeFunc &&quantize_func, const int direction,
                         size_t stride = 1) {
#ifdef __ARM_FEATURE_SVE2
        std::tie(real_eb, real_ebx2, real_ebx2_r) = quantizer.get_all_eb();
#endif
#ifdef __AVX2__
        std::tie(real_eb, real_ebx2, real_ebx2_r) = quantizer.get_all_eb();
        ebx2_r_avx = _mm256_set1_pd(real_ebx2_r);
        ebx2_avx = _mm256_set1_pd(real_ebx2);
        if constexpr (std::is_same_v<T, float>) {
            // Round toward zero: for a float err, err <= eb_f then holds exactly when err <= eb.
            float eb_f = static_cast<float>(real_eb);
            if (static_cast<double>(eb_f) > real_eb) eb_f = std::nextafter(eb_f, 0.0f);
            rel_eb_avx_f = _mm256_set1_ps(eb_f);
            nrel_eb_avx_f = _mm256_set1_ps(-eb_f);
        }
        else if constexpr (std::is_same_v<T, double>) {
            rel_eb_avx_d = _mm256_set1_pd(real_eb);
            nrel_eb_avx_d = _mm256_set1_pd(-real_eb);
        }
#endif

        if constexpr (N == 1) {  // mirror the N==2 structure below
            if constexpr (std::is_integral_v<T>) {
                return interpolation_1d(data, begin[0], end[0], stride, interp_func, quantize_func);
            } else if constexpr (Tuning == TUNING::DISABLED) {
                // Floating builds go through the SIMD 1D path (interpolation_1d_simd ->
                // interp_*_and_quantize_1D_line, with scalar/AVX2/SVE2 variants). This is
                // the cross-ISA bit-exact path. The legacy scalar interpolation_1d diverged
                // from it at boundaries (different edge prediction).
                return interpolation_1d_simd<CompMode, SkipOverwrite>(data, begin[0], end[0], stride, interp_func, quantize_func);
            } else {  // single-thread tuning keeps the legacy API (as N==2 does)
                return interpolation_1d(data, begin[0], end[0], stride, interp_func, quantize_func);
            }
        } else if constexpr (N == 2) {  // old API
            double predict_error = 0;
            size_t stride2x = stride * 2;
            const std::array<int, N> dims = dim_sequences[direction];
            // size_t max_interp_seq_length = 0;
            //  for (uint i = 0; i < N; ++i)
            //     max_interp_seq_length = std::max(max_interp_seq_length, (end[i]-begin[i])/stride );
            if constexpr (Tuning == TUNING::DISABLED && !std::is_integral_v<T>) {
                std::array<size_t, N> strides;
                std::array<size_t, N> begin_idx = begin, end_idx = end;
                strides[dims[0]] = 1;
                for (uint i = 1; i < N; ++i) {
                    begin_idx[dims[i]] = (begin[dims[i]] ? begin[dims[i]] + stride2x : 0);
                    strides[dims[i]] = stride2x;
                }
                if(direction == 0 ){//xy
                    // constexpr bool _fuse_line2d = true;   // row-level dim0<->dim1 fusion for the 2D plane
                    // if (_fuse_line2d) {
                    //     std::array<size_t, N> sy = strides;   // callee mutates strides
                    //     predict_error += interpolation_1d_simd_2d_yx<CompMode, SkipOverwrite>(
                    //         data, begin_idx, end_idx, sy, stride, interp_func, quantize_func,
                    //         (begin[0] ? begin[0] + stride : 0), begin[1]);
                    // } else {
                        predict_error += interpolation_1d_simd_2d_y<CompMode, false>(data, begin_idx, end_idx, dims[0], strides, stride, interp_func, quantize_func);
                        begin_idx[1] = begin[1];
                        begin_idx[0] = (begin[0] ? begin[0] + stride : 0);
                        strides[0] = stride;
                        predict_error += interpolation_1d_simd_2d_x<CompMode, SkipOverwrite>(data, begin_idx, end_idx, dims[1], strides, stride, interp_func, quantize_func);
                    // }
                }
                else {
                    // constexpr bool _fuse_line2d = true;
                    // if (_fuse_line2d) {
                    //     std::array<size_t, N> by = begin_idx, sy = strides;   // dim0 (y) pass config
                    //     by[0] = begin[0]; by[1] = (begin[1] ? begin[1] + stride : 0); sy[1] = stride;
                    //     predict_error += interpolation_1d_simd_2d_xy<CompMode, SkipOverwrite>(
                    //         data, by, end_idx, sy, stride, interp_func, quantize_func,
                    //         (begin[0] ? begin[0] + stride2x : 0), begin[1]);
                    // } else {
                        predict_error += interpolation_1d_simd_2d_x<CompMode, false>(data, begin_idx, end_idx, dims[0], strides, stride, interp_func, quantize_func);
                        begin_idx[0] = begin[0];
                        begin_idx[1] = (begin[1] ? begin[1] + stride : 0);
                        strides[1] = stride;
                        predict_error += interpolation_1d_simd_2d_y<CompMode, SkipOverwrite>(data, begin_idx, end_idx, dims[1], strides, stride, interp_func, quantize_func);
                    // }
                }
            }
            else {
                for (size_t j = (begin[dims[1]] ? begin[dims[1]] + stride2x : 0); j <= end[dims[1]]; j += stride2x) {
                    size_t begin_offset =
                        begin[dims[0]] * original_dim_offsets[dims[0]] + j * original_dim_offsets[dims[1]];
                    predict_error += interpolation_1d(
                        data, begin_offset, begin_offset + (end[dims[0]] - begin[dims[0]]) * original_dim_offsets[dims[0]],
                        stride * original_dim_offsets[dims[0]], interp_func, quantize_func);
                }
                for (size_t i = (begin[dims[0]] ? begin[dims[0]] + stride : 0); i <= end[dims[0]]; i += stride) {
                    size_t begin_offset =
                        i * original_dim_offsets[dims[0]] + begin[dims[1]] * original_dim_offsets[dims[1]];
                    predict_error += interpolation_1d(
                        data, begin_offset, begin_offset + (end[dims[1]] - begin[dims[1]]) * original_dim_offsets[dims[1]],
                        stride * original_dim_offsets[dims[1]], interp_func, quantize_func);
                }
            }

            return predict_error;
        } else if constexpr (N == 3 || N == 4) {  // new API (for faster speed)
            double predict_error = 0;
            size_t stride2x = stride * 2;
            const std::array<int, N> dims = dim_sequences[direction];
            std::array<size_t, N> strides;
            std::array<size_t, N> begin_idx = begin, end_idx = end;
            strides[dims[0]] = 1;
            for (uint i = 1; i < N; i++) {
                begin_idx[dims[i]] = (begin[dims[i]] ? begin[dims[i]] + stride2x : 0);
                strides[dims[i]] = stride2x;
            }

            if constexpr (Tuning == TUNING::DISABLED && N == 3 && !std::is_integral_v<T>) {//avx &&stride<=2
                constexpr bool _fuse = false;      // x+y cache fusion, always on (set false to use the old path below)
                constexpr size_t _fuse_tile = 1;     // z-slab thickness (planes); 1 is optimal
                if (direction != 0 && direction != 5) {  // the SIMD sweeps only cover xyz and zyx
                    predict_error += interpolation_1d_fastest_dim_first(data, begin_idx, end_idx, dims[0], strides, stride, interp_func, quantize_func);
                    for (uint i = 1; i < N; i++) {
                        begin_idx[dims[i]] = begin[dims[i]];
                        begin_idx[dims[i - 1]] = (begin[dims[i - 1]] ? begin[dims[i - 1]] + stride : 0);
                        strides[dims[i - 1]] = stride;
                        predict_error += interpolation_1d_fastest_dim_first(data, begin_idx, end_idx, dims[i], strides, stride, interp_func, quantize_func);
                    }
                    return predict_error;
                }
                if(direction ==0 ){//xyz
                    // constexpr bool _fuse_xyz = true;   // OFF: z done as a separate pass; keep only xy-line fusion
                    std::array<size_t, N> bz = begin_idx, sz0 = strides;   // z-pass config (kept before the y mutations)

                    if (_fuse == false || stride !=1 ) {
                        predict_error += interpolation_1d_simd_3d_z<CompMode, false>(data, begin_idx, end_idx, dims[0], strides, stride, interp_func, quantize_func);
                    }
                    begin_idx[1] = begin[1];
                    begin_idx[0] = (begin[0] ? begin[0] + stride : 0);
                    strides[0] = stride;
                    if constexpr (_fuse) {
                        // fuse y & x: both loop dim0(z) outer, no cross-plane dep -> tile the z-loop so each
                        // z-slab stays cached across both passes.
                        std::array<size_t, N> by = begin_idx, sy = strides;               // y-config
                        std::array<size_t, N> bx = by, sx = sy;                            // x-config (derived)
                        bx[2] = begin[2]; bx[1] = (begin[1] ? begin[1] + stride : 0); sx[1] = stride;
                        size_t z_ext = end_idx[0] - by[0] + 1;
                        size_t tile_w = _fuse_tile * stride;
                        // constexpr bool _fuse_line = true;   // row-level y<->x wavefront inside the slab
                        for (size_t zi = 0; zi < z_ext; zi += tile_w) {
                            if (_fuse == true && stride == 1) {
                                size_t gi = (by[0] + zi) - bz[0];
                                if ((gi & 1) != 0) {
                                    std::array<size_t, N> sz2 = sz0;
                                    predict_error += interpolation_1d_simd_3d_z<CompMode, false>(data, bz, end_idx, dims[0], sz2, stride, interp_func, quantize_func, gi, gi + 2);
                                }
                            }
                            std::array<size_t, N> sy2 = sy, sx2 = sx;   // fresh copies (callees mutate strides)
                            // if (_fuse_line) {
                                predict_error += interpolation_1d_simd_3d_yx<CompMode, SkipOverwrite>(data, by, end_idx, sy2, stride, interp_func, quantize_func, zi, zi + tile_w, bx[1], bx[2]);
                            // } else {
                            //     predict_error += interpolation_1d_simd_3d_y<CompMode, false>(data, by, end_idx, dims[1], sy2, stride, interp_func, quantize_func, zi, zi + tile_w);
                            //     predict_error += interpolation_1d_simd_3d_x<CompMode, SkipOverwrite>(data, bx, end_idx, dims[2], sx2, stride, interp_func, quantize_func, zi, zi + tile_w);
                            // }
                        }
                    } else {
                        predict_error += interpolation_1d_simd_3d_y<CompMode, false>(data, begin_idx, end_idx, dims[1], strides, stride, interp_func, quantize_func);
                        begin_idx[2] = begin[2];
                        begin_idx[1] = (begin[1] ? begin[1] + stride : 0);
                        strides[1] = stride;
                        predict_error += interpolation_1d_simd_3d_x<CompMode, SkipOverwrite>(data, begin_idx, end_idx, dims[2], strides, stride, interp_func, quantize_func);
                    }
                }
                else{//zyx
                    // z-pass config (explicit; z is the FINAL pass here and reads fully xy-processed planes)
                    std::array<size_t, N> bz5;
                    bz5[0] = begin[0];
                    bz5[1] = (begin[1] ? begin[1] + stride : 0);
                    bz5[2] = (begin[2] ? begin[2] + stride : 0);
                    std::array<size_t, N> sz5 = strides; sz5[1] = stride; sz5[2] = stride;
                    const size_t zn = (end_idx[0] >= bz5[0]) ? (end_idx[0] - bz5[0]) / stride + 1 : 0;
                    size_t zgi = 1;   // next pending z window (z targets sit on odd grid indices)
                    if constexpr (_fuse) {
                        std::array<size_t, N> bx = begin_idx, sx = strides;               // x-config (initial state)
                        std::array<size_t, N> by = bx, sy = sx;                            // y-config (derived)
                        by[1] = begin[1]; by[2] = (begin[2] ? begin[2] + stride : 0); sy[2] = stride;
                        size_t z_ext = end_idx[0] - bx[0] + 1;
                        size_t tile_w = _fuse_tile * sx[0];
                        // constexpr bool _fuse_line = true;   // row-level x->y wavefront inside the slab
                        for (size_t zi = 0; zi < z_ext; zi += tile_w) {
                            std::array<size_t, N> sx2 = sx, sy2 = sy;
                            // if constexpr (_fuse_line) {
                                predict_error += interpolation_1d_simd_3d_xy<CompMode>(data, by, end_idx, sy2, stride, interp_func, quantize_func, zi, zi + tile_w, bx[1], bx[2]);
                            // } else {
                            //     predict_error += interpolation_1d_simd_3d_x<CompMode, false>(data, bx, end_idx, dims[0], sx2, stride, interp_func, quantize_func, zi, zi + tile_w);
                            //     predict_error += interpolation_1d_simd_3d_y<CompMode, false>(data, by, end_idx, dims[1], sy2, stride, interp_func, quantize_func, zi, zi + tile_w);
                            // }
                            if (_fuse == true && stride == 1 && zn >= 2) {
                                // z-interp every odd plane whose furthest forward predictor (gi+3, and the
                                // quad1 first plane also reads +3) is now fully xy-processed.
                                size_t ze = bx[0] + zi;                      // newest xy-complete plane (element)
                                while (zgi + 1 <= zn - 1 && bz5[0] + (zgi + 3) * stride <= ze) {
                                    std::array<size_t, N> sz2 = sz5;
                                    predict_error += interpolation_1d_simd_3d_z<CompMode, SkipOverwrite>(data, bz5, end_idx, dims[2], sz2, stride, interp_func, quantize_func, zgi, zgi + 2);
                                    zgi += 2;
                                }
                            }
                        }
                    } else {
                        predict_error += interpolation_1d_simd_3d_x<CompMode, false>(data, begin_idx, end_idx, dims[0], strides, stride, interp_func, quantize_func);
                        begin_idx[1] = begin[1];
                        begin_idx[2] = (begin[2] ? begin[2] + stride : 0);
                        strides[2] = stride;
                        predict_error += interpolation_1d_simd_3d_y<CompMode, false>(data, begin_idx, end_idx, dims[1], strides, stride, interp_func, quantize_func);
                    }
                    if (_fuse == true && stride == 1) {
                        // drain: remaining odd planes (their windows include the quad2/edge formulas,
                        // which only read backward or +1 -- all xy-complete by now)
                        if (zn >= 2) {
                            for (; zgi <= zn - 1; zgi += 2) {
                                std::array<size_t, N> sz2 = sz5;
                                predict_error += interpolation_1d_simd_3d_z<CompMode, SkipOverwrite>(data, bz5, end_idx, dims[2], sz2, stride, interp_func, quantize_func, zgi, zgi + 2);
                            }
                        }
                    } else {
                        // z-pass (last, separate); explicit z-config correct for both branches
                        begin_idx[0] = begin[0];
                        begin_idx[1] = (begin[1] ? begin[1] + stride : 0);
                        begin_idx[2] = (begin[2] ? begin[2] + stride : 0);
                        strides[1] = stride;
                        strides[2] = stride;
                        predict_error += interpolation_1d_simd_3d_z<CompMode, SkipOverwrite>(data, begin_idx, end_idx, dims[2], strides, stride, interp_func, quantize_func);
                    }
                }
            }
            else{
                predict_error += interpolation_1d_fastest_dim_first(data, begin_idx, end_idx, dims[0], strides, stride, interp_func, quantize_func);
                for (uint i = 1; i < N; i++) {
                begin_idx[dims[i]] = begin[dims[i]];
                begin_idx[dims[i - 1]] = (begin[dims[i - 1]] ? begin[dims[i - 1]] + stride : 0);
                strides[dims[i - 1]] = stride;

                predict_error += interpolation_1d_fastest_dim_first(data, begin_idx, end_idx, dims[i], strides, stride, interp_func, quantize_func);
                }
            }


            return predict_error;
        } else {
            throw std::runtime_error("Unsupported dimension in InterpolationDecomposition");
        }
    }

    int interp_level = -1;
    int interp_id;
    uint blocksize;
    std::vector<std::string> interpolators = {"linear", "cubic"};
    int16_t *quant_inds;
    uint64_t quant_index = 0;
    Quantizer quantizer;
    size_t num_elements;
    std::array<size_t, N> original_dimensions;
    std::array<size_t, N> original_dim_offsets;
    std::vector<std::array<int, N>> dim_sequences;
    int direction_sequence_id;
    size_t anchor_stride = 0;
    int radius = 32768;
    double real_eb = 1;
    double real_ebx2_r = 1;
    double real_ebx2 = 1;

    double eb_alpha = -1;
    double eb_beta = -1;
    double eb_ratio = 0.5;  // To be deprecated

    static constexpr size_t AVX_256_parallelism = 32 / sizeof(T);
    size_t max_dim = 1;

    // plane-major batched z-gather scratch (4 lanes x NB rows); raw buffer, grown on demand
    T *z_batch_buf = nullptr;
    size_t z_batch_cap = 0;
    T *interp_buffer_1 = nullptr;
    T *interp_buffer_2 = nullptr;
    T *interp_buffer_3 = nullptr;
    T *interp_buffer_4 = nullptr;
    T *pred_buffer = nullptr;
    size_t buffer_len = 1024;
#ifdef __AVX2__
    // for avx
    __m256d radius_avx;
    __m256d nradius_avx;
    __m256d zero_avx_d;
    __m256d ebx2_r_avx;
    __m256d ebx2_avx;

    // for float
    __m256 rel_eb_avx_f;
    __m256 nrel_eb_avx_f;

    __m256 nradius_avx_f;

    // for double
    __m256d rel_eb_avx_d;
    __m256d nrel_eb_avx_d;
#endif
#ifdef __ARM_FEATURE_SVE2
    int SVE2_parallelism;
#endif
    //std::vector<int> visited;
};

template <TUNING Tuning, class T, uint N, class Quantizer>
InterpolationDecomposition<Tuning, T, N, Quantizer> make_decomposition_interpolation(const Config &conf, Quantizer quantizer) {
    return InterpolationDecomposition<Tuning, T, N, Quantizer>(conf, quantizer);
}
}  // namespace Falcons
#include "Interpolation_quantizer.inl"
#endif
