/**
 * @file InterpolationDecomposition_Omp.hpp
 * @author Bing Lu
 * @brief Falcons parallel interpolation decomposition.
 *
 * Copyright (c) 2026 Bing Lu.
 */

#ifdef _OPENMP

#ifndef Falcons_INTERPOLATION_DECOMPOSITION_OMP_HPP
#define Falcons_INTERPOLATION_DECOMPOSITION_OMP_HPP
#if defined(__linux__)
#include <sys/mman.h>
#endif
#include <omp.h>
#include <cmath>
#include <cstring>
#include <type_traits>
#include "Decomposition.hpp"
#include "Falcons/def.hpp"
#include "Falcons/quantizer/Quantizer.hpp"
#include "Falcons/quantizer/Quantizer_Omp.hpp"
#include "Falcons/utils/Config.hpp"
#include "Falcons/utils/FileUtil.hpp"
#include "Falcons/utils/Interpolators.hpp"
#include "Falcons/utils/Iterator.hpp"
#include "Falcons/utils/MemoryUtil.hpp"
#include "Falcons/utils/Timer.hpp"
#include "Falcons/utils/BlockwiseIterator.hpp"

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __ARM_FEATURE_SVE2
#include <arm_sve.h>
#endif

namespace Falcons {
template <class T, uint N, class QuantizerOMP>
class InterpolationDecomposition_OMP : public concepts::DecompositionInterface_OMP<T, int16_t, N> {

    public:
    InterpolationDecomposition_OMP(const Config &conf, QuantizerOMP quantizer) : quantizer(quantizer) {
        static_assert(std::is_base_of<concepts::QuantizerOMPInterface<T, int>, QuantizerOMP>::value,
                      "must implement the quantizer interface");
    }

    T *decompress(const Config &conf, int16_t** local_quant_inds_vec, T *dec_data) override {
#ifdef Falcons_PRINT_TIMINGS
        Timer timer(true);
#endif
        init();

        auto default_nThreads = omp_get_max_threads();
        //std::cout<<"max threads: "<<default_nThreads<<std::endl;

        legacy_stream = conf.falconsDataVer != versionInt(Falcons_DATA_VER);
        quantizer.set_legacy_unfused(legacy_stream);

        size_t max_usable_threads = default_nThreads;
        for (uint i = 1; i < N; ++i)
            max_usable_threads = std::min(max_usable_threads, original_dimensions[i]);
        omp_set_num_threads(max_usable_threads);

        nThreads = actual_team_size();
        //std::cout<<"used threads: "<<nThreads<<std::endl;

        if constexpr (N != 1) {
#ifdef __ARM_FEATURE_SVE2
            buffer_len = max_dim +  2 * SVE2_parallelism - max_dim % SVE2_parallelism;
#else
            buffer_len =  (max_dim + 2 * AVX_256_parallelism - max_dim % AVX_256_parallelism);
#endif
            size_t total_buffer_len = buffer_len * nThreads;
            interp_buffer_1 = new T[total_buffer_len];
            interp_buffer_2 = new T[total_buffer_len];
            interp_buffer_3 = new T[total_buffer_len];
            interp_buffer_4 = new T[total_buffer_len];
        }

        // pred_buffer = new T[total_buffer_len];   // [保留] pred_buffer 当前未被读取
#ifdef Falcons_PRINT_TIMINGS
        timer.stop("init");
        timer.start();
#endif
        // #pragma omp parallel for   // zero-fill unnecessary (bit-exact verified, see non-OMP version)
        // for(size_t i =0;i<total_buffer_len;++i)
        //     /* pred_buffer[i] = */ interp_buffer_1[i] = interp_buffer_2[i] = interp_buffer_3[i] = interp_buffer_4[i] = T(0);
#ifdef Falcons_PRINT_TIMINGS
        timer.stop("buffer init");
        timer.start();
#endif

        // this->quant_inds = quant_inds;
        double eb = quantizer.get_eb();
        //visited.resize(num_elements);
        // int** local_quant_inds_vec = new int*[nThreads];
        // quantizer.init_local_unpred(nThreads, each_num);
        CacheLineInt* local_quant_index_vec = new CacheLineInt[nThreads];
        // auto each_num = (num_elements / nThreads) << 1;
        #pragma omp parallel for
        for (size_t i = 0; i < nThreads; ++i) {
            // local_quant_inds_vec[i] = new int[each_num];
            local_quant_index_vec[i].value = 0;
        }
        local_quant_inds = local_quant_inds_vec;
        local_quant_index = local_quant_index_vec;
        quantizer.reset_local_unpred_idx(nThreads);  // rewind per-thread outlier read cursors (shared with compress)
        // std::cout << "pass" << std::endl;
        auto start_level = interp_level;
        if (anchor_stride == 0) {                                               // check whether used anchor points
            *dec_data = quantizer.recover2(0, local_quant_inds[0][0], 0);  // root code from thread 0, slot 0
            local_quant_index[0].value = 1;                                 // thread 0's level codes follow the root
        } else {
            recover_anchor_grid2(dec_data);  // recover anchor points, not needed because because all outliers were previously unpacked.
            start_level--;
        }
#ifdef Falcons_PRINT_TIMINGS
        timer.stop("other");
        timer.start();
#endif
        for (int level = start_level; level > 0; level--) {
            // break;
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
                for (uint i = 0; i < N; ++i) {
                    end_idx[i] += interp_block_size;
                    if (end_idx[i] > original_dimensions[i] - 1) {
                        end_idx[i] = original_dimensions[i] - 1;
                    }
                }
                auto recover_and_save = [&](size_t idx, T &d, T pred, int tid) {
                    d = quantizer.recover2(pred, local_quant_inds[tid][local_quant_index[tid].value++], tid);
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
#ifdef Falcons_PRINT_TIMINGS
        timer.stop("interp decomp");
#endif
        quantizer.postdecompress_data();
        // std::cout << "Decompressed quantization indices: " << local_quant_index[0].value <<std::endl;


        if constexpr (N != 1) {
            delete [] interp_buffer_1;
            delete [] interp_buffer_2;
            delete [] interp_buffer_3;
            delete [] interp_buffer_4;
        }
        // delete [] pred_buffer;   // [保留] pred_buffer 当前未被读取

        omp_set_num_threads(default_nThreads);
        return dec_data;
    }

    // compress given the error bound
    std::tuple<int16_t**, CacheLineInt*> compress(const Config &conf, T *data) override {
#ifdef Falcons_PRINT_TIMINGS
        Timer timer(true);
#endif
        std::copy_n(conf.dims.begin(), N, original_dimensions.begin());

        interp_id = conf.interpAlgo;
        direction_sequence_id = conf.interpDirection;
        anchor_stride = conf.interpAnchorStride;
        // blocksize = 1024000;  // a empirical value. Can be very large but not helpful
        eb_alpha = conf.interpAlpha;
        eb_beta = conf.interpBeta;

        init();
        blocksize = max_dim;
        auto default_nThreads = omp_get_max_threads();
        //std::cout<<"max threads: "<<default_nThreads<<std::endl;

        size_t max_usable_threads = default_nThreads;
        for (uint i = 1; i < N; ++i)
            max_usable_threads = std::min(max_usable_threads, original_dimensions[i]);
        omp_set_num_threads(max_usable_threads);

        nThreads = actual_team_size();
        if constexpr (N != 1) {
#ifdef __ARM_FEATURE_SVE2
            buffer_len = max_dim +  2 * SVE2_parallelism - max_dim % SVE2_parallelism;
#else
            buffer_len =  (max_dim + 2 * AVX_256_parallelism - max_dim % AVX_256_parallelism) ; // 32 / sizeof(T);
#endif
            size_t total_buffer_len = buffer_len * nThreads;
            interp_buffer_1 = new T[total_buffer_len];
            interp_buffer_2 = new T[total_buffer_len];
            interp_buffer_3 = new T[total_buffer_len];
            interp_buffer_4 = new T[total_buffer_len];
        }
#ifdef Falcons_PRINT_TIMINGS
        timer.stop("init0 time: " );
        timer.start();
#endif
        // pred_buffer = new T[total_buffer_len];   // [保留] pred_buffer 当前未被读取
        // #pragma omp parallel for   // zero-fill unnecessary (bit-exact verified, see non-OMP version)
        // for(size_t i =0; i < total_buffer_len; ++i) {
        //     /* pred_buffer[i] = */ interp_buffer_1[i] = interp_buffer_2[i] = interp_buffer_3[i] = interp_buffer_4[i] = T(0);
        // }
#ifdef Falcons_PRINT_TIMINGS
        timer.stop("buffer init" );
        timer.start();
#endif
        int16_t** local_quant_inds_vec = new int16_t*[nThreads];
        CacheLineInt* local_quant_index_vec = new CacheLineInt[nThreads];

        // A schedule can load one thread far beyond the average when an outer dimension is short.
        // Per sweep a thread gets at most pts/T + pts/c points for an outer loop of c iterations,
        // and summed over levels the pts/c part stays below 8 * num / (shortest dimension).
        size_t shortest_dim = *std::min_element(original_dimensions.begin(), original_dimensions.end());
        size_t each_num = std::min(num_elements, (num_elements / nThreads) * 4 + 8 * (num_elements / shortest_dim) + 256);
        quantizer.init_local_unpred(nThreads, each_num);

        constexpr size_t alignment = 64;          // cache line alignment
        constexpr size_t elems_per_cacheline = alignment / sizeof(int16_t);
        // +1 extra cacheline of slack per thread: store_quant4/8 write step (up to 8) int16
        // at a time, so the final SIMD store of a region must not overflow into the next
        // thread's region (or past the buffer). Mirrors the serial "+16 padding" on quant_inds.
        size_t padded_each_num = (((each_num + elems_per_cacheline - 1) / elems_per_cacheline) + 1) * elems_per_cacheline;
        total_quant_inds = new (std::align_val_t(alignment)) int16_t[nThreads * padded_each_num];
#if defined(__linux__)
        // THP on this array pays off only when each thread's quant slice is large enough that
        // TLB relief outweighs the huge-page first-touch/setup cost. Measured: the hint wins by
        // 5-15% while the per-thread slice (~num_elements/nThreads * 2B) stays >= 16 MiB, but
        // regresses as it shrinks (+7% on 512^3 at 64 threads / 4 MiB; +30% on a 2D 1800x3600
        // field at 32 threads / 0.4 MiB). Gate on the per-thread footprint so it self-enables
        // for large slices (few threads or big data) and self-disables for small ones, where it
        // would only add setup overhead. 16 MiB == 8 huge pages worth; safely above the highest
        // measured losing slice (~8 MiB).
        if ((num_elements / static_cast<size_t>(nThreads)) * sizeof(int16_t) >= (static_cast<size_t>(16) << 20)) {
            uintptr_t _mb = reinterpret_cast<uintptr_t>(total_quant_inds) & ~static_cast<uintptr_t>(4095);
            madvise(reinterpret_cast<void *>(_mb),
                    nThreads * padded_each_num * sizeof(int16_t) + (reinterpret_cast<uintptr_t>(total_quant_inds) - _mb),
                    MADV_HUGEPAGE);
        }
#endif

        #pragma omp parallel for
        for (size_t i = 0; i < nThreads; ++i) {
            local_quant_inds_vec[i] = total_quant_inds + i * padded_each_num;
            local_quant_index_vec[i].value = 0;
        }

        // #pragma omp parallel for
        // for (size_t i = 0; i < nThreads; ++i) {
        //     if (i == 0)
        //         local_quant_inds_vec[i] = new int[each_num << 1];
        //     else local_quant_inds_vec[i] = new int[each_num];
        //     local_quant_index_vec[i].value = 0;
        // }
        local_quant_inds = local_quant_inds_vec;
        local_quant_index = local_quant_index_vec;

        double eb = quantizer.get_eb();
        auto start_level = interp_level;
        if (anchor_stride == 0) {  // check whether to use anchor points
            local_quant_inds[0][0] = quantizer.quantize_and_overwrite2(*data, 0, 0);  // root code -> thread 0, slot 0 (serialized with the level codes)
            local_quant_index[0].value = 1;                                           // thread 0's level codes follow the root
        } else {
            build_anchor_grid2(data);  // losslessly saving anchor points
            start_level--;
        }

        // [pure RLE+FSE] no per-thread frequency histogram: the mode is the zero-error
        // code (radius) known a priori and there is no Huffman fallback, so the
        // quant-value statistics that used to feed Huffman are not collected at all.

#ifdef Falcons_PRINT_TIMINGS
        timer.stop("init2 time: " );
        timer.start();
#endif
        for (int level = start_level; level > 0; level--) {
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
                for (uint i = 0; i < N; ++i) {
                    end_idx[i] += interp_block_size;
                    if (end_idx[i] > original_dimensions[i] - 1) {
                        end_idx[i] = original_dimensions[i] - 1;
                    }
                }

                auto quantize_and_save = [&](size_t idx, T &d, T pred, int tid) {
                        int quant_val = quantizer.quantize_and_overwrite2(d, pred, tid);
                        local_quant_inds[tid][local_quant_index[tid].value++] = quant_val;
                        // (quantizer.quantize_and_overwrite2(d, pred, tid));
                        // quant_inds[idx] = (quantizer.quantize_and_overwrite(d, pred, idx));
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
#ifdef Falcons_PRINT_TIMINGS
        timer.stop("Pure Interpolation time: " );
        timer.start();
#endif
        quantizer.set_eb(eb);
        quantizer.postcompress_data();
        if constexpr (N != 1) {
            delete [] interp_buffer_1;
            delete [] interp_buffer_2;
            delete [] interp_buffer_3;
            delete [] interp_buffer_4;
        }
        // delete [] pred_buffer;   // [保留] pred_buffer 当前未被读取

        omp_set_num_threads(default_nThreads);
#ifdef Falcons_PRINT_TIMINGS
        timer.stop("other: " );
#endif
        return {local_quant_inds_vec, local_quant_index_vec};
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

    void save2(uchar *&c)  {
        write(original_dimensions.data(), N, c);
        write(blocksize, c);
        write(interp_id, c);
        write(direction_sequence_id, c);
        write(anchor_stride, c);
        write(eb_alpha, c);
        write(eb_beta, c);


        quantizer.save2(c);
    }

    void save3(uchar *&c, int tid)  {
        quantizer.save3(c, tid);
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

    void load2(const uchar *&c, size_t &remaining_length) {
        read(original_dimensions.data(), N, c, remaining_length);
        read(blocksize, c, remaining_length);
        read(interp_id, c, remaining_length);
        read(direction_sequence_id, c, remaining_length);
        read(anchor_stride, c, remaining_length);
        read(eb_alpha, c, remaining_length);
        read(eb_beta, c, remaining_length);
        quantizer.load2(c, remaining_length);
    }

    void load3(const uchar *&c, size_t &remaining_length, int tid) {
        quantizer.load3(c, remaining_length, tid);
    }

    void init_local_unpred(int nThreads) {
        quantizer.init_local_unpred(nThreads, 0);
    }
    std::pair<int, int> get_out_range() override { return quantizer.get_out_range(); }

    int num_threads() const { return static_cast<int>(nThreads); }
    size_t local_unpred_count(int tid) const { return quantizer.local_unpred_count(tid); }

   private:
    // The team OpenMP really delivers, which can be below the request (e.g. OMP_THREAD_LIMIT).
    // Fixing it here keeps the stream layout equal to the team that fills it.
    static size_t actual_team_size() {
        int team = 1;
        #pragma omp parallel
        #pragma omp single
        team = omp_get_num_threads();
        omp_set_num_threads(team);
        return team;
    }

    template <COMPMODE CompMode, bool SkipOverwrite>
    ALWAYS_INLINE void quantize_point(size_t idx, T &d, T pred, int tid) {
        (void)idx;
        if constexpr (CompMode == COMPMODE::COMP) {
            int quant_val;
            if constexpr (SkipOverwrite) {
                quant_val = quantizer.quantize_without_overwrite2(d, pred, tid);
            } else {
                quant_val = quantizer.quantize_and_overwrite2(d, pred, tid);
            }
            local_quant_inds[tid][local_quant_index[tid].value++] = quant_val;
        } else {
            d = quantizer.recover2(pred, local_quant_inds[tid][local_quant_index[tid].value++], tid);
        }
    }

    void init() {

        radius = quantizer.get_out_range().second / 2;
        //assert(blocksize % 2 == 0 && "Interpolation block size should be even numbers");
        assert((anchor_stride & anchor_stride - 1) == 0 && "Anchor stride should be 0 or 2's exponentials");
        num_elements = 1;
        interp_level = -1;
	    bool use_anchor = false;
        max_dim = 1;
        for (uint i = 0; i < N; ++i) {
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
#ifdef __ARM_FEATURE_SVE2
        SVE2_parallelism = svcntb() / sizeof(T);
#endif
#ifdef __AVX2__
        radius_avx = _mm256_set1_pd(radius);
        nradius_avx = _mm256_set1_pd(-radius);
        zero_avx_d = _mm256_set1_pd(0);


        if constexpr (std::is_same_v<T, float>) {
            nradius_avx_f = _mm256_set1_ps(-radius);   // int16 escape sentinel = -radius = -32768
        }
#endif
        original_dim_offsets[N - 1] = 1;
        for (int i = N - 2; i >= 0; --i) {
            original_dim_offsets[i] = original_dim_offsets[i + 1] * original_dimensions[i + 1];
        }

        dim_sequences = std::vector<std::array<int, N>>();
        auto sequence = std::array<int, N>();
        for (uint i = 0; i < N; ++i) {
            sequence[i] = i;
        }
        do {
            dim_sequences.push_back(sequence);
        } while (std::next_permutation(sequence.begin(), sequence.end()));
        /*
        if constexpr (N==3){

            auto d_size = original_dimensions;
            reduced_dim_offsets.resize(interp_level );
            level_prefix.resize(interp_level , 0);

            int level = 0;
            while(level < interp_level){
                //grid_leaps[level][0] = 1;
                reduced_dim_offsets[level][2] = 1;
                reduced_dim_offsets[level][1] = d_size[2];
                reduced_dim_offsets[level][0] = d_size[1] * d_size[2];




                if(level + 1 < interp_level ){
                    d_size[0] = (d_size[0] + 1) >> 1;
                    d_size[1] = (d_size[1] + 1) >> 1;
                    d_size[2] = (d_size[2] + 1) >> 1;
                    level_prefix[level] = d_size[0] *  d_size[1] * d_size[2];
                }
                ++level;
            }
        }*/



    }

    void build_anchor_grid(T *data) {  // store anchor points. steplength: anchor_stride on each dimension
        std::array<size_t, N> strides;
        std::array<size_t, N> begins{0};
        std::fill(strides.begin(), strides.end(), anchor_stride);
        foreach_omp
            <T, N>(data, 0, begins, original_dimensions, strides, original_dim_offsets,
                   [&](T *d) { auto idx = d - data; quantizer.save_unpred( *d, idx);});
    }

    void build_anchor_grid2(T *data) {  // store anchor points. steplength: anchor_stride on each dimension
        std::array<size_t, N> strides;
        std::array<size_t, N> begins{0};
        std::fill(strides.begin(), strides.end(), anchor_stride);
        foreach_omp2
            <T, N>(data, 0, begins, original_dimensions, strides, original_dim_offsets,
                   [&](T *d, int tid) {quantizer.save_unpred2(*d, tid);}); //local_quant_inds[0][local_quant_index[0].value++] =
    }

    void recover_anchor_grid(T *data) {  // recover anchor points. steplength: anchor_stride on each dimension
        std::array<size_t, N> strides;
        std::array<size_t, N> begins{0};
        std::fill(strides.begin(), strides.end(), anchor_stride);
        foreach_omp
            <T, N>(data, 0, begins, original_dimensions, strides, original_dim_offsets, [&](T *d) {
                *d = quantizer.recover_unpred(); //d - data
                // local_quant_index[0].value++;
                //quant_index++;
            });
    }

    void recover_anchor_grid2(T *data) {  // recover anchor points. steplength: anchor_stride on each dimension
        std::array<size_t, N> strides;
        std::array<size_t, N> begins{0};
        std::fill(strides.begin(), strides.end(), anchor_stride);
        foreach_omp2
            <T, N>(data, 0, begins, original_dimensions, strides, original_dim_offsets, [&](T *d, int tid) {
                *d = quantizer.recover_unpred2(tid); //d - data
                // local_quant_index[0].value++;
                //quant_index++;
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
        // std::cout << "Successfully run to this inte0." <<std::endl;
        if (interp_func == "linear" || n < 5) {
            // if (pb == PB_predict_overwrite) {
            #pragma omp parallel for
            for (size_t i = 1; i < n - 1; i += 2) {
                auto tid = omp_get_thread_num();
                T *d = data + begin + i * stride;
                quantize_func(d - data, *d, interp_linear(*(d - stride), *(d + stride)), tid);
            }
            if (n % 2 == 0) {
                T *d = data + begin + (n - 1) * stride;
                if (n < 4) {
                    quantize_func(d - data, *d, *(d - stride), 0);
                } else {
                    quantize_func(d - data, *d, interp_linear1(*(d - stride3x), *(d - stride)), 0);
                }
            }
            // }
        } else {
            T *d;
            size_t i;
            #pragma omp parallel for
            for (i = 3; i < n - 3; i += 2) {
                auto tid = omp_get_thread_num();
                d = data + begin + i * stride;
                quantize_func(d - data, *d,
                              interp_cubic(*(d - stride3x), *(d - stride), *(d + stride), *(d + stride3x)), tid);
            }
            d = data + begin + stride;
            quantize_func(d - data, *d, interp_quad_1(*(d - stride), *(d + stride), *(d + stride3x)), 0);
            i = n % 2 == 0 ? n - 3 : n - 2;
            d = data + begin + i * stride;
            quantize_func(d - data, *d, interp_quad_2(*(d - stride3x), *(d - stride), *(d + stride)), 0);
            if (n % 2 == 0) {
                d = data + begin + (n - 1) * stride;
                quantize_func(d - data, *d, interp_quad_3(*(d - stride5x), *(d - stride3x), *(d - stride)), 0);
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
        size_t stride2x = 2 * stride;
        if (interp_func == "linear") {
            begins[direction] = 1;
            ends[direction] = n - 1;
            strides[direction] = 2;
            foreach_omp2
                <T, N>(data, offset, begins, ends, strides, dim_offsets,
                       [&](T *d, int tid) { quantize_func(d - data, *d, interp_linear(*(d - stride), *(d + stride)), tid); });
            if (n % 2 == 0) {
                begins[direction] = n - 1;
                ends[direction] = n;
                foreach_omp2 //todo: this is infficient when direction = 0
                    <T, N>(data, offset, begins, ends, strides, dim_offsets, [&](T *d, int tid) {
                        if (n < 3)
                            quantize_func(d - data, *d, *(d - stride), tid);
                        else
                            quantize_func(d - data, *d, interp_linear1(*(d - stride2x), *(d - stride)), tid);
                    });
            }
        } else {
            size_t stride3x = 3 * stride;
            size_t i_start = 3;
            begins[direction] = i_start;
            ends[direction] = (n >= 3) ? (n - 3) : 0;
            strides[direction] = 2;
            foreach_omp2
                <T, N>(data, offset, begins, ends, strides, dim_offsets, [&](T *d, int tid) {
                    quantize_func(d - data, *d,
                                  interp_cubic(*(d - stride3x), *(d - stride), *(d + stride), *(d + stride3x)), tid);
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

                foreach_omp2 //todo: this is infficient when direction = 0
                    <T, N>(data, offset, begins, ends, strides, dim_offsets, [&](T *d, int tid) {
                        if (boundary >= 3) {
                            if (boundary + 3 < n)
                                quantize_func(
                                    d - data, *d,
                                    interp_cubic(*(d - stride3x), *(d - stride), *(d + stride), *(d + stride3x)), tid);
                            else if (boundary + 1 < n)
                                quantize_func(d - data, *d,
                                              interp_quad_2(*(d - stride3x), *(d - stride), *(d + stride)), tid);
                            else
                                quantize_func(d - data, *d, interp_linear1(*(d - stride3x), *(d - stride)), tid);
                        } else {
                            if (boundary + 3 < n)
                                quantize_func(d - data, *d,
                                              interp_quad_1(*(d - stride), *(d + stride), *(d + stride3x)), tid);

                            else if (boundary + 1 < n)

                                quantize_func(d - data, *d, interp_linear(*(d - stride), *(d + stride)), tid);

                            else
                                quantize_func(d - data, *d, *(d - stride), tid);

                        }
                    });
            }
        }
        return predict_error;
    }

    // template<typename U = T, typename = std::enable_if_t<std::is_same_v<U, float>>>
    // ALWAYS_INLINE void quantize_1D_float (__m256& sum, __m256& ori_avx, __m256& quant_avx, T tmp[8]) {

    //     __m256d quant_avx_low  = _mm256_cvtps_pd(_mm256_castps256_ps128(quant_avx));
    //     quant_avx_low  = _mm256_round_pd(_mm256_mul_pd(quant_avx_low,  ebx2_r_avx),  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    //     __m256d mask_low = _mm256_and_pd(
    //         _mm256_cmp_pd(quant_avx_low, nradius_avx, _CMP_GT_OQ),
    //         _mm256_cmp_pd(quant_avx_low, radius_avx, _CMP_LT_OQ)
    //     );
    //     quant_avx_low = _mm256_blendv_pd(zero_avx_d, quant_avx_low, mask_low);

    //     __m256d quant_avx_high = _mm256_cvtps_pd(_mm256_extractf128_ps(quant_avx, 1));
    //     quant_avx_high = _mm256_round_pd(_mm256_mul_pd(quant_avx_high, ebx2_r_avx), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    //     __m256d mask_high = _mm256_and_pd(
    //         _mm256_cmp_pd(quant_avx_high, nradius_avx, _CMP_GT_OQ),
    //         _mm256_cmp_pd(quant_avx_high, radius_avx, _CMP_LT_OQ)
    //     );
    //     quant_avx_high = _mm256_blendv_pd(zero_avx_d, quant_avx_high, mask_high);

    //     // dequantization for decompression
    //     __m256d decompressed_low = _mm256_fmadd_pd(quant_avx_low, ebx2_avx,
    //                             _mm256_cvtps_pd(_mm256_castps256_ps128(sum)));
    //     __m256d decompressed_high = _mm256_fmadd_pd(quant_avx_high, ebx2_avx,
    //                             _mm256_cvtps_pd(_mm256_extractf128_ps(sum, 1)));

    //     quant_avx = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(quant_avx_low)),
    //                     _mm256_cvtpd_ps(quant_avx_high), 1);

    //     __m256 decompressed = _mm256_insertf128_ps(
    //         _mm256_castps128_ps256(_mm256_cvtpd_ps(decompressed_low)),
    //         _mm256_cvtpd_ps(decompressed_high), 1);

    //     __m256 err_dequan = _mm256_sub_ps(decompressed, ori_avx);
    //     quant_avx = _mm256_add_ps(quant_avx, radius_avx_f);

    //     _mm256_storeu_ps(tmp, decompressed);

    //     __m256 mask = _mm256_and_ps(
    //             _mm256_cmp_ps(err_dequan, nrel_eb_avx_f, _CMP_GE_OQ),
    //             _mm256_cmp_ps(err_dequan, rel_eb_avx_f, _CMP_LE_OQ)
    //     );

    //     quant_avx = _mm256_blendv_ps(zero_avx_f, quant_avx, mask);
    // }


    // template<typename U = T, typename = std::enable_if_t<std::is_same_v<U, double>>>
    // ALWAYS_INLINE void quantize_1D_double (__m256d& sum, __m256d& ori_avx, __m256d& quant_avx, T tmp[4]) {
    //     quant_avx = _mm256_round_pd(_mm256_mul_pd(quant_avx,  ebx2_r_avx),  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    //     __m256d mask = _mm256_and_pd(
    //         _mm256_cmp_pd(quant_avx, nradius_avx, _CMP_GT_OQ),
    //         _mm256_cmp_pd(quant_avx, radius_avx, _CMP_LT_OQ)
    //     );
    //     quant_avx = _mm256_blendv_pd(zero_avx_d, quant_avx, mask);

    //     __m256d decompressed = _mm256_fmadd_pd(quant_avx, ebx2_avx, sum);
    //     _mm256_storeu_pd(tmp, decompressed);
    //     __m256d err_dequan = _mm256_sub_pd(decompressed, ori_avx);

    //     mask = _mm256_and_pd(
    //             _mm256_cmp_pd(err_dequan, nrel_eb_avx_d, _CMP_GE_OQ),
    //             _mm256_cmp_pd(err_dequan, rel_eb_avx_d, _CMP_LE_OQ)
    //     );
    //     quant_avx = _mm256_add_pd(quant_avx,  radius_avx);
    //     quant_avx = _mm256_blendv_pd(zero_avx_d, quant_avx, mask);
    // }


    // template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    // ALWAYS_INLINE void interp_linear_and_quantize_1D(const T * buf, const size_t &len, T* data,
    //     size_t&  offset, size_t& cur_ij_offset, int& tid, QuantizeFunc &&quantize_func) {
    //     if(len == 1)
    //         return;

    //     auto odd_len = len / 2;
    //     auto even_len = len - odd_len;
    //     size_t i = 0;

    //     if constexpr (std::is_same_v<T, float>) {
    //         const size_t step = AVX_256_parallelism;
    //         const __m256 factor = _mm256_set1_ps(0.5f);

    //         for (; i + 1  <= even_len; i += step) { // 3 is not AVX_256_parallelism - 1 !!
    //             // predict
    //             __m256 va = _mm256_loadu_ps(buf + i );
    //             __m256 vb = _mm256_loadu_ps(buf + i + 1);
    //             __m256 sum = _mm256_add_ps(va, vb);
    //             sum = _mm256_mul_ps(sum, factor);

    //             // quantize
    //             size_t start = (i << 1) + 1;
    //             // i = k / 2;

    //             if constexpr (CompMode == COMPMODE::COMP) {
    //                 T ori[8];
    //                 size_t base = start * offset;
    //                 size_t offsetx2 = offset << 1;

    //                 ori[0] = data[base];
    //                 ori[1] = data[base + offsetx2];
    //                 ori[2] = data[base + (offsetx2 << 1)];
    //                 ori[3] = data[base + 3 * offsetx2];
    //                 ori[4] = data[base + (offsetx2 << 2)];
    //                 ori[5] = data[base + 5 * offsetx2];
    //                 ori[6] = data[base + 6 * offsetx2];
    //                 ori[7] = data[base + 7 * offsetx2];


    //                 __m256 ori_avx = _mm256_loadu_ps(ori);
    //                 __m256 quant_avx = _mm256_sub_ps(ori_avx, sum); // prediction error
    //                 float tmp[8];
    //                 quantize_1D_float(sum, ori_avx, quant_avx, tmp);
    //                 int quant_vals[8];
    //                 __m256i quant_avx_i = _mm256_cvtps_epi32(quant_avx);
    //                 _mm256_storeu_si256(reinterpret_cast<__m256i*>(quant_vals), quant_avx_i);
    //                 size_t j = 0;
    //                 #pragma unroll
    //                 for ( ; j < step && i + j + 1 < odd_len; ++j) {
    //                     if (quant_vals[j] != 0)
    //                         data[(start + (j << 1)) * offset] = tmp[j];
    //                     else
    //                         quantizer.save_unpred2(ori[j], tid);
    //
    //                 }
    //                 _mm256_storeu_si256(
    //                     reinterpret_cast<__m256i*>(local_quant_inds[tid] + local_quant_index[tid].value),
    //                     quant_avx_i
    //                 );
    //                 local_quant_index[tid].value += j;
    //             }
    //             else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
    //                 __m256i quant_avx_i = _mm256_loadu_si256(
    //                     reinterpret_cast<__m256i*>(local_quant_inds[tid] + local_quant_index[tid].value));
    //                 int quant_vals[8];
    //                 _mm256_storeu_si256(reinterpret_cast<__m256i*>(quant_vals), quant_avx_i);
    //                 quant_avx_i = _mm256_sub_epi32(quant_avx_i, radius_avx_256i);

    //                 __m256d decompressed_low  = _mm256_cvtepi32_pd(_mm256_castsi256_si128(quant_avx_i));
    //                 decompressed_low = _mm256_mul_pd(decompressed_low, ebx2_avx);

    //                 __m256d decompressed_high = _mm256_cvtepi32_pd(_mm256_extracti128_si256(quant_avx_i, 1));
    //                 decompressed_high = _mm256_mul_pd(decompressed_high, ebx2_avx);

    //                  __m256 decompressed = _mm256_insertf128_ps(
    //                     _mm256_castps128_ps256(_mm256_cvtpd_ps(decompressed_low)),
    //                     _mm256_cvtpd_ps(decompressed_high), 1);
    //                 decompressed = _mm256_add_ps(decompressed, sum);
    //                 float tmp[8];
    //                 _mm256_storeu_ps(tmp, decompressed);

    //                 size_t j = 0;
    //                 for ( ; j < step && i + j + 1< odd_len; ++j) {
    //                     if (quant_vals[j] != 0)
    //                         data[(start + (j << 1)) * offset] = tmp[j];
    //                     else
    //                         data[(start + (j << 1)) * offset] = quantizer.recover_unpred2(tid);
    //                 }
    //                 local_quant_index[tid].value += j;
    //             }
    //         }
    //     }
    //     else if constexpr (std::is_same_v<T, double>) {
    //         const size_t step = AVX_256_parallelism;
    //         const __m256d factor = _mm256_set1_pd(0.5);

    //         for (; i + 1 <= even_len; i += step) { // 3 is not AVX_256_parallelism - 1 !!
    //             __m256d va = _mm256_loadu_pd(buf + i);
    //             __m256d vb = _mm256_loadu_pd(buf + i + 1);

    //             __m256d sum = _mm256_add_pd(va, vb);
    //             sum = _mm256_mul_pd(sum, factor);

    //             size_t start = (i << 1) + 1;
    //             if constexpr (CompMode == COMPMODE::COMP) {
    //                 T ori[4];
    //                 size_t base = start * offset;
    //                 size_t offsetx2 = offset << 1;

    //                 ori[0] = data[base];
    //                 ori[1] = data[base + offsetx2];
    //                 ori[2] = data[base + (offsetx2 << 1)];
    //                 ori[3] = data[base + 3 * offsetx2];

    //                 __m256d ori_avx = _mm256_loadu_pd(ori);
    //                 __m256d quant_avx = _mm256_sub_pd(ori_avx, sum); // prediction error
    //                 T tmp[4];
    //                 quantize_1D_double(sum, ori_avx, quant_avx, tmp);

    //                 int quant_vals[4];
    //                 __m128i quant_avx_i = _mm256_cvtpd_epi32(quant_avx);

    //                 _mm_storeu_si128(reinterpret_cast<__m128i*>(quant_vals), quant_avx_i);
    //                 size_t j = 0;
    //                 for ( ; j < step && i + j + 1 < odd_len; ++j) {
    //                     if (quant_vals[j] != 0)
    //                         data[(start + (j << 1)) * offset] = tmp[j];
    //                     else
    //                         quantizer.save_unpred2(ori[j], tid);
    //
    //                 }
    //                 _mm_storeu_si128(
    //                     reinterpret_cast<__m128i*>(local_quant_inds[tid] + local_quant_index[tid].value),
    //                     quant_avx_i
    //                 );
    //                 local_quant_index[tid].value += j;
    //             }
    //             else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
    //                 __m128i quant_avx_i = _mm_loadu_si128(
    //                     reinterpret_cast<__m128i*>(local_quant_inds[tid] + local_quant_index[tid].value));
    //                 int quant_vals[4];
    //                 _mm_storeu_si128(reinterpret_cast<__m128i*>(quant_vals), quant_avx_i);
    //                 quant_avx_i = _mm_sub_epi32(quant_avx_i, radius_avx_128i);

    //                 __m256d decompressed = _mm256_fmadd_pd(_mm256_cvtepi32_pd(quant_avx_i),
    //                                         ebx2_avx, sum);
    //                 T tmp[4];
    //                 _mm256_storeu_pd(tmp, decompressed);

    //                 size_t j = 0;
    //                 for ( ; j < step && i + j + 1 < odd_len; ++j) {
    //                     if (quant_vals[j] != 0)
    //                         data[(start + (j << 1)) * offset] = tmp[j];
    //                     else
    //                         data[(start + (j << 1)) * offset] = quantizer.recover_unpred2(tid);
    //                 }
    //                 local_quant_index[tid].value += j;
    //             }

    //         }
    //     }
    //     T pred_edge;
    //     if(len < 3 )
    //         pred_edge = buf[even_len - 1];
    //     else
    //         pred_edge = interp_linear1(buf[even_len - 2], buf[even_len - 1]);
    //     int last = 2 * odd_len - 1;
    //     quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset , data[last * offset], pred_edge, tid);
    // }


    // template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    // ALWAYS_INLINE void interp_cubic_and_quantize_1D(const T * buf, const size_t &len, T* data,
    //     size_t&  offset, size_t& cur_ij_offset, int& tid, QuantizeFunc &&quantize_func) {
    //    // assert(len <= max_dim);
    //     if(len == 1)
    //         return;

    //     auto odd_len = len / 2;
    //     auto even_len = len - odd_len;

    //     T pred_first;
    //     if(even_len < 2)
    //         pred_first = (buf[0]);
    //     else if(even_len < 3)
    //         pred_first = interp_linear(buf[0], buf[1]);
    //     else
    //         pred_first = interp_quad_1(buf[0], buf[1], buf[2]);
    //     quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + offset , data[offset], pred_first, tid);

    //     size_t i = 0;
    //     if constexpr (std::is_same_v<T, float>) {
    //         const size_t step = AVX_256_parallelism;
    //         const __m256 nine  = _mm256_set1_ps(9.0f);
    //         const __m256 factor = _mm256_set1_ps(1.0f / 16.0f);

    //         for (; i + 3  <= even_len; i += step) { // 3 is not AVX_256_parallelism - 1 !!
    //             __m256 va = _mm256_loadu_ps(buf + i);
    //             __m256 vb = _mm256_loadu_ps(buf + i + 1);
    //             __m256 vc = _mm256_loadu_ps(buf + i + 2);
    //             __m256 vd = _mm256_loadu_ps(buf + i + 3);

    //              __m256 sum = _mm256_add_ps(vb, vc);
    //              sum = _mm256_mul_ps(sum, nine);
    //              sum = _mm256_sub_ps(sum, va);
    //             sum = _mm256_sub_ps(sum, vd);
    //             sum = _mm256_mul_ps(sum, factor);

    //             size_t start = (i << 1) + 3;

    //             if constexpr (CompMode == COMPMODE::COMP) {
    //                 T ori[8];
    //                 size_t base = start * offset;
    //                 size_t offsetx2 = offset << 1;

    //                 ori[0] = data[base];
    //                 ori[1] = data[base + offsetx2];
    //                 ori[2] = data[base + (offsetx2 << 1)];
    //                 ori[3] = data[base + 3 * offsetx2];
    //                 ori[4] = data[base + (offsetx2 << 2)];
    //                 ori[5] = data[base + 5 * offsetx2];
    //                 ori[6] = data[base + 6 * offsetx2];
    //                 ori[7] = data[base + 7 * offsetx2];


    //                 __m256 ori_avx = _mm256_loadu_ps(ori);
    //                 __m256 quant_avx = _mm256_sub_ps(ori_avx, sum); // prediction error
    //                 float tmp[8];
    //                 quantize_1D_float(sum, ori_avx, quant_avx, tmp);

    //                 int quant_vals[8];
    //                 __m256i quant_avx_i = _mm256_cvtps_epi32(quant_avx);
    //                 _mm256_storeu_si256(reinterpret_cast<__m256i*>(quant_vals), quant_avx_i);
    //                 size_t j = 0;
    //                 for ( ; j < step && i + j + 3 < even_len; ++j) {
    //                     if (quant_vals[j] != 0)
    //                         data[(start + (j << 1)) * offset] = tmp[j];
    //                     else
    //                         quantizer.save_unpred2(ori[j], tid);
    //
    //                 }

    //                 _mm256_storeu_si256(
    //                     reinterpret_cast<__m256i*>(local_quant_inds[tid] + local_quant_index[tid].value),
    //                     quant_avx_i
    //                 );
    //                 local_quant_index[tid].value += j;
    //             }
    //             else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
    //                 __m256i quant_avx_i = _mm256_loadu_si256(
    //                     reinterpret_cast<__m256i*>(local_quant_inds[tid] + local_quant_index[tid].value));
    //                 int quant_vals[8];
    //                 _mm256_storeu_si256(reinterpret_cast<__m256i*>(quant_vals), quant_avx_i);
    //                 quant_avx_i = _mm256_sub_epi32(quant_avx_i, radius_avx_256i);

    //                 __m256d decompressed_low  = _mm256_cvtepi32_pd(_mm256_castsi256_si128(quant_avx_i));
    //                 decompressed_low = _mm256_mul_pd(decompressed_low, ebx2_avx);

    //                 __m256d decompressed_high = _mm256_cvtepi32_pd(_mm256_extracti128_si256(quant_avx_i, 1));
    //                 decompressed_high = _mm256_mul_pd(decompressed_high, ebx2_avx);

    //                  __m256 decompressed = _mm256_insertf128_ps(
    //                     _mm256_castps128_ps256(_mm256_cvtpd_ps(decompressed_low)),
    //                     _mm256_cvtpd_ps(decompressed_high), 1);
    //                 decompressed = _mm256_add_ps(decompressed, sum);
    //                 float tmp[8];
    //                 _mm256_storeu_ps(tmp, decompressed);

    //                 size_t j = 0;
    //                 for ( ; j < step && i + j + 3 < even_len; ++j) {
    //                     if (quant_vals[j] != 0)
    //                         data[(start + (j << 1)) * offset] = tmp[j];
    //                     else
    //                         data[(start + (j << 1)) * offset] = quantizer.recover_unpred2(tid);
    //                 }
    //                 local_quant_index[tid].value += j;
    //             }
    //         }
    //     }
    //     else if constexpr (std::is_same_v<T, double>) {
    //         const size_t step = AVX_256_parallelism;
    //         const __m256d nine  = _mm256_set1_pd(9.0);
    //         const __m256d factor = _mm256_set1_pd(1.0 / 16.0);

    //         for (; i + 3 <= even_len; i += step) { // 3 is not AVX_256_parallelism - 1 !!
    //             __m256d va = _mm256_loadu_pd(buf + i);
    //             __m256d vb = _mm256_loadu_pd(buf + i + 1);
    //             __m256d vc = _mm256_loadu_pd(buf + i + 2);
    //             __m256d vd = _mm256_loadu_pd(buf + i + 3);

    //             __m256d sum = _mm256_add_pd(vb, vc);
    //              sum = _mm256_mul_pd(sum, nine);
    //              sum = _mm256_sub_pd(sum, va);
    //             sum = _mm256_sub_pd(sum, vd);
    //             sum = _mm256_mul_pd(sum, factor);
    //             // _mm256_storeu_pd(p + i + 1, sum);
    //             size_t start = (i << 1) + 3;
    //             // T pred[4];
    //             // _mm256_storeu_pd(pred, sum);

    //             if constexpr (CompMode == COMPMODE::COMP) {
    //                 T ori[4];
    //                 size_t base = start * offset;
    //                 size_t offsetx2 = offset << 1;

    //                 ori[0] = data[base];
    //                 ori[1] = data[base + offsetx2];
    //                 ori[2] = data[base + (offsetx2 << 1)];
    //                 ori[3] = data[base + 3 * offsetx2];

    //                 __m256d ori_avx = _mm256_loadu_pd(ori);
    //                 __m256d quant_avx = _mm256_sub_pd(ori_avx, sum); // prediction error
    //                 T tmp[4];
    //                 quantize_1D_double(sum, ori_avx, quant_avx, tmp);

    //                 int quant_vals[4];
    //                 __m128i quant_avx_i = _mm256_cvtpd_epi32(quant_avx);

    //                 _mm_storeu_si128(reinterpret_cast<__m128i*>(quant_vals), quant_avx_i);
    //                 size_t j = 0;
    //                 for ( ; j < step && i + j + 3 < even_len; ++j) {
    //                     if (quant_vals[j] != 0)
    //                         data[(start + (j << 1)) * offset] = tmp[j];
    //                     else
    //                         quantizer.save_unpred2(ori[j], tid);
    //
    //                 }
    //                 _mm_storeu_si128(
    //                     reinterpret_cast<__m128i*>(local_quant_inds[tid] + local_quant_index[tid].value),
    //                     quant_avx_i
    //                 );
    //                 local_quant_index[tid].value += j;
    //             }
    //             else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
    //                 __m128i quant_avx_i = _mm_loadu_si128(
    //                 reinterpret_cast<__m128i*>(local_quant_inds[tid] + local_quant_index[tid].value));
    //                 int quant_vals[4];
    //                 _mm_storeu_si128(reinterpret_cast<__m128i*>(quant_vals), quant_avx_i);
    //                 quant_avx_i = _mm_sub_epi32(quant_avx_i, radius_avx_128i);

    //                 __m256d decompressed = _mm256_fmadd_pd(_mm256_cvtepi32_pd(quant_avx_i),
    //                                         ebx2_avx, sum);
    //                 T tmp[4];
    //                 _mm256_storeu_pd(tmp, decompressed);

    //                 size_t j = 0;
    //                 for ( ; j < step && i + j + 3 < even_len; ++j) {
    //                     if (quant_vals[j] != 0)
    //                         data[(start + (j << 1)) * offset] = tmp[j];
    //                     else
    //                         data[(start + (j << 1)) * offset] = quantizer.recover_unpred2(tid);
    //                 }
    //                 local_quant_index[tid].value += j;
    //             }
    //         }
    //     }
    //     if(odd_len > 1){
    //         if(odd_len < even_len){//the only boundary is p[len- 1]
    //             //odd_len < even_len so even_len > 2
    //             T edge_pred;
    //             edge_pred = interp_quad_2(buf[even_len - 3], buf[even_len - 2], buf[even_len - 1]);
    //             int last = 2 * odd_len - 1;
    //             quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[last * offset], edge_pred, tid);

    //         }
    //         else{//the boundary points are is p[len -2 ] and p[len -1 ]
    //             T edge_pred;
    //             if(odd_len > 2){ //len - 2
    //              //odd_len = even_len so even_len > 2
    //                 edge_pred = interp_quad_2(buf[even_len - 3],  buf[even_len - 2], buf[even_len - 1]);
    //                 int last = 2 * odd_len - 3;
    //                 quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[last * offset], edge_pred, tid);
    //             }
    //             //len -1
    //             //odd_len = even_len so even_len > 1
    //                 edge_pred = interp_linear1(buf[even_len - 2], buf[even_len - 1]);
    //                 int last = 2 * odd_len - 1;
    //                 quantize_point<CompMode, SkipOverwrite>(cur_ij_offset + last * offset, data[last * offset], edge_pred, tid);


    //         }
    //     }
    // }


    // template <COMPMODE CompMode>
    // ALWAYS_INLINE void interp_linear_and_quantize(const T * a, const T* b, size_t &len, T* data,
    //     size_t& offset, size_t& cur_ij_offset, int tid) {
    //     size_t i = 0;

    //     if constexpr (std::is_same_v<T, float>) {
    //         constexpr size_t step = AVX_256_parallelism;
    //         const __m256 factor = _mm256_set1_ps(0.5f);

    //         for (; i  < len; i += step) {
    //             __m256 va = _mm256_loadu_ps(a + i);
    //             __m256 vb = _mm256_loadu_ps(b + i);

    //             __m256 sum = _mm256_add_ps(va, vb);
    //             sum = _mm256_mul_ps(sum, factor);

    //             // _mm256_storeu_ps(p + i, sum);
    //             quantize_float<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
    //         }
    //     }
    //     else if constexpr (std::is_same_v<T, double>) {
    //         constexpr size_t step = AVX_256_parallelism;
    //         const __m256d factor = _mm256_set1_pd(0.5);

    //         for (; i  < len; i += step) {
    //             __m256d va = _mm256_loadu_pd(a + i);
    //             __m256d vb = _mm256_loadu_pd(b + i);

    //             __m256d sum = _mm256_add_pd(va, vb);
    //             sum = _mm256_mul_pd(sum, factor);
    //             // _mm256_storeu_pd(p + i, sum);
    //             // size_t start = i;
    //             quantize_double<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
    //         }
    //     }

    // }

    // template <COMPMODE CompMode>
    // ALWAYS_INLINE void interp_cubic_and_quantize(const T * a, const T* b, T* c, T*d, size_t &len, T* data,
    //     size_t& offset, size_t& cur_ij_offset, int tid) {

    //     size_t i = 0;

    //     if constexpr (std::is_same_v<T, float>) {
    //         constexpr size_t step = AVX_256_parallelism;
    //         const __m256 nine  = _mm256_set1_ps(9.0f);
    //         const __m256 factor = _mm256_set1_ps(1.0f / 16.0f);

    //         for (; i  < len; i += step) {
    //             __m256 va = _mm256_loadu_ps(a + i);
    //             __m256 vb = _mm256_loadu_ps(b + i);
    //             __m256 vc = _mm256_loadu_ps(c + i);
    //             __m256 vd = _mm256_loadu_ps(d + i);

    //              __m256 sum = _mm256_add_ps(vb, vc);
    //              sum = _mm256_mul_ps(sum, nine);
    //              sum = _mm256_sub_ps(sum, va);
    //             sum = _mm256_sub_ps(sum, vd);
    //             sum = _mm256_mul_ps(sum, factor);

    //             // _mm256_storeu_ps(p + i, sum);
    //             quantize_float<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
    //         }

    //     }
    //     else if constexpr (std::is_same_v<T, double>) {
    //         constexpr size_t step = AVX_256_parallelism;
    //         const __m256d nine  = _mm256_set1_pd(9.0);
    //         const __m256d factor = _mm256_set1_pd(1.0 / 16.0);

    //         for (; i  < len; i += step) {
    //             __m256d va = _mm256_loadu_pd(a + i);
    //             __m256d vb = _mm256_loadu_pd(b + i);
    //             __m256d vc = _mm256_loadu_pd(c + i);
    //             __m256d vd = _mm256_loadu_pd(d + i);

    //             __m256d sum = _mm256_add_pd(vb, vc);
    //              sum = _mm256_mul_pd(sum, nine);
    //              sum = _mm256_sub_pd(sum, va);
    //             sum = _mm256_sub_pd(sum, vd);

    //             sum = _mm256_mul_pd(sum, factor);
    //             // _mm256_storeu_pd(p + i, sum);
    //             quantize_double<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
    //         }
    //     }

    // }

    // template <COMPMODE CompMode>
    // ALWAYS_INLINE void interp_equal_and_quantize(const T * a, size_t &len, T* data,
    //     size_t& offset, size_t& cur_ij_offset, int tid) {

    //     size_t i = 0;

    //     if constexpr (std::is_same_v<T, float>) {
    //         constexpr size_t step = AVX_256_parallelism;
    //         for (; i  < len; i += step) {
    //             __m256 sum = _mm256_loadu_ps(a + i);
    //             // _mm256_storeu_ps(p + i, sum);
    //             quantize_float<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
    //         }

    //     }
    //     else if constexpr (std::is_same_v<T, double>) {
    //         constexpr size_t step = AVX_256_parallelism;
    //         for (; i  < len; i += step) {
    //             __m256d sum = _mm256_loadu_pd(a + i);
    //             quantize_double<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
    //         }
    //     }
    // }


    // template <COMPMODE CompMode>
    // ALWAYS_INLINE void interp_linear1_and_quantize(const T * a, const T* b, size_t &len, T* data,
    //     size_t& offset, size_t& cur_ij_offset, int tid) {
    //     size_t i = 0;

    //     if constexpr (std::is_same_v<T, float>) {
    //         constexpr size_t step = AVX_256_parallelism;
    //         const __m256 factor = _mm256_set1_ps(0.5f);
    //         const __m256 three = _mm256_set1_ps(3.0f);
    //         for (; i  < len; i += step) {
    //             __m256 vb = _mm256_loadu_ps(b + i);
    //             __m256 va = _mm256_loadu_ps(a + i);
    //             vb= _mm256_mul_ps(vb, three);
    //             __m256 sum = _mm256_sub_ps(vb, va);
    //             sum = _mm256_mul_ps(sum, factor);

    //             // _mm256_storeu_ps(p + i, sum);
    //             quantize_float<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
    //         }
    //     }
    //     else if constexpr (std::is_same_v<T, double>) {
    //         constexpr size_t step = AVX_256_parallelism;
    //         const __m256d factor = _mm256_set1_pd(0.5);
    //         const __m256d three = _mm256_set1_pd(3.0);
    //         for (; i  < len; i += step) {
    //             __m256d va = _mm256_loadu_pd(a + i);
    //             __m256d vb = _mm256_loadu_pd(b + i);
    //             va = _mm256_mul_pd(va, three);
    //             __m256d sum = _mm256_sub_pd(vb, va);
    //             sum = _mm256_mul_pd(sum, factor);
    //             // _mm256_storeu_pd(p + i, sum);
    //             // size_t start = i;
    //             quantize_double<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
    //         }
    //     }
    // }

    // template <COMPMODE CompMode>
    // ALWAYS_INLINE void interp_quad1_and_quantize(const T * a, const T* b, const T* c, size_t &len, T* data,
    //     size_t& offset, size_t& cur_ij_offset, int tid) {
    //     size_t i = 0;
    //     if constexpr (std::is_same_v<T, float>) {
    //         constexpr size_t step = AVX_256_parallelism;
    //         const __m256 factor = _mm256_set1_ps(0.125f);
    //         const __m256 six = _mm256_set1_ps(6.0f);
    //         const __m256 three = _mm256_set1_ps(3.0f);

    //         for (; i  < len; i += step) {
    //             __m256 va = _mm256_loadu_ps(a + i);
    //             va = _mm256_mul_ps(va, three);
    //             __m256 vb = _mm256_loadu_ps(b + i);
    //             __m256 vc = _mm256_loadu_ps(c + i);
    //             vb = _mm256_fmsub_ps(vb, six, vc);
    //             __m256 sum = _mm256_add_ps(va, vb);
    //             sum = _mm256_mul_ps(sum, factor);

    //             // _mm256_storeu_ps(p + i, sum);
    //             quantize_float<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
    //         }
    //     }
    //     else if constexpr (std::is_same_v<T, double>) {
    //         constexpr size_t step = AVX_256_parallelism;
    //         const __m256d factor = _mm256_set1_pd(0.125);
    //         const __m256d six = _mm256_set1_pd(6.0);
    //         const __m256d three = _mm256_set1_pd(3.0);

    //         for (; i  < len; i += step) {
    //             __m256d va = _mm256_loadu_pd(a + i);
    //             __m256d vb = _mm256_loadu_pd(b + i);
    //             va = _mm256_mul_pd(va, three);
    //             __m256d vc = _mm256_loadu_pd(c + i);
    //             vb = _mm256_fmsub_pd(vb, six, vc);
    //             __m256d sum = _mm256_add_pd(va, vb);
    //             sum = _mm256_mul_pd(sum, factor);
    //             // _mm256_storeu_pd(p + i, sum);
    //             // size_t start = i;
    //             quantize_double<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
    //         }
    //     }

    // }

    // template <COMPMODE CompMode>
    // ALWAYS_INLINE void interp_quad2_and_quantize(const T * a, const T* b, const T* c, size_t &len, T* data,
    //     size_t& offset, size_t& cur_ij_offset, int tid) {
    //     size_t i = 0;
    //     if constexpr (std::is_same_v<T, float>) {
    //         constexpr size_t step = AVX_256_parallelism;
    //         const __m256 factor = _mm256_set1_ps(0.125f);
    //         const __m256 six = _mm256_set1_ps(6.0f);
    //         const __m256 three = _mm256_set1_ps(3.0f);

    //         for (; i  < len; i += step) {
    //             __m256 va = _mm256_loadu_ps(a + i);
    //             __m256 vc = _mm256_loadu_ps(c + i);
    //             vc = _mm256_mul_ps(vc, three);
    //             __m256 vb = _mm256_loadu_ps(b + i);

    //             vb = _mm256_fmsub_ps(vb, six, va);
    //             __m256 sum = _mm256_add_ps(vc, vb);
    //             sum = _mm256_mul_ps(sum, factor);

    //             // _mm256_storeu_ps(p + i, sum);
    //             quantize_float<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
    //         }
    //     }
    //     else if constexpr (std::is_same_v<T, double>) {
    //         constexpr size_t step = AVX_256_parallelism;
    //         const __m256d factor = _mm256_set1_pd(0.125);
    //         const __m256d six = _mm256_set1_pd(6.0);
    //         const __m256d three = _mm256_set1_pd(3.0);

    //         for (; i  < len; i += step) {
    //             __m256d vc = _mm256_loadu_pd(c + i);
    //             __m256d va = _mm256_loadu_pd(a + i);
    //             vc = _mm256_mul_pd(vc, three);
    //             __m256d vb = _mm256_loadu_pd(b + i);

    //             vb = _mm256_fmsub_pd(vb, six, va);
    //             __m256d sum = _mm256_add_pd(vc, vb);
    //             sum = _mm256_mul_pd(sum, factor);
    //             // _mm256_storeu_pd(p + i, sum);
    //             // size_t start = i;
    //             quantize_double<CompMode, step, false, SkipOverwrite>(sum, i, data, offset, len, tid);
    //         }
    //     }

    // }

    // template <COMPMODE CompMode, int step, typename U = T, typename = std::enable_if_t<std::is_same_v<U, float>>>
    // ALWAYS_INLINE void quantize_float (__m256 sum, size_t& start, T*& data, size_t& offset, size_t& len, int tid) {
    //     if constexpr (CompMode == COMPMODE::COMP) {
    //         T ori[8] = {
    //             data[(start) * offset],
    //             data[(start + 1) * offset],
    //             data[(start + 2) * offset],
    //             data[(start + 3) * offset],
    //             data[(start + 4) * offset],
    //             data[(start + 5) * offset],
    //             data[(start + 6) * offset],
    //             data[(start + 7) * offset]
    //         };
    //         __m256 ori_avx = _mm256_loadu_ps(ori);
    //         __m256 quant_avx = _mm256_sub_ps(ori_avx, sum); // prediction error
    //         // calculate quantization code
    //         __m256d quant_avx_low  = _mm256_cvtps_pd(_mm256_castps256_ps128(quant_avx));
    //         quant_avx_low  = _mm256_round_pd(_mm256_mul_pd(quant_avx_low,  ebx2_r_avx),  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    //         __m256d mask_low = _mm256_and_pd(
    //             _mm256_cmp_pd(quant_avx_low, nradius_avx, _CMP_GT_OQ),
    //             _mm256_cmp_pd(quant_avx_low, radius_avx, _CMP_LT_OQ)
    //         );
    //         quant_avx_low = _mm256_blendv_pd(zero_avx_d, quant_avx_low, mask_low);

    //         __m256d quant_avx_high = _mm256_cvtps_pd(_mm256_extractf128_ps(quant_avx, 1));
    //         quant_avx_high = _mm256_round_pd(_mm256_mul_pd(quant_avx_high, ebx2_r_avx), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    //         __m256d mask_high = _mm256_and_pd(
    //             _mm256_cmp_pd(quant_avx_high, nradius_avx, _CMP_GT_OQ),
    //             _mm256_cmp_pd(quant_avx_high, radius_avx, _CMP_LT_OQ)
    //         );
    //         quant_avx_high = _mm256_blendv_pd(zero_avx_d, quant_avx_high, mask_high);

    //         // dequantization for decompression
    //         __m256d decompressed_low = _mm256_fmadd_pd(quant_avx_low, ebx2_avx,
    //                                 _mm256_cvtps_pd(_mm256_castps256_ps128(sum)));
    //         __m256d decompressed_high = _mm256_fmadd_pd(quant_avx_high, ebx2_avx,
    //                                 _mm256_cvtps_pd(_mm256_extractf128_ps(sum, 1)));

    //         quant_avx = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(quant_avx_low)),
    //                         _mm256_cvtpd_ps(quant_avx_high), 1);

    //         __m256 decompressed = _mm256_insertf128_ps(
    //             _mm256_castps128_ps256(_mm256_cvtpd_ps(decompressed_low)),
    //             _mm256_cvtpd_ps(decompressed_high), 1);

    //         __m256 err_dequan = _mm256_sub_ps(decompressed, ori_avx);
    //         quant_avx = _mm256_add_ps(quant_avx, radius_avx_f);
    //         float tmp[8];
    //         _mm256_storeu_ps(tmp, decompressed);

    //         __m256 mask = _mm256_and_ps(
    //                 _mm256_cmp_ps(err_dequan, nrel_eb_avx_f, _CMP_GE_OQ),
    //                 _mm256_cmp_ps(err_dequan, rel_eb_avx_f, _CMP_LE_OQ)
    //         );

    //         quant_avx = _mm256_blendv_ps(zero_avx_f, quant_avx, mask);
    //         // float verify[8];
    //         // _mm256_storeu_ps(verify, quant_avx);
    //         int quant_vals[8];
    //         __m256i quant_avx_i = _mm256_cvtps_epi32(quant_avx);
    //         _mm256_storeu_si256(reinterpret_cast<__m256i*>(quant_vals), quant_avx_i);
    //         size_t j = 0;
    //         #pragma unroll
    //         for ( ; j < step && start + j < len; ++j) {
    //             if (quant_vals[j] != 0)
    //                 data[(start + j) * offset] = tmp[j];
    //             else
    //                 quantizer.save_unpred2(ori[j], tid);
    //
    //         }

    //         _mm256_storeu_si256(
    //             reinterpret_cast<__m256i*>(local_quant_inds[tid] + local_quant_index[tid].value),
    //             quant_avx_i
    //         );
    //         local_quant_index[tid].value += j;
    //     }
    //     else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
    //         __m256i quant_avx_i = _mm256_loadu_si256(
    //             reinterpret_cast<__m256i*>(local_quant_inds[tid] + local_quant_index[tid].value));
    //         int quant_vals[8];
    //         _mm256_storeu_si256(reinterpret_cast<__m256i*>(quant_vals), quant_avx_i);
    //         quant_avx_i = _mm256_sub_epi32(quant_avx_i, radius_avx_256i);

    //         __m256d decompressed_low  = _mm256_cvtepi32_pd(_mm256_castsi256_si128(quant_avx_i));
    //         decompressed_low = _mm256_mul_pd(decompressed_low, ebx2_avx);

    //         __m256d decompressed_high = _mm256_cvtepi32_pd(_mm256_extracti128_si256(quant_avx_i, 1));
    //         decompressed_high = _mm256_mul_pd(decompressed_high, ebx2_avx);

    //             __m256 decompressed = _mm256_insertf128_ps(
    //             _mm256_castps128_ps256(_mm256_cvtpd_ps(decompressed_low)),
    //             _mm256_cvtpd_ps(decompressed_high), 1);
    //         decompressed = _mm256_add_ps(decompressed, sum);
    //         float tmp[8];
    //         _mm256_storeu_ps(tmp, decompressed);

    //         size_t j = 0;
    //         #pragma unroll
    //         for ( ; j < step && start + j < len; ++j) {
    //             if (quant_vals[j] != 0)
    //                 data[(start + j) * offset] = tmp[j];
    //             else
    //                 data[(start + j) * offset] = quantizer.recover_unpred2(tid);
    //         }
    //         local_quant_index[tid].value += j;
    //     }
    // }

    // template <COMPMODE CompMode, int step, typename U = T, typename = std::enable_if_t<std::is_same_v<U, double>>>
    // ALWAYS_INLINE void quantize_double (__m256d sum, size_t& start, T*& data, size_t& offset, size_t& len, int tid) {
    //     if constexpr (CompMode == COMPMODE::COMP) {
    //         T ori[4] = {
    //             data[(start) * offset],
    //             data[(start + 1) * offset],
    //             data[(start + 2) * offset],
    //             data[(start + 3) * offset],
    //         };
    //         __m256d ori_avx = _mm256_loadu_pd(ori);
    //         __m256d quant_avx = _mm256_sub_pd(ori_avx, sum); // prediction error
    //         quant_avx = _mm256_round_pd(_mm256_mul_pd(quant_avx,  ebx2_r_avx),  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    //         __m256d mask = _mm256_and_pd(
    //             _mm256_cmp_pd(quant_avx, nradius_avx, _CMP_GT_OQ),
    //             _mm256_cmp_pd(quant_avx, radius_avx, _CMP_LT_OQ)
    //         );
    //         quant_avx = _mm256_blendv_pd(zero_avx_d, quant_avx, mask);

    //         __m256d decompressed = _mm256_fmadd_pd(quant_avx, ebx2_avx, sum);
    //         T tmp[4];
    //         _mm256_storeu_pd(tmp, decompressed);
    //         __m256d err_dequan = _mm256_sub_pd(decompressed, ori_avx);

    //         mask = _mm256_and_pd(
    //                 _mm256_cmp_pd(err_dequan, nrel_eb_avx_d, _CMP_GE_OQ),
    //                 _mm256_cmp_pd(err_dequan, rel_eb_avx_d, _CMP_LE_OQ)
    //         );
    //         quant_avx = _mm256_add_pd(quant_avx,  radius_avx);
    //         quant_avx = _mm256_blendv_pd(zero_avx_d, quant_avx, mask);

    //         int quant_vals[4];
    //         __m128i quant_avx_i = _mm256_cvtpd_epi32(quant_avx);

    //         _mm_storeu_si128(reinterpret_cast<__m128i*>(quant_vals), quant_avx_i);
    //         size_t j = 0;
    //         #pragma unroll
    //         for ( ; j < step && start + j < len; ++j) {
    //             if (quant_vals[j] != 0)
    //                 data[(start + j) * offset] = tmp[j];
    //             else
    //                 quantizer.save_unpred2(ori[j], tid);
    //         }
    //         _mm_storeu_si128(
    //             reinterpret_cast<__m128i*>(local_quant_inds[tid] + local_quant_index[tid].value),
    //             quant_avx_i
    //         );
    //         local_quant_index[tid].value += j;
    //     }
    //     else if constexpr (CompMode == COMPMODE::DECOMP) { // decomp
    //         __m128i quant_avx_i = _mm_loadu_si128(
    //             reinterpret_cast<__m128i*>(local_quant_inds[tid] + local_quant_index[tid].value));
    //         int quant_vals[4];
    //         _mm_storeu_si128(reinterpret_cast<__m128i*>(quant_vals), quant_avx_i);
    //         quant_avx_i = _mm_sub_epi32(quant_avx_i, radius_avx_128i);

    //         __m256d decompressed = _mm256_fmadd_pd(_mm256_cvtepi32_pd(quant_avx_i),
    //                                 ebx2_avx, sum);
    //         T tmp[4];
    //         _mm256_storeu_pd(tmp, decompressed);

    //         size_t j = 0;
    //         #pragma unroll
    //         for ( ; j < step && start + j < len; ++j) {
    //             if (quant_vals[j] != 0)
    //                 data[(start + j) * offset] = tmp[j];
    //             else
    //                 data[(start + j) * offset] = quantizer.recover_unpred2(tid);
    //         }
    //         local_quant_index[tid].value += j;
    //     }
    // }

    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void interp_linear_and_quantize_1D(const T * buf, const size_t &len, T* data,
        size_t&  offset, size_t& cur_ij_offset, int& tid, QuantizeFunc &&quantize_func);

    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void interp_cubic_and_quantize_1D(const T * buf, const size_t &len, T* data,
        size_t&  offset, size_t& cur_ij_offset, int& tid, QuantizeFunc &&quantize_func);

    template <COMPMODE CompMode, bool EnableInnerOmp = false, bool SkipOverwrite = false, class QuantizeFunc>
    ALWAYS_INLINE void interp_linear_and_quantize_1D_line(T *data, const size_t &len,
        size_t& offset, size_t& cur_ij_offset, int& tid, QuantizeFunc &&quantize_func);

    template <COMPMODE CompMode, bool EnableInnerOmp = false, bool SkipOverwrite = false, class QuantizeFunc>
    ALWAYS_INLINE void interp_cubic_and_quantize_1D_line(T *data, const size_t &len,
        size_t& offset, size_t& cur_ij_offset, int& tid, QuantizeFunc &&quantize_func);

    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void interp_linear_and_quantize(const T * a, const T* b, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func);

    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void interp_cubic_and_quantize(const T * a, const T* b, T* c, T*d, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func);

    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void interp_equal_and_quantize(const T * a, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func);

    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void interp_linear1_and_quantize(const T * a, const T* b, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func);

    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void interp_quad1_and_quantize(const T * a, const T* b, const T* c, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func);

    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    ALWAYS_INLINE void interp_quad2_and_quantize(const T * a, const T* b, const T* c, size_t &len, T* data,
        size_t& offset, size_t& cur_ij_offset, int tid, QuantizeFunc &&quantize_func);

#ifdef __AVX2__
    template<typename U = T, typename = std::enable_if_t<std::is_same_v<U, float>>>
    ALWAYS_INLINE void quantize_1D_float (__m256& sum, __m256& ori_avx, __m256& quant_avx, T tmp[8]);

    template<typename U = T, typename = std::enable_if_t<std::is_same_v<U, double>>>
    ALWAYS_INLINE void quantize_1D_double (__m256d& sum, __m256d& ori_avx, __m256d& quant_avx, T tmp[4]);

    template <COMPMODE CompMode, int step, bool FullOnly = false, bool SkipOverwrite = false, typename U = T, typename = std::enable_if_t<std::is_same_v<U, float>>>
    ALWAYS_INLINE void quantize_float (__m256& sum, size_t& start, T*& data, size_t& offset, size_t& len, int tid);

    template <COMPMODE CompMode, int step, bool FullOnly = false, bool SkipOverwrite = false, typename U = T, typename = std::enable_if_t<std::is_same_v<U, double>>>
    ALWAYS_INLINE void quantize_double (__m256d& sum, size_t& start, T*& data, size_t& offset, size_t& len, int tid);
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
        const size_t& step, svbool_t& pg, svbool_t& pg64, int tid);

    template <COMPMODE CompMode, bool FullOnly = false, bool SkipOverwrite = false, typename U = T, typename = std::enable_if_t<std::is_same_v<U, double>>>
    ALWAYS_INLINE void quantize_double (svfloat64_t& sum, size_t& start, T*& data, size_t& offset, size_t& len,
        const size_t& step, svbool_t& pg64, int tid);

#endif
    // void interp_cubic(const T * a,const T * b,const T * c,const T * d,T * p, const size_t &len){
    //    // assert(len <= max_dim);
    //      constexpr bool is_float  = std::is_same_v<T, float>;
    //     constexpr bool is_double = std::is_same_v<T, double>;

    //     size_t i = 0;

    //     if constexpr (is_float) {
    //         const size_t step = AVX_256_parallelism;
    //         const __m256 nine  = _mm256_set1_ps(9.0f);
    //         const __m256 factor = _mm256_set1_ps(1.0f / 16.0f);

    //         for (; i  < len; i += step) {
    //             __m256 va = _mm256_loadu_ps(a + i);
    //             __m256 vb = _mm256_loadu_ps(b + i);
    //             __m256 vc = _mm256_loadu_ps(c + i);
    //             __m256 vd = _mm256_loadu_ps(d + i);

    //              __m256 sum = _mm256_add_ps(vb, vc);
    //              sum = _mm256_mul_ps(sum, nine);
    //              sum = _mm256_sub_ps(sum, va);
    //             sum = _mm256_sub_ps(sum, vd);



    //             sum = _mm256_mul_ps(sum, factor);

    //             _mm256_storeu_ps(p + i, sum);
    //         }
    //     }
    //     else if constexpr (is_double) {
    //         const size_t step = AVX_256_parallelism;
    //         const __m256d nine  = _mm256_set1_pd(9.0);
    //         const __m256d factor = _mm256_set1_pd(1.0 / 16.0);

    //         for (; i  < len; i += step) {
    //             __m256d va = _mm256_loadu_pd(a + i);
    //             __m256d vb = _mm256_loadu_pd(b + i);
    //             __m256d vc = _mm256_loadu_pd(c + i);
    //             __m256d vd = _mm256_loadu_pd(d + i);

    //             __m256d sum = _mm256_add_pd(vb, vc);
    //              sum = _mm256_mul_pd(sum, nine);
    //              sum = _mm256_sub_pd(sum, va);
    //             sum = _mm256_sub_pd(sum, vd);



    //             sum = _mm256_mul_pd(sum, factor);
    //             _mm256_storeu_pd(p + i, sum);
    //         }
    //     }
    //     /*
    //     for (; i < len; i++) {
    //         p[i] = (-a[i] + T(9) * b[i] + T(9) * c[i] - d[i]) / T(16);
    //     }*/


    // }

    // void interp_linear(const T * a,const T * b,T * p, const size_t &len){
    //     // assert(len <= max_dim);

    //     size_t i = 0;
    //     // double ebx2_r = quantizer.get_double_eb_reciprocal();
    //     if constexpr (std::is_same_v<T, float>) {
    //         const size_t step = AVX_256_parallelism;
    //         const __m256 factor = _mm256_set1_ps(0.5f);
    //         for (; i  < len; i += step) {
    //             __m256 va = _mm256_loadu_ps(a + i);
    //             __m256 vb = _mm256_loadu_ps(b + i);

    //             __m256 sum = _mm256_add_ps(va, vb);
    //             sum = _mm256_mul_ps(sum, factor);

    //             _mm256_storeu_ps(p + i, sum);
    //         }
    //     }
    //     else if constexpr (std::is_same_v<T, double>) {
    //         const size_t step = AVX_256_parallelism;
    //         // const __m256d nine  = _mm256_set1_pd(9.0);
    //         const __m256d factor = _mm256_set1_pd(0.5);

    //         for (; i  < len; i += step) {
    //             __m256d va = _mm256_loadu_pd(a + i);
    //             __m256d vb = _mm256_loadu_pd(b + i);

    //             __m256d sum = _mm256_add_pd(va, vb);

    //             sum = _mm256_mul_pd(sum, factor);
    //             _mm256_storeu_pd(p + i, sum);
    //         }
    //     }
    //     /*
    //     for (; i < len; i++) {
    //         p[i] = (-a[i] + T(9) * b[i] + T(9) * c[i] - d[i]) / T(16);
    //     }*/
    // }

    //     void interp_linear_1D(const T * buf, T * p, const size_t &len){
    //    // assert(len <= max_dim);
    //     // constexpr bool is_double = std::is_same_v<T, double>;
    //     if(len == 1)
    //         return;

    //     auto odd_len = len / 2;
    //     auto even_len = len - odd_len;
    //     size_t i = 0;

    //     if constexpr (std::is_same_v<T, float>) {
    //         const size_t step = AVX_256_parallelism;
    //         const __m256 factor = _mm256_set1_ps(0.5f);

    //         for (; i + 1  <= even_len; i += step) { // 3 is not AVX_256_parallelism - 1 !!
    //             __m256 va = _mm256_loadu_ps(buf + i);
    //             __m256 vb = _mm256_loadu_ps(buf + i + 1);


    //              __m256 sum = _mm256_add_ps(va, vb);
    //             sum = _mm256_mul_ps(sum, factor);

    //             _mm256_storeu_ps(p + i, sum);
    //         }
    //     }
    //     else if constexpr (std::is_same_v<T, double>) {
    //         const size_t step = AVX_256_parallelism;
    //         const __m256d factor = _mm256_set1_pd(0.5);

    //         for (; i + 1 <= even_len; i += step) { // 3 is not AVX_256_parallelism - 1 !!
    //             __m256d va = _mm256_loadu_pd(buf + i);
    //             __m256d vb = _mm256_loadu_pd(buf + i + 1);

    //             __m256d sum = _mm256_add_pd(va, vb);
    //             sum = _mm256_mul_pd(sum, factor);
    //             _mm256_storeu_pd(p + i, sum);
    //         }
    //     }
    //     if(len <3 ){
    //         p[odd_len - 1] = buf[even_len - 1];
    //     }
    //     else {
    //         p[odd_len - 1] = interp_linear1(buf[even_len - 2], buf[even_len - 1]);
    //     }
    // }

    // void interp_cubic_1D(const T * buf, T * p, const size_t &len){
    //    // assert(len <= max_dim);
    //      constexpr bool is_float  = std::is_same_v<T, float>;
    //     constexpr bool is_double = std::is_same_v<T, double>;
    //     if(len == 1)
    //         return;

    //     auto odd_len = len / 2;
    //     auto even_len = len - odd_len;

    //     if(even_len < 2)
    //         p[0] = (buf[0]);

    //     else if(even_len < 3)
    //         p[0] = interp_linear(buf[0], buf[1]);
    //     else {
    //         p[0] = interp_quad_1(buf[0], buf[1], buf[2]) ;
    //     }

    //     size_t i = 0;

    //     if constexpr (is_float) {
    //         const size_t step = AVX_256_parallelism;
    //         const __m256 nine  = _mm256_set1_ps(9.0f);
    //         const __m256 factor = _mm256_set1_ps(1.0f / 16.0f);

    //         for (; i + 3  <= even_len; i += step) { // 3 is not AVX_256_parallelism - 1 !!
    //             __m256 va = _mm256_loadu_ps(buf + i);
    //             __m256 vb = _mm256_loadu_ps(buf + i + 1);
    //             __m256 vc = _mm256_loadu_ps(buf + i + 2);
    //             __m256 vd = _mm256_loadu_ps(buf + i + 3);

    //              __m256 sum = _mm256_add_ps(vb, vc);
    //              sum = _mm256_mul_ps(sum, nine);
    //              sum = _mm256_sub_ps(sum, va);
    //             sum = _mm256_sub_ps(sum, vd);
    //             sum = _mm256_mul_ps(sum, factor);

    //             _mm256_storeu_ps(p + i + 1, sum);
    //         }
    //     }
    //     else if constexpr (is_double) {
    //         const size_t step = AVX_256_parallelism;
    //         const __m256d nine  = _mm256_set1_pd(9.0);
    //         const __m256d factor = _mm256_set1_pd(1.0 / 16.0);

    //         for (; i + 3 <= even_len; i += step) { // 3 is not AVX_256_parallelism - 1 !!
    //             __m256d va = _mm256_loadu_pd(buf + i);
    //             __m256d vb = _mm256_loadu_pd(buf + i + 1);
    //             __m256d vc = _mm256_loadu_pd(buf + i + 2);
    //             __m256d vd = _mm256_loadu_pd(buf + i + 3);

    //             __m256d sum = _mm256_add_pd(vb, vc);
    //              sum = _mm256_mul_pd(sum, nine);
    //              sum = _mm256_sub_pd(sum, va);
    //             sum = _mm256_sub_pd(sum, vd);
    //             sum = _mm256_mul_pd(sum, factor);
    //             _mm256_storeu_pd(p + i + 1, sum);
    //         }
    //     }
    //     if(odd_len > 1){
    //         if(odd_len < even_len){//the only boundary is p[len- 1]
    //             //odd_len < even_len so even_len > 2
    //             p[odd_len - 1] = interp_quad_2(buf[even_len - 3], buf[even_len - 2], buf[even_len - 1]);

    //         }
    //         else{//the boundary points are is p[len -2 ] and p[len -1 ]
    //             if(odd_len > 2){ //len - 2
    //              //odd_len = even_len so even_len > 2
    //                 p[odd_len - 2] = interp_quad_2(buf[even_len - 3],  buf[even_len - 2], buf[even_len - 1]);
    //             }
    //             //len -1
    //             //odd_len = even_len so even_len > 1
    //                 p[odd_len - 1] = interp_linear1(buf[even_len - 2], buf[even_len - 1]);


    //         }
    //     }
    //     /*
    //     for (; i < len; i++) {
    //         p[i] = (-a[i] + T(9) * b[i] + T(9) * c[i] - d[i]) / T(16);
    //     }*/
    // }

    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
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
        if (interp_func == "linear") {
            size_t last_linear_stride = (SkipOverwrite ? 3 * stride : 2 * stride);
            begins[direction] = 1;
            ends[direction] = (n >= 1) ? (n - 1) : 0;
            strides[direction] = 2;
            size_t vector_len = ends[1] > begins[1] ? (ends[1]-begins[1]-1)/strides[1] + 1 : 0;
            size_t nthreads = omp_get_max_threads();
            size_t total_iters = (ends[0] - begins[0] + 1) >> 1;

            size_t base = total_iters / nthreads;
            size_t rem  = total_iters % nthreads;
            #pragma omp parallel //num_threads(omp_get_max_threads())
            {
                size_t tid = omp_get_thread_num();

                size_t my_iters = base + (tid < rem);
                size_t iter_start = tid < rem
                    ? tid * (base + 1)
                    : rem * (base + 1) + (tid - rem) * base;
                size_t start_idx = begins[0] + iter_start * strides[0];
                size_t end_idx = start_idx + my_iters * strides[0];


                auto buffer_offset = buffer_len * tid;
                auto cur_buffer_1 = interp_buffer_1 + buffer_offset;
                auto cur_buffer_2 = interp_buffer_2 + buffer_offset;

                if (start_idx < ends[0])
                {
                    auto cur_ij_offset = offset + start_idx * dim_offsets[0];
                    size_t buffer_idx = 0;
                    for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                            auto cur_offset =  cur_ij_offset + k;
                            cur_buffer_1[buffer_idx] = data[cur_offset - stride];
                            cur_buffer_2[buffer_idx] = data[cur_offset + stride];
                            ++buffer_idx;
                    }
                    interp_linear_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1,cur_buffer_2, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, tid, quantize_func);
                    for(size_t i = start_idx + strides[0]; i < end_idx; i += strides[0]) {
                        auto cur_ij_offset = offset + i * dim_offsets[0];
                        size_t buffer_idx = 0;
                        auto temp_buffer = cur_buffer_1;
                        cur_buffer_1 = cur_buffer_2;
                        cur_buffer_2 = temp_buffer;
                        for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                            auto cur_offset =  cur_ij_offset + stride + k;
                            cur_buffer_2[buffer_idx++] = data[cur_offset];
                        }
                        interp_linear_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1,cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[1], cur_ij_offset, tid, quantize_func);
                    }
                }
            }
            // for (size_t j = begins[1]; j < ends[1]; j += strides[1]) {
            // #pragma omp parallel for
            // for(size_t i = begins[0]; i < ends[0]; i += strides[0]){
            //     auto tid = omp_get_thread_num();
            //     auto buffer_offset = buffer_len * tid;
            //     auto cur_buffer_1 = interp_buffer_1 + buffer_offset;
            //     auto cur_buffer_2 = interp_buffer_2 + buffer_offset;
            //     auto cur_ij_offset = offset + i * dim_offsets[0];
            //     size_t buffer_idx = 0;
            //     // if( i == begins[0]) {
            //     for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
            //             auto cur_offset =  cur_ij_offset + k;
            //             cur_buffer_1[buffer_idx] = data[cur_offset - stride];
            //             cur_buffer_2[buffer_idx] = data[cur_offset + stride];
            //             ++buffer_idx;

            //     }
            //     // }
            //     // else{
            //     //     auto temp_buffer = cur_buffer_1;
            //     //     cur_buffer_1 = cur_buffer_2;
            //     //     cur_buffer_2 = temp_buffer;

            //     //     buffer_idx = 0;
            //     //     for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
            //     //         auto cur_offset =  cur_ij_offset + stride + k;
            //     //         cur_buffer_2[buffer_idx++] = data[cur_offset];

            //     //     }

            //     // }

            //     interp_linear_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1,cur_buffer_2, vector_len,
            //         data + cur_ij_offset, strides[1], cur_ij_offset, tid);
            // }

            if (n % 2 == 0) {
                auto cur_ij_offset = offset + (n - 1) * dim_offsets[0];

                if(n < 3) {
                    size_t buffer_idx = 0;
                    auto cur_buffer_2 = interp_buffer_2;
                    for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                        auto cur_offset =  cur_ij_offset + k;
                        cur_buffer_2[buffer_idx] = data[cur_offset - stride];  // fix: was cur_ij_offset-stride+2k (double k), read OOB
                        ++buffer_idx;
                    }
                    interp_equal_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, 0, quantize_func);
                }

                else {
                    size_t buffer_idx = 0;
                    auto cur_buffer_1 = interp_buffer_1;
                    auto cur_buffer_2 = interp_buffer_2;
                    for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                        auto cur_offset =  cur_ij_offset + k;
                        cur_buffer_1[buffer_idx] = data[cur_offset - last_linear_stride];
                        cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                        ++buffer_idx;
                    }
                    interp_linear1_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1, cur_buffer_2, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, 0, quantize_func);
                }
            }
        } else {
            size_t stride3x = 3 * stride;
            size_t i_start = 3;
            begins[direction] = i_start;
            ends[direction] = (n >= 3) ? (n - 3) : 0;
            strides[direction] = 2;
            size_t vector_len = ends[1] > begins[1] ? (ends[1]-begins[1]-1) / strides[1] + 1 : 0;

            // #pragma omp parallel for
            // for (size_t j = begins[1]; j < ends[1]; j += strides[1]) {
            auto cur_buffer_1 = interp_buffer_1;
            auto cur_buffer_2 = interp_buffer_2;
            auto cur_buffer_3 = interp_buffer_3;
            auto cur_buffer_4 = interp_buffer_4;

            if (n >= 5) {
                auto cur_ij_offset = offset + dim_offsets[0];
                for (size_t i = 0; i < vector_len; ++i) {
                    auto cur_offset =  cur_ij_offset + begins[1] + i * strides[1];
                    cur_buffer_2[i] = data[cur_offset - stride];
                    cur_buffer_3[i] = data[cur_offset + stride];
                    cur_buffer_4[i] = data[cur_offset + stride3x];
                }
                interp_quad1_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, 0, quantize_func);
            }
            else if (n >= 3) {
                auto cur_ij_offset = offset + dim_offsets[0];
                for (size_t i = 0; i < vector_len; ++i) {
                    auto cur_offset =  cur_ij_offset + begins[1] + i * strides[1];
                    cur_buffer_2[i] = data[cur_offset - stride];
                    cur_buffer_3[i] = data[cur_offset + stride];
                }
                interp_linear_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, cur_buffer_3, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, 0, quantize_func);
            }
            else if (n >= 1) {
                auto cur_ij_offset = offset + dim_offsets[0];
                for (size_t i = 0; i < vector_len; ++i) {
                    auto cur_offset =  cur_ij_offset + begins[1] + i * strides[1];
                    cur_buffer_2[i] = data[cur_offset - stride];
                }
                interp_equal_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, vector_len,
                    data + cur_ij_offset, strides[1], cur_ij_offset, 0, quantize_func);
            }
            size_t nthreads = omp_get_max_threads();
            size_t total_iters = (ends[0] - begins[0] + 1) >> 1;

            size_t base = total_iters / nthreads;
            size_t rem  = total_iters % nthreads;
            #pragma omp parallel //num_threads(omp_get_max_threads())
            {
                size_t tid = omp_get_thread_num();
                size_t my_iters = base + (tid < rem);
                size_t iter_start = tid < rem
                    ? tid * (base + 1)
                    : rem * (base + 1) + (tid - rem) * base;
                size_t start_idx = begins[0] + iter_start * strides[0];
                size_t end_idx = start_idx + my_iters * strides[0];


                auto buffer_offset = buffer_len * tid;
                auto cur_buffer_1 = interp_buffer_1 + buffer_offset;
                auto cur_buffer_2 = interp_buffer_2 + buffer_offset;
                auto cur_buffer_3 = interp_buffer_3 + buffer_offset;
                auto cur_buffer_4 = interp_buffer_4 + buffer_offset;

                if (start_idx < ends[0])
                {
                    auto cur_ij_offset = offset + start_idx * dim_offsets[0];
                    size_t buffer_idx = 0;
                    for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                        auto cur_offset =  cur_ij_offset + k;
                        cur_buffer_1[buffer_idx] = data[cur_offset -  stride3x];
                        //cur_buffer_1[buffer_idx] = data[0];
                        cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                        // cur_buffer_2[buffer_idx] = data[0];
                        cur_buffer_3[buffer_idx] = data[cur_offset + stride];
                        //cur_buffer_3[buffer_idx] = data[0];
                        cur_buffer_4[buffer_idx] = data[cur_offset +  stride3x];
                        //  cur_buffer_4[buffer_idx] = data[0];
                        ++buffer_idx;

                    }
                    interp_cubic_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1,cur_buffer_2,cur_buffer_3,cur_buffer_4, vector_len,
                        data + cur_ij_offset, strides[1], cur_ij_offset, tid, quantize_func);
                    for(size_t i = start_idx + strides[0]; i < end_idx; i += strides[0]) {
                        auto cur_ij_offset = offset + i * dim_offsets[0];
                        auto temp_buffer = cur_buffer_1;
                        cur_buffer_1 = cur_buffer_2;
                        cur_buffer_2 = cur_buffer_3;
                        cur_buffer_3 = cur_buffer_4;
                        cur_buffer_4 = temp_buffer;

                        size_t buffer_idx = 0;
                        for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                            auto cur_offset =  cur_ij_offset + stride3x + k;
                            cur_buffer_4[buffer_idx++] = data[cur_offset];
                        }
                        interp_cubic_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1,cur_buffer_2,cur_buffer_3,cur_buffer_4, vector_len,
                        data + cur_ij_offset, strides[1], cur_ij_offset, tid, quantize_func);
                    }
                }
            }


            // #pragma omp parallel for
            // for(size_t i = begins[0]; i < ends[0]; i += strides[0]){
            //     auto tid = omp_get_thread_num();
            //     auto buffer_offset = buffer_len * tid;
            //     auto cur_buffer_1 = interp_buffer_1 + buffer_offset;
            //     auto cur_buffer_2 = interp_buffer_2 + buffer_offset;
            //     auto cur_buffer_3 = interp_buffer_3 + buffer_offset;
            //     auto cur_buffer_4 = interp_buffer_4 + buffer_offset;
            //     // for(size_t i = begins[0]; i < ends[0]; i += strides[0]){
            //     auto cur_ij_offset = offset + i * dim_offsets[0];
            //     size_t buffer_idx = 0;
            //     for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
            //         auto cur_offset =  cur_ij_offset + k;
            //         cur_buffer_1[buffer_idx] = data[cur_offset -  stride3x];
            //         //cur_buffer_1[buffer_idx] = data[0];
            //         cur_buffer_2[buffer_idx] = data[cur_offset - stride];
            //         // cur_buffer_2[buffer_idx] = data[0];
            //         cur_buffer_3[buffer_idx] = data[cur_offset + stride];
            //         //cur_buffer_3[buffer_idx] = data[0];
            //         cur_buffer_4[buffer_idx] = data[cur_offset +  stride3x];
            //         //  cur_buffer_4[buffer_idx] = data[0];
            //         ++buffer_idx;

            //     }
            //         interp_cubic_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1,cur_buffer_2,cur_buffer_3,cur_buffer_4, vector_len,
            //             data + cur_ij_offset, strides[1], cur_ij_offset, tid);


            // }
            if (n % 2 == 1) {
                if (n > 3) {
                    // size_t buffer_idx = 0;
                    auto cur_ij_offset = offset + (n - 2) * dim_offsets[0];
                    // cur_buffer_1 = cur_buffer_2;
                    // cur_buffer_2 = cur_buffer_3;
                    // cur_buffer_3 = cur_buffer_4;
                    size_t buffer_idx = 0;
                    for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                        auto cur_offset =  cur_ij_offset + k;
                        cur_buffer_1[buffer_idx] = data[cur_offset -  stride3x];
                        cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                        cur_buffer_3[buffer_idx] = data[cur_offset + stride];
                        // cur_buffer_4[buffer_idx] = data[cur_offset +  stride3x];
                        ++buffer_idx;

                    }
                    interp_quad2_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1, cur_buffer_2, cur_buffer_3, vector_len,
                        data + cur_ij_offset, strides[1], cur_ij_offset, 0, quantize_func);

                }
            }
            else {
                if (n > 4) {
                    // size_t buffer_idx = 0;
                    auto cur_ij_offset = offset + (n - 3) * dim_offsets[0];
                    size_t buffer_idx = 0;
                    for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                        auto cur_offset =  cur_ij_offset + k;
                        cur_buffer_1[buffer_idx] = data[cur_offset -  stride3x];
                        cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                        cur_buffer_3[buffer_idx] = data[cur_offset + stride];
                        // cur_buffer_4[buffer_idx] = data[cur_offset +  stride3x];
                        ++buffer_idx;
                    }
                    interp_quad2_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1, cur_buffer_2, cur_buffer_3, vector_len,
                        data + cur_ij_offset, strides[1], cur_ij_offset, 0, quantize_func);
                }
                if (n > 2) {
                    // size_t buffer_idx = 0;
                    size_t buffer_idx = 0;
                    auto cur_ij_offset = offset + (n - 1) * dim_offsets[0];
                    for (size_t k = begins[1]; k < ends[1]; k += strides[1]) {
                        auto cur_offset =  cur_ij_offset + k;
                        cur_buffer_1[buffer_idx] = data[cur_offset -  stride3x];
                        cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                        // cur_buffer_3[buffer_idx] = data[cur_offset + stride];
                        // cur_buffer_4[buffer_idx] = data[cur_offset +  stride3x];
                        ++buffer_idx;
                    }

                    interp_linear1_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1, cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[1], cur_ij_offset, 0, quantize_func);
                }
            }
        //     std::vector<size_t> boundaries;
        //     boundaries.push_back(1);
        //     if (n % 2 == 1 && n > 3) {
        //         boundaries.push_back(n - 2);
        //     }
        //     if (n % 2 == 0 && n > 4) {
        //         boundaries.push_back(n - 3);
        //     }
        //     if (n % 2 == 0 && n > 2) {
        //         boundaries.push_back(n - 1);
        //     }

        //     for (auto boundary : boundaries) {
        //         begins[direction] = boundary;
        //         ends[direction] = boundary + 1;
        //         foreach_omp2
        //             <T, N>(data, offset, begins, ends, strides, dim_offsets, [&](T *d, int tid) {
        //                 if (boundary >= 3) {
        //                     if (boundary + 3 < n)
        //                         quantize_func(
        //                             d - data, *d,
        //                             interp_cubic(*(d - stride3x), *(d - stride), *(d + stride), *(d + stride3x)), tid);
        //                     else if (boundary + 1 < n)
        //                         quantize_func(d - data, *d,
        //                                       interp_quad_2(*(d - stride3x), *(d - stride), *(d + stride)), tid);
        //                     else
        //                         quantize_point<CompMode, SkipOverwrite>(d - data, *d, interp_linear1(*(d - stride3x), *(d - stride)), tid);
        //                 } else {
        //                     if (boundary + 3 < n)
        //                         quantize_func(d - data, *d,
        //                                       interp_quad_1(*(d - stride), *(d + stride), *(d + stride3x)), tid);
        //                     else if (boundary + 1 < n)
        //                         quantize_point<CompMode, SkipOverwrite>(d - data, *d, interp_linear(*(d - stride), *(d + stride)), tid);
        //                     else
        //                         quantize_point<CompMode, SkipOverwrite>(d - data, *d, *(d - stride), tid);
        //                 }
        //             });
        //     }
        }
        return predict_error;
    }

template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
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
            // size_t odd_len = n/2;//, even_len = n - odd_len;
            #pragma omp parallel for
            for (size_t i = begins[0]; i < ends[0]; i += strides[0]) {
                auto tid = omp_get_thread_num();
                auto cur_ij_offset = offset + i * dim_offsets[0];
                interp_linear_and_quantize_1D_line<CompMode, false, SkipOverwrite>(
                    data, n, dim_offsets[direction], cur_ij_offset, tid, quantize_func);

            }
        } else {
            //size_t stride3x = 3 * stride;
            //size_t i_start = 3;
            begins[direction] = 1;
            ends[direction] = n;
            strides[direction] = 2;
            // size_t odd_len = n/2;//, even_len = n - odd_len;
            #pragma omp parallel for
            for (size_t i = begins[0]; i < ends[0]; i += strides[0]) {
                auto tid = omp_get_thread_num();
                auto cur_ij_offset = offset + i * dim_offsets[0];
                interp_cubic_and_quantize_1D_line<CompMode, false, SkipOverwrite>(
                    data, n, dim_offsets[direction], cur_ij_offset, tid, quantize_func);
            }
        }
        return predict_error;
    }

    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    double interpolation_1d_simd_3d_z(T *data, const std::array<size_t, N> &begin_idx,
                                              const std::array<size_t, N> &end_idx, const size_t &direction,
                                              std::array<size_t, N> &strides, const size_t &math_stride,
                                              const std::string &interp_func, QuantizeFunc &&quantize_func) {
        assert(direction==0  && N==3);
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

        if (interp_func == "linear") {
            size_t last_linear_stride = (SkipOverwrite ? 3 * stride : 2 * stride);
            // begins[direction] = 1;
            // ends[direction] = n - 1;
            // strides[direction] = 2;
            // foreach_omp2
            //     <T, N>(data, offset, begins, ends, strides, dim_offsets,
            //            [&](T *d, int tid) { quantize_point<CompMode, SkipOverwrite>(d - data, *d, interp_linear(*(d - stride), *(d + stride)), tid); });
            // if (n % 2 == 0) {
            //     begins[direction] = n - 1;
            //     ends[direction] = n;
            //     foreach_omp2 //todo: this is infficient when direction = 0
            //         <T, N>(data, offset, begins, ends, strides, dim_offsets, [&](T *d, int tid) {
            //             if (n < 3)
            //                 quantize_point<CompMode, SkipOverwrite>(d - data, *d, *(d - stride), tid);
            //             else
            //                 quantize_point<CompMode, SkipOverwrite>(d - data, *d, interp_linear1(*(d - stride2x), *(d - stride)), tid);
            //         });
            // }
            begins[direction] = 1;
            ends[direction] = (n >= 1) ? (n - 1) : 0;
            strides[direction] = 2;
            size_t vector_len = ends[2] > begins[2] ? (ends[2]-begins[2]-1)/strides[2] + 1 : 0;
            #pragma omp parallel for
            for (size_t j = begins[1]; j < ends[1]; j += strides[1]) {
                auto tid = omp_get_thread_num();
                auto buffer_offset = buffer_len * tid;
                auto cur_buffer_1 = interp_buffer_1 + buffer_offset;
                auto cur_buffer_2 = interp_buffer_2 + buffer_offset;
                // auto cur_pred_buffer = pred_buffer + buffer_offset;
                for(size_t i = begins[0]; i < ends[0]; i += strides[0]){
                    auto cur_ij_offset = offset + i * dim_offsets[0] + j * dim_offsets[1];
                    size_t buffer_idx = 0;
                    if( i == begins[0]) {
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
                    interp_linear_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1,cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, tid, quantize_func);

                }
                if (n % 2 == 0) {
                    auto cur_ij_offset = offset + (n - 1) * dim_offsets[0] + j * dim_offsets[1];
                    if(n < 3) {
                        // n==2: the interior loop above (i in [1, n-1)) never ran, so cur_buffer_2
                        // still holds stale rows from a prior level/sweep. Fill it with the single
                        // available neighbor row (below) before predicting, or compress/decompress
                        // diverge on the leftover buffer contents.
                        size_t buffer_idx = 0;
                        for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                            cur_buffer_2[buffer_idx++] = data[cur_ij_offset - stride + k];
                        }
                        interp_equal_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, tid, quantize_func);
                    }
                    else {
                        size_t buffer_idx = 0;
                        for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                            auto cur_offset =  cur_ij_offset - last_linear_stride + k;
                           cur_buffer_1[buffer_idx++] = data[cur_offset];
                        }
                        interp_linear1_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1, cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, tid, quantize_func);
                    }
                }
            }
            // if (n % 2 == 0) {
            //     begins[direction] = n - 1;
            //     ends[direction] = n;
            //     foreach_omp2 //todo: this is infficient when direction = 0
            //         <T, N>(data, offset, begins, ends, strides, dim_offsets, [&](T *d, int tid) {
            //             if (n < 3)
            //                 quantize_point<CompMode, SkipOverwrite>(d - data, *d, *(d - stride), tid);
            //             else
            //                 quantize_point<CompMode, SkipOverwrite>(d - data, *d, interp_linear1(*(d - stride2x), *(d - stride)), tid);
            //         });
            // }
        } else {
            size_t stride3x = 3 * stride;
            size_t i_start = 3;
            begins[direction] = i_start;
            ends[direction] = (n >= 3) ? (n - 3) : 0;
            strides[direction] = 2;
            size_t vector_len = ends[2] > begins[2] ? (ends[2]-begins[2]-1)/strides[2] + 1 : 0;

            #pragma omp parallel for
            for (size_t j = begins[1]; j < ends[1]; j += strides[1]) {
                auto tid = omp_get_thread_num();
                auto buffer_offset = buffer_len * tid;
                auto cur_buffer_1 = interp_buffer_1 + buffer_offset;
                auto cur_buffer_2 = interp_buffer_2 + buffer_offset;
                auto cur_buffer_3 = interp_buffer_3 + buffer_offset;
                auto cur_buffer_4 = interp_buffer_4 + buffer_offset;

                // auto cur_ij_offset = offset + dim_offsets[0] + j * dim_offsets[1];
                // for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                //     auto cur_offset =  cur_ij_offset + k;
                //     cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                //     cur_buffer_3[buffer_idx] = data[cur_offset + stride];
                //     cur_buffer_4[buffer_idx] = data[cur_offset +  stride3x];
                //     ++buffer_idx;
                // }

                if (n >= 5) {
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
                        data + cur_ij_offset, strides[2], cur_ij_offset, tid, quantize_func);
                }
                else if (n >= 3) {
                    size_t buffer_idx = 0;
                    auto cur_ij_offset = offset + dim_offsets[0] + j * dim_offsets[1];
                    for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                        auto cur_offset =  cur_ij_offset + k;
                        cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                        cur_buffer_3[buffer_idx] = data[cur_offset + stride];
                        ++buffer_idx;
                    }
                    interp_linear_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, cur_buffer_3, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, tid, quantize_func);
                }
                else if (n >= 1) {
                    size_t buffer_idx = 0;
                    auto cur_ij_offset = offset + dim_offsets[0] + j * dim_offsets[1];
                    for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                        auto cur_offset =  cur_ij_offset + k;
                        cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                        ++buffer_idx;
                    }
                    interp_equal_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, tid, quantize_func);
                }
                for(size_t i = begins[0]; i < ends[0]; i += strides[0]){
                    auto cur_ij_offset = offset + i * dim_offsets[0] + j * dim_offsets[1];
                    size_t buffer_idx = 0;
                    // if( i == begins[0]){
                    //     for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                    //         auto cur_offset =  cur_ij_offset + k;
                    //         cur_buffer_1[buffer_idx] = data[cur_offset -  stride3x];
                    //        //cur_buffer_1[buffer_idx] = data[0];
                    //         cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                    //        // cur_buffer_2[buffer_idx] = data[0];
                    //         cur_buffer_3[buffer_idx] = data[cur_offset + stride];
                    //        //cur_buffer_3[buffer_idx] = data[0];
                    //         cur_buffer_4[buffer_idx] = data[cur_offset +  stride3x];
                    //       //  cur_buffer_4[buffer_idx] = data[0];
                    //         ++buffer_idx;

                    //     }
                    // }

                    // else{
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
                    // }
                    interp_cubic_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1,cur_buffer_2,cur_buffer_3,cur_buffer_4, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, tid, quantize_func);

                    // interp_cubic(cur_buffer_1,cur_buffer_2,cur_buffer_3,cur_buffer_4, cur_pred_buffer, vector_len);
                    // buffer_idx = 0;
                    // for (size_t k = begins[2]; k < ends[2]; k += strides[2]){
                    //     auto pred = cur_pred_buffer[buffer_idx++];
                    //     auto d = data + cur_ij_offset + k;
                    //   // if (d-data < 0 || d-data>=num_elements)
                    //   //      std::cout<<i<<" "<<j<<" "<<k<<std::endl;
                    //     quantize_point<CompMode, SkipOverwrite>(d - data, *d, pred, tid);

                    // }

                }
                if (n % 2 == 1) {
                    if (n > 3) {
                        // size_t buffer_idx = 0;
                        auto cur_ij_offset = offset + (n - 2) * dim_offsets[0] + j * dim_offsets[1];
                        cur_buffer_1 = cur_buffer_2;
                        cur_buffer_2 = cur_buffer_3;
                        cur_buffer_3 = cur_buffer_4;
                        interp_quad2_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1, cur_buffer_2, cur_buffer_3, vector_len,
                            data + cur_ij_offset, strides[2], cur_ij_offset, tid, quantize_func);

                    }
                }
                else {
                    if (n > 4) {
                        // size_t buffer_idx = 0;
                        auto cur_ij_offset = offset + (n - 3) * dim_offsets[0] + j * dim_offsets[1];
                        interp_quad2_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                            data + cur_ij_offset, strides[2], cur_ij_offset, tid, quantize_func);
                    }
                //     // else if (n > 2) {
                //         // size_t buffer_idx = 0;
                //         auto cur_ij_offset = offset + (n - 1) * dim_offsets[0] + j * dim_offsets[1];
                //         interp_linear1_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, cur_buffer_3, vector_len,
                //             data + cur_ij_offset, strides[2], cur_ij_offset, tid);
                //     // }
                }
            }
            if (n % 2 == 0 && n > 2) {
                /* final plane n-1: scalar edge predictors read data directly (no buffer state) */
                size_t bnd = n - 1;
                #pragma omp parallel for
                for (size_t j = begins[1]; j < ends[1]; j += strides[1]) {
                    int tid = omp_get_thread_num();
                    auto row = offset + bnd * dim_offsets[0] + j * dim_offsets[1];
                    for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                        T *d = data + row + k;
                        if (bnd >= 3) {
                            if (bnd + 3 < n)
                                quantize_point<CompMode, SkipOverwrite>(d - data, *d,
                                    interp_cubic(*(d - stride3x), *(d - stride), *(d + stride), *(d + stride3x)), tid);
                            else if (bnd + 1 < n)
                                quantize_point<CompMode, SkipOverwrite>(d - data, *d,
                                    interp_quad_2(*(d - stride3x), *(d - stride), *(d + stride)), tid);
                            else
                                quantize_point<CompMode, SkipOverwrite>(d - data, *d, interp_linear1(*(d - stride3x), *(d - stride)), tid);
                        } else {
                            if (bnd + 3 < n)
                                quantize_point<CompMode, SkipOverwrite>(d - data, *d,
                                    interp_quad_1(*(d - stride), *(d + stride), *(d + stride3x)), tid);
                            else if (bnd + 1 < n)
                                quantize_point<CompMode, SkipOverwrite>(d - data, *d, interp_linear(*(d - stride), *(d + stride)), tid);
                            else
                                quantize_point<CompMode, SkipOverwrite>(d - data, *d, *(d - stride), tid);
                        }
                    }
                }
            }
        }
        return predict_error;
    }

    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    double interpolation_1d_simd_3d_y(T *data, const std::array<size_t, N> &begin_idx,
                                              const std::array<size_t, N> &end_idx, const size_t &direction,
                                              std::array<size_t, N> &strides, const size_t &math_stride,
                                              const std::string &interp_func, QuantizeFunc &&quantize_func,
                                              size_t zi_begin = 0, size_t zi_end = ~size_t(0)) {
        assert(direction==1  && N==3);
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
        if (zi_begin > begins[0]) begins[0] = zi_begin;
        if (zi_end < ends[0]) ends[0] = zi_end;

        if (interp_func == "linear") {
            size_t last_linear_stride = (SkipOverwrite ? 3 * stride : 2 * stride);
            // begins[direction] = 1;
            // ends[direction] = n - 1;
            // strides[direction] = 2;
            // foreach_omp2
            //     <T, N>(data, offset, begins, ends, strides, dim_offsets,
            //            [&](T *d, int tid) { quantize_point<CompMode, SkipOverwrite>(d - data, *d, interp_linear(*(d - stride), *(d + stride)), tid); });
            // if (n % 2 == 0) {
            //     begins[direction] = n - 1;
            //     ends[direction] = n;
            //     foreach_omp2
            //         <T, N>(data, offset, begins, ends, strides, dim_offsets, [&](T *d, int tid) {
            //             if (n < 3)
            //                 quantize_point<CompMode, SkipOverwrite>(d - data, *d, *(d - stride), tid);
            //             else
            //                 quantize_point<CompMode, SkipOverwrite>(d - data, *d, interp_linear1(*(d - stride2x), *(d - stride)), tid);
            //         });
            // }
            begins[direction] = 1;
            ends[direction] = (n >= 1) ? (n - 1) : 0;
            strides[direction] = 2;
            size_t vector_len = ends[2] > begins[2] ? (ends[2]-begins[2]-1)/strides[2] + 1 : 0;

            #pragma omp parallel for
            for (size_t i = begins[0]; i < ends[0]; i += strides[0]) {
                int tid = omp_get_thread_num();
                auto buffer_offset = buffer_len * tid;
                auto cur_buffer_1 = interp_buffer_1 + buffer_offset;
                auto cur_buffer_2 = interp_buffer_2 + buffer_offset;
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
                        data + cur_ij_offset, strides[2], cur_ij_offset, tid, quantize_func);
                    // interp_linear(cur_buffer_1,cur_buffer_2, cur_pred_buffer, vector_len);
                    // buffer_idx = 0;
                    // for (size_t k = begins[2]; k < ends[2]; k += strides[2]){
                    //     auto pred = cur_pred_buffer[buffer_idx++];
                    //     auto d = data + cur_ij_offset + k;
                    //   // if (d-data < 0 || d-data>=num_elements)
                    //   //      std::cout<<i<<" "<<j<<" "<<k<<std::endl;
                    //     quantize_point<CompMode, SkipOverwrite>(d - data, *d, pred, tid);

                    // }

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
                        data + cur_ij_offset, strides[2], cur_ij_offset, tid, quantize_func);
                    }
                    else {
                        size_t buffer_idx = 0;
                        for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                            auto cur_offset =  cur_ij_offset - last_linear_stride + k;
                           cur_buffer_1[buffer_idx++] = data[cur_offset];
                        }
                        interp_linear1_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1, cur_buffer_2, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, tid, quantize_func);
                    }
                }

            }
            // if (n % 2 == 0) {
            //     begins[direction] = n - 1;
            //     ends[direction] = n;
            //     foreach_omp2 //todo: this is infficient when direction = 0
            //         <T, N>(data, offset, begins, ends, strides, dim_offsets, [&](T *d, int tid) {
            //             if (n < 3)
            //                 quantize_point<CompMode, SkipOverwrite>(d - data, *d, *(d - stride), tid);
            //             else
            //                 quantize_point<CompMode, SkipOverwrite>(d - data, *d, interp_linear1(*(d - stride2x), *(d - stride)), tid);
            //         });
            // }
        } else {
            size_t stride3x = 3 * stride;
            size_t i_start = 3;
            begins[direction] = i_start;
            ends[direction] = (n >= 3) ? (n - 3) : 0;
            strides[direction] = 2;
            size_t vector_len = ends[2] > begins[2] ? (ends[2]-begins[2]-1)/strides[2] + 1 : 0;

            #pragma omp parallel for
            for (size_t i = begins[0]; i < ends[0]; i += strides[0]) {
                int tid = omp_get_thread_num();
                auto buffer_offset = buffer_len * tid;
                auto cur_buffer_1 = interp_buffer_1 + buffer_offset;
                auto cur_buffer_2 = interp_buffer_2 + buffer_offset;
                auto cur_buffer_3 = interp_buffer_3 + buffer_offset;
                auto cur_buffer_4 = interp_buffer_4 + buffer_offset;
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
                        data + cur_ij_offset, strides[2], cur_ij_offset, tid, quantize_func);
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
                        data + cur_ij_offset, strides[2], cur_ij_offset, tid, quantize_func);
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
                        data + cur_ij_offset, strides[2], cur_ij_offset, tid, quantize_func);
                }
                for(size_t j = begins[1]; j < ends[1]; j += strides[1]){
                    auto cur_ij_offset = offset + i * dim_offsets[0] + j * dim_offsets[1];
                    size_t buffer_idx = 0;
                    // if( j == begins[1]){
                    //     for (size_t k = begins[2]; k < ends[2]; k += strides[2]) {
                    //         auto cur_offset =  cur_ij_offset + k;
                    //         //if (visited[cur_offset- stride3x]==0 or visited[cur_offset+ stride3x]==0)
                    //         //   std::cout<<"e1 "<<i<<" "<<j<<" "<<k<<" "<<stride3x<<std::endl;
                    //         cur_buffer_1[buffer_idx] = data[cur_offset -  stride3x];
                    //         //cur_buffer_1[buffer_idx] = data[0];
                    //         cur_buffer_2[buffer_idx] = data[cur_offset - stride];
                    //         // cur_buffer_2[buffer_idx] = data[0];
                    //         cur_buffer_3[buffer_idx] = data[cur_offset + stride];
                    //         //cur_buffer_3[buffer_idx] = data[0];
                    //         cur_buffer_4[buffer_idx] = data[cur_offset +  stride3x];
                    //         //  cur_buffer_4[buffer_idx] = data[0];
                    //         buffer_idx++;

                    //     }
                    // }
                    // else{
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
                    // }

                    interp_cubic_and_quantize<CompMode, SkipOverwrite>(cur_buffer_1, cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                        data + cur_ij_offset, strides[2], cur_ij_offset, tid, quantize_func);

                    // interp_cubic(cur_buffer_1,cur_buffer_2,cur_buffer_3,cur_buffer_4, cur_pred_buffer, vector_len);
                    // buffer_idx = 0;
                    // for (size_t k = begins[2]; k < ends[2]; k += strides[2]){
                    //     auto pred = cur_pred_buffer[buffer_idx++];
                    //     auto d = data + cur_ij_offset + k;
                    //   // if (d-data < 0 || d-data>=num_elements)
                    //   //      std::cout<<i<<" "<<j<<" "<<k<<std::endl;
                    //     quantize_point<CompMode, SkipOverwrite>(d - data, *d, pred, tid);

                    // }

                }
                if (n % 2 == 1) {
                    if (n > 3) {
                        // size_t buffer_idx = 0;
                        auto cur_ij_offset = offset + i * dim_offsets[0] + (n - 2) * dim_offsets[1];
                        interp_quad2_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, cur_buffer_3, cur_buffer_4, vector_len,
                            data + cur_ij_offset, strides[2], cur_ij_offset, tid, quantize_func);

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
                            data + cur_ij_offset, strides[2], cur_ij_offset, tid, quantize_func);
                    }
                    // else if (n > 2) {
                        // size_t buffer_idx = 0;
                        // auto cur_ij_offset = offset + i * dim_offsets[0] + (n - 1) * dim_offsets[1];
                        // interp_linear1_and_quantize<CompMode, SkipOverwrite>(cur_buffer_2, cur_buffer_3, vector_len,
                        //     data + cur_ij_offset, strides[2], cur_ij_offset, tid);
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
                foreach_omp2
                    <T, N>(data, offset, begins, ends, strides, dim_offsets, [&](T *d, int tid) {
                        if (boundary >= 3) {
                            if (boundary + 3 < n)
                                quantize_func(
                                    d - data, *d,
                                    interp_cubic(*(d - stride3x), *(d - stride), *(d + stride), *(d + stride3x)), tid);
                            else if (boundary + 1 < n)
                                quantize_func(d - data, *d,
                                              interp_quad_2(*(d - stride3x), *(d - stride), *(d + stride)), tid);
                            else
                                quantize_point<CompMode, SkipOverwrite>(d - data, *d, interp_linear1(*(d - stride3x), *(d - stride)), tid);
                        } else {
                            if (boundary + 3 < n)
                                quantize_func(d - data, *d,
                                              interp_quad_1(*(d - stride), *(d + stride), *(d + stride3x)), tid);
                            else if (boundary + 1 < n)
                                quantize_point<CompMode, SkipOverwrite>(d - data, *d, interp_linear(*(d - stride), *(d + stride)), tid);
                            else
                                quantize_point<CompMode, SkipOverwrite>(d - data, *d, *(d - stride), tid);
                        }
                    });
            }
        }
        return predict_error;
    }


    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    double interpolation_1d_simd_3d_x(T *data, const std::array<size_t, N> &begin_idx,
                                              const std::array<size_t, N> &end_idx, const size_t &direction,
                                              std::array<size_t, N> &strides, const size_t &math_stride,
                                              const std::string &interp_func, QuantizeFunc &&quantize_func,
                                              size_t zi_begin = 0, size_t zi_end = ~size_t(0)) {
        // assert(direction==2 && N==3);
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
        if (zi_begin > begins[0]) begins[0] = zi_begin;
        if (zi_end < ends[0]) ends[0] = zi_end;

        if (interp_func == "linear") {
            // begins[direction] = 1;
            // ends[direction] = n - 1;
            // strides[direction] = 2;
            // size_t stride2x = 2 * stride;
            // foreach_omp2
            //     <T, N>(data, offset, begins, ends, strides, dim_offsets,
            //            [&](T *d, int tid) { quantize_point<CompMode, SkipOverwrite>(d - data, *d, interp_linear(*(d - stride), *(d + stride)), tid); });
            // if (n % 2 == 0) {
            //     begins[direction] = n - 1;
            //     ends[direction] = n;
            //     foreach_omp2
            //         <T, N>(data, offset, begins, ends, strides, dim_offsets, [&](T *d, int tid) {
            //             if (n < 3)
            //                 quantize_point<CompMode, SkipOverwrite>(d - data, *d, *(d - stride), tid);
            //             else
            //                 quantize_point<CompMode, SkipOverwrite>(d - data, *d, interp_linear1(*(d - stride2x), *(d - stride)), tid);
            //         });
            // }

            begins[direction] = 1;
            ends[direction] = n;
            strides[direction] = 2;
            // size_t odd_len = n/2;//, even_len = n - odd_len;
            #pragma omp parallel for collapse(2) schedule(static)
            for (size_t i = begins[0]; i < ends[0]; i += strides[0]) {
                for(size_t j = begins[1]; j < ends[1]; j += strides[1]){
                    int tid = omp_get_thread_num();
                    auto cur_ij_offset = offset + i * dim_offsets[0] + j * dim_offsets[1];
                    if (n >= 64) {
                        auto even_buffer = interp_buffer_1 + buffer_len * tid;
                        size_t even_len = n - n / 2;
                        for (size_t k = 0; k < even_len; ++k)
                            even_buffer[k] = data[cur_ij_offset + (k << 1) * dim_offsets[direction]];
                        interp_linear_and_quantize_1D<CompMode,
                            CompMode == COMPMODE::COMP && SkipOverwrite>(
                            even_buffer, n, data + cur_ij_offset, dim_offsets[direction],
                            cur_ij_offset, tid, quantize_func);
                    } else {
                        interp_linear_and_quantize_1D_line<CompMode, false, SkipOverwrite>(
                            data, n, dim_offsets[direction], cur_ij_offset, tid, quantize_func);
                    }
                    // for (size_t k = 0; k < odd_len; ++k ){
                    //     auto pred = cur_pred_buffer[k];
                    //     auto d = data + cur_ij_offset + (2 * k + 1) * dim_offsets[2];
                    //   // if (d-data < 0 || d-data>=num_elements)
                    //   //      std::cout<<i<<" "<<j<<" "<<k<<std::endl;
                    //     quantize_point<CompMode, SkipOverwrite>(d - data, *d, pred, tid);

                    // }

                }
            }
        } else {
            //size_t stride3x = 3 * stride;
            //size_t i_start = 3;
            begins[direction] = 1;
            ends[direction] = n;
            strides[direction] = 2;
            // size_t odd_len = n/2;//, even_len = n - odd_len;
            #pragma omp parallel for collapse(2) schedule(static)
            for (size_t i = begins[0]; i < ends[0]; i += strides[0]) {
                for(size_t j = begins[1]; j < ends[1]; j += strides[1]){
                    int tid = omp_get_thread_num();
                    auto cur_ij_offset = offset + i * dim_offsets[0] + j * dim_offsets[1];
                    if (n >= 64) {
                        auto even_buffer = interp_buffer_1 + buffer_len * tid;
                        size_t even_len = n - n / 2;
                        for (size_t k = 0; k < even_len; ++k)
                            even_buffer[k] = data[cur_ij_offset + (k << 1) * dim_offsets[direction]];
                        interp_cubic_and_quantize_1D<CompMode,
                            CompMode == COMPMODE::COMP && SkipOverwrite>(
                            even_buffer, n, data + cur_ij_offset, dim_offsets[direction],
                            cur_ij_offset, tid, quantize_func);
                    } else {
                        interp_cubic_and_quantize_1D_line<CompMode, false, SkipOverwrite>(
                            data, n, dim_offsets[direction], cur_ij_offset, tid, quantize_func);
                    }
                    // interp_cubic_1D(cur_buffer,cur_pred_buffer, n);
                    // for (size_t k = 0; k < odd_len; ++k ){
                    //     auto pred = cur_pred_buffer[k];
                    //     auto d = data + cur_ij_offset + (2 * k + 1) * dim_offsets[2];
                    //   // if (d-data < 0 || d-data>=num_elements)
                    //   //      std::cout<<i<<" "<<j<<" "<<k<<std::endl;
                    //     quantize_point<CompMode, SkipOverwrite>(d - data, *d, pred, tid);

                    // }

                }
            }
        }
        return predict_error;
    }


    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
    double interpolation_1d_simd(T *data, size_t begin, size_t end, size_t math_stride,
                                 const std::string &interp_func, QuantizeFunc &&quantize_func) {
        size_t n = (end - begin) / math_stride + 1;
        if (n <= 1) {
            return 0;
        }
        size_t stride = math_stride * original_dim_offsets[0];
        size_t cur_ij_offset = begin * original_dim_offsets[0];
        int tid = omp_get_thread_num();
        if (interp_func == "linear") {
            interp_linear_and_quantize_1D_line<CompMode, true, SkipOverwrite>(data, n, stride, cur_ij_offset, tid, quantize_func);
        } else {
            interp_cubic_and_quantize_1D_line<CompMode, true, SkipOverwrite>(data, n, stride, cur_ij_offset, tid, quantize_func);
        }
        return 0;
    }




    template <COMPMODE CompMode, bool SkipOverwrite, class QuantizeFunc>
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
        if constexpr (N == 1) {  // old API
            if constexpr (std::is_integral_v<T>) {
                return interpolation_1d(data, begin[0], end[0], stride, interp_func, quantize_func);
            } else {
                return interpolation_1d_simd<CompMode, SkipOverwrite>(data, begin[0], end[0], stride, interp_func, quantize_func);
            }
        }
        // else if constexpr (N == 2) {  // old API
        //     double predict_error = 0;
        //     size_t stride2x = stride * 2;
        //     const std::array<int, N> dims = dim_sequences[direction];
        //     for (size_t j = (begin[dims[1]] ? begin[dims[1]] + stride2x : 0); j <= end[dims[1]]; j += stride2x) {
        //         size_t begin_offset =
        //             begin[dims[0]] * original_dim_offsets[dims[0]] + j * original_dim_offsets[dims[1]];
        //         predict_error += interpolation_1d(
        //             data, begin_offset, begin_offset + (end[dims[0]] - begin[dims[0]]) * original_dim_offsets[dims[0]],
        //             stride * original_dim_offsets[dims[0]], interp_func, quantize_func);
        //     }
        //     for (size_t i = (begin[dims[0]] ? begin[dims[0]] + stride : 0); i <= end[dims[0]]; i += stride) {
        //         size_t begin_offset =
        //             i * original_dim_offsets[dims[0]] + begin[dims[1]] * original_dim_offsets[dims[1]];
        //         predict_error += interpolation_1d(
        //             data, begin_offset, begin_offset + (end[dims[1]] - begin[dims[1]]) * original_dim_offsets[dims[1]],
        //             stride * original_dim_offsets[dims[1]], interp_func, quantize_func);
        //     }
        //     return predict_error;
        // }
        else if constexpr (N == 2) {
            double predict_error = 0;
            size_t stride2x = stride * 2;
            const std::array<int, N> dims = dim_sequences[direction];
            std::array<size_t, N> strides;
            std::array<size_t, N> begin_idx = begin, end_idx = end;
            strides[dims[0]] = 1;
            // size_t max_interp_seq_length = 0;
            //  for (uint i = 0; i < N; ++i)
            //     max_interp_seq_length = std::max(max_interp_seq_length, (end[i]-begin[i])/stride );
            for (uint i = 1; i < N; ++i) {
                begin_idx[dims[i]] = (begin[dims[i]] ? begin[dims[i]] + stride2x : 0);
                strides[dims[i]] = stride2x;
            }
            if constexpr (!std::is_integral_v<T>) {
                if(direction == 0 ){//xy
                    predict_error += interpolation_1d_simd_2d_y<CompMode, false>(data, begin_idx, end_idx, dims[0], strides, stride, interp_func, quantize_func);
                    // std::cout << "after x direction" << std::endl;
                    begin_idx[1] = begin[1];
                    begin_idx[0] = (begin[0] ? begin[0] + stride : 0);
                    strides[0] = stride;
                    predict_error += interpolation_1d_simd_2d_x<CompMode, SkipOverwrite>(data, begin_idx, end_idx, dims[1], strides, stride, interp_func, quantize_func);
                }
                else {
                    predict_error += interpolation_1d_simd_2d_x<CompMode, false>(data, begin_idx, end_idx, dims[0], strides, stride, interp_func, quantize_func);
                    // std::cout << "after x direction" << std::endl;
                    begin_idx[0] = begin[0];
                    begin_idx[1] = (begin[1] ? begin[1] + stride : 0);
                    strides[1] = stride;
                    predict_error += interpolation_1d_simd_2d_y<CompMode, SkipOverwrite>(data, begin_idx, end_idx, dims[1], strides, stride, interp_func, quantize_func);
                }
            } else {
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
            // std::cout << "after y direction" << std::endl;
            // const std::array<int, N> dims = dim_sequences[direction];
            // for (size_t j = (begin[dims[1]] ? begin[dims[1]] + stride2x : 0); j <= end[dims[1]]; j += stride2x) {
            //     size_t begin_offset =
            //         begin[dims[0]] * original_dim_offsets[dims[0]] + j * original_dim_offsets[dims[1]];
            //     predict_error += interpolation_1d(
            //         data, begin_offset, begin_offset + (end[dims[0]] - begin[dims[0]]) * original_dim_offsets[dims[0]],
            //         stride * original_dim_offsets[dims[0]], interp_func, quantize_func);
            // }
            // for (size_t i = (begin[dims[0]] ? begin[dims[0]] + stride : 0); i <= end[dims[0]]; i += stride) {
            //     size_t begin_offset =
            //         i * original_dim_offsets[dims[0]] + begin[dims[1]] * original_dim_offsets[dims[1]];
            //     predict_error += interpolation_1d(
            //         data, begin_offset, begin_offset + (end[dims[1]] - begin[dims[1]]) * original_dim_offsets[dims[1]],
            //         stride * original_dim_offsets[dims[1]], interp_func, quantize_func);
            // }
            return predict_error;
        }
        else if constexpr (N == 3 || N == 4) {  // new API (for faster speed)
// #ifdef Falcons_PRINT_TIMINGS
//             Timer timer(true);
// #endif
            double predict_error = 0;
            size_t stride2x = stride * 2;
            const std::array<int, N> dims = dim_sequences[direction];
            std::array<size_t, N> strides;
            std::array<size_t, N> begin_idx = begin, end_idx = end;
            strides[dims[0]] = 1;
            for (uint i = 1; i < N; ++i) {
                begin_idx[dims[i]] = (begin[dims[i]] ? begin[dims[i]] + stride2x : 0);
                strides[dims[i]] = stride2x;
            }
            if constexpr (N == 3 && !std::is_integral_v<T>){//avx
                if (direction != 0 && direction != 5) {  // the SIMD sweeps only cover xyz and zyx
                    predict_error += interpolation_1d_fastest_dim_first(data, begin_idx, end_idx, dims[0], strides, stride, interp_func, quantize_func);
                    for (uint i = 1; i < N; ++i) {
                        begin_idx[dims[i]] = begin[dims[i]];
                        begin_idx[dims[i - 1]] = (begin[dims[i - 1]] ? begin[dims[i - 1]] + stride : 0);
                        strides[dims[i - 1]] = stride;
                        predict_error += interpolation_1d_fastest_dim_first(data, begin_idx, end_idx, dims[i], strides, stride, interp_func, quantize_func);
                    }
                    return predict_error;
                }
                if(direction == 0 ){//xyz
                    predict_error += interpolation_1d_simd_3d_z<CompMode, false>(data, begin_idx, end_idx, dims[0], strides, stride, interp_func, quantize_func);
                    begin_idx[1] = begin[1];
                    begin_idx[0] = (begin[0] ? begin[0] + stride : 0);
                    strides[0] = stride;
                    predict_error += interpolation_1d_simd_3d_y<CompMode, false>(data, begin_idx, end_idx, dims[1], strides, stride, interp_func, quantize_func);
                    begin_idx[2] = begin[2];
                    begin_idx[1] = (begin[1] ? begin[1] + stride : 0);
                    strides[1] = stride;
                    predict_error += interpolation_1d_simd_3d_x<CompMode, SkipOverwrite>(data, begin_idx, end_idx, dims[2], strides, stride, interp_func, quantize_func);
                }
                else{//zyx
                    predict_error += interpolation_1d_simd_3d_x<CompMode, false>(data, begin_idx, end_idx, dims[0], strides, stride, interp_func, quantize_func);
                    begin_idx[1] = begin[1];
                    begin_idx[2] = (begin[2] ? begin[2] + stride : 0);
                    strides[2] = stride;
                    predict_error += interpolation_1d_simd_3d_y<CompMode, false>(data, begin_idx, end_idx, dims[1], strides, stride, interp_func, quantize_func);
                    // z-pass (last, separate); explicit z-config correct for both branches
                    begin_idx[0] = begin[0];
                    begin_idx[1] = (begin[1] ? begin[1] + stride : 0);
                    begin_idx[2] = (begin[2] ? begin[2] + stride : 0);
                    strides[1] = stride;
                    strides[2] = stride;
                    predict_error += interpolation_1d_simd_3d_z<CompMode, SkipOverwrite>(data, begin_idx, end_idx, dims[2], strides, stride, interp_func, quantize_func);
                }

            }

            else{
                predict_error += interpolation_1d_fastest_dim_first(data, begin_idx, end_idx, dims[0], strides, stride, interp_func, quantize_func);
                for (uint i = 1; i < N; ++i) {
                begin_idx[dims[i]] = begin[dims[i]];
                begin_idx[dims[i - 1]] = (begin[dims[i - 1]] ? begin[dims[i - 1]] + stride : 0);
                strides[dims[i - 1]] = stride;

                predict_error += interpolation_1d_fastest_dim_first(data, begin_idx, end_idx, dims[i], strides, stride, interp_func, quantize_func);
                }
            }
// #ifdef Falcons_PRINT_TIMINGS
//             timer.stop("interpolation");
// #endif
            return predict_error;
        } else {
            throw std::runtime_error("Unsupported dimension in InterpolationDecomposition");
        }

    }

    int interp_level = -1;
    int interp_id;
    uint blocksize;
    std::vector<std::string> interpolators = {"linear", "cubic"};
    QuantizerOMP quantizer;
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
    size_t nThreads = 1;
    bool legacy_stream = false;  // reading a 1.0.0 stream
    size_t buffer_len = 1024;
    T *interp_buffer_1 = nullptr;
    T *interp_buffer_2 = nullptr;
    T *interp_buffer_3 = nullptr;
    T *interp_buffer_4 = nullptr;
    T *pred_buffer = nullptr;
    // std::vector<std::vector<int>> local_quant_inds;

    int16_t* total_quant_inds;
    int16_t** local_quant_inds;
    CacheLineInt* local_quant_index;
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
    //std::vector<size_t> level_prefix;
    //std::vector<std::array<size_t,N> >reduced_dim_offsets;


};

template <class T, uint N, class QuantizerOMP>
InterpolationDecomposition_OMP<T, N, QuantizerOMP> make_decomposition_interpolation_omp(const Config &conf, QuantizerOMP quantizer) {
    return  InterpolationDecomposition_OMP<T, N, QuantizerOMP>(conf, quantizer);
}

}  // namespace Falcons

#include "Interpolation_quantizer_Omp.inl"
#endif
#endif
