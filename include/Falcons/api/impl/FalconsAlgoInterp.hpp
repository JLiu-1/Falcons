/**
 * @file FalconsAlgoInterp.hpp
 * @author Bing Lu
 * @brief Falcons interpolation algorithm integration and tuning.
 *
 * Copyright (c) 2026 Bing Lu.
 */

#ifndef Falcons_FalconsALGO_INTERP_HPP
#define Falcons_FalconsALGO_INTERP_HPP

#ifdef _OPENMP
#include "Falcons/decomposition/InterpolationDecomposition_Omp.hpp"
#include "Falcons/quantizer/LinearQuantizer_Omp.hpp"
#include "Falcons/compressor/FalconsGenericCompressor_Omp.hpp"
#endif

#include <cstdlib>
#include "Falcons/encoder/RleFseEncoder.hpp"
#include "Falcons/api/impl/FalconsAlgoLorenzoReg.hpp"
#include "Falcons/decomposition/BlockwiseDecomposition.hpp"
#include "Falcons/decomposition/InterpolationDecomposition.hpp"
#include "Falcons/lossless/Lossless_zstd.hpp"
#include "Falcons/quantizer/LinearQuantizer.hpp"
#include "Falcons/utils/Config.hpp"
#include "Falcons/utils/Extraction.hpp"
#include "Falcons/utils/QuantOptimizatioin.hpp"
#include "Falcons/utils/Sample.hpp"
#include "Falcons/utils/Statistic.hpp"

namespace Falcons {
template <class T, uint N>
size_t Falcons_compress_Interp(Config &conf, T *data, uchar *cmpData, size_t cmpCap) {
    assert(N == conf.N);
    assert(conf.cmprAlgo == ALGO_INTERP);
    calAbsErrorBound(conf, data);
    if (conf.interpAnchorStride < 0) {  // set default anchor stride
        std::array<size_t, 4> anchor_strides = {4096, 128, 32, 16};
        conf.interpAnchorStride = anchor_strides[N - 1];
    }
    // conf.interpAlgo = 0;
    // conf.interpDirection = 0;
    // conf.interpAlpha = 1;
    // conf.interpBeta = 1;
#ifdef Falcons_PRINT_TIMINGS
    std::cout << "interpAlgo " << int(conf.interpAlgo) << "  interpDirection " << int(conf.interpDirection) << std::endl;
    std::cout << "interpAlpha " << conf.interpAlpha << " interpBeta " << conf.interpBeta << std::endl;
#endif
    #ifdef _OPENMP
    int threads = omp_get_max_threads();
    // Integers take the serial path: LinearQuantizerOMP has no overflow-safe integer arithmetic.
    if (threads > 1 && conf.openmp && std::is_floating_point_v<T>) {
        auto falcons = make_compressor_falcons_generic_omp<T, N>(
        make_decomposition_interpolation_omp<T, N>(conf, LinearQuantizerOMP<T>(conf.absErrorBound, conf.quantbinCnt / 2)),
        RleFseEncoder<int16_t>(),  Lossless_zstd());
        return falcons->compress(conf, data, cmpData, cmpCap);
    }
    else {auto falcons = make_compressor_falcons_generic<T, N>(
        make_decomposition_interpolation<TUNING::DISABLED, T, N>(conf, LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2)),
        RleFseEncoder<int16_t>(), Lossless_zstd());
    return falcons->compress(conf, data, cmpData, cmpCap);}
    #else
        auto falcons = make_compressor_falcons_generic<T, N>(
        make_decomposition_interpolation<TUNING::DISABLED, T, N>(conf, LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2)),
        RleFseEncoder<int16_t>(), Lossless_zstd());
            return falcons->compress(conf, data, cmpData, cmpCap);
    #endif


}

// The OMP layout starts with [int streams][size_t offsets[0] == 0]. A serial stream has its zstd frame
// at byte 8. The layout decides the decoder; the caller's thread count must not.
inline bool Falcons_stream_is_omp(const Config &conf, const uchar *cmpData, size_t cmpSize) {
    static const uchar zeros[8] = {0};
    return conf.openmp && cmpSize >= 12 && memcmp(cmpData + 4, zeros, 8) == 0;
}

template <class T, uint N>
void Falcons_decompress_Interp(const Config &conf, const uchar *cmpData, size_t cmpSize, T *decData) {
    assert(conf.cmprAlgo == ALGO_INTERP);
    auto cmpDataPos = cmpData;
   // std::cout<<"decomp started"<<std::endl;
    #ifdef _OPENMP
    if (Falcons_stream_is_omp(conf, cmpData, cmpSize)) {
    auto falcons = make_compressor_falcons_generic_omp<T, N>(
        make_decomposition_interpolation_omp<T, N>(conf, LinearQuantizerOMP<T>(conf.absErrorBound, conf.quantbinCnt / 2)),
        RleFseEncoder<int16_t>(), Lossless_zstd());
         falcons->decompress(conf, cmpDataPos, cmpSize, decData);
    }
    else {
        auto falcons = make_compressor_falcons_generic<T, N>(
        make_decomposition_interpolation<TUNING::DISABLED, T, N>(conf, LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2)),
        RleFseEncoder<int16_t>(), Lossless_zstd());
         falcons->decompress(conf, cmpDataPos, cmpSize, decData);
    }
    #else
    if (Falcons_stream_is_omp(conf, cmpData, cmpSize))
        throw std::invalid_argument("This Falcons stream was written with -P, rebuild Falcons with OpenMP to read it");
    auto falcons = make_compressor_falcons_generic<T, N>(
        make_decomposition_interpolation<TUNING::DISABLED, T, N>(conf, LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2)),
        RleFseEncoder<int16_t>(), Lossless_zstd());
         falcons->decompress(conf, cmpDataPos, cmpSize, decData);
    #endif
   
}

template <class T, uint N>
double interp_compress_test(
    const std::vector<std::vector<T>> &sampled_blocks, const Config &conf, int block_size, uchar *cmpData,
    size_t cmpCap) {  // test interp cmp on a set of sampled data blocks and return the compression ratio
    // std::vector<std::vector<int> > quant_inds_vec(sampled_blocks.size());
    int16_t** quant_inds = new int16_t*[sampled_blocks.size()];
    uint64_t* quant_inds_size = new uint64_t[sampled_blocks.size()];
    std::vector<std::vector<T> > unpred_vec(sampled_blocks.size());
    // #ifdef _OPENMP
    // #pragma omp parallel for
    // #endif
    // for (size_t k = 0; k < sampled_blocks.size(); ++k) {
    //     unpred_buffer[k] = new uchar[sampled_blocks.size() * sizeof(T)
    //         * std::accumulate(conf.dims.begin(), conf.dims.end(), 1, std::multiplies<size_t>())];
    //     unpred_buffer_pos[k] = unpred_buffer[k];
    // }
    size_t sample_size = sampled_blocks.size();
    #ifdef _OPENMP
    #pragma omp parallel for if(conf.openmp)
    #endif
    for (size_t k = 0; k < sample_size; ++k) {
        auto falcons =
            make_decomposition_interpolation<TUNING::ENABLED, T, N>(conf, LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2));
        auto cur_block = sampled_blocks[k];
        std::tie(quant_inds[k], quant_inds_size[k]) = falcons.compress(conf, cur_block.data());
        unpred_vec[k] = falcons.test_unpred();
    }
    std::vector<size_t> prefix_sum(sample_size << 1);
    size_t cur_size = 0;
    for (size_t k = 0; k < sample_size << 1; ++k) {
        prefix_sum[k] = cur_size;
        if(k < sample_size) {
            cur_size += unpred_vec[k].size() * sizeof(T);
        }
        else {
            cur_size += quant_inds_size[k - sample_size] * sizeof(int16_t);
        }
    }
    uchar* buffer = new uchar[cur_size];
    for (size_t k = 0; k < sample_size << 1; ++k) {
        if(k < sample_size) {
            if (!unpred_vec[k].empty())
                memcpy(buffer + prefix_sum[k], reinterpret_cast<const uchar*>(unpred_vec[k].data()), unpred_vec[k].size() * sizeof(T));
        }
        else {
            memcpy(buffer + prefix_sum[k], reinterpret_cast<const uchar*>(quant_inds[k - sample_size]), quant_inds_size[k - sample_size] * sizeof(int16_t));
        }
    }
    auto lossless = Lossless_zstd();
    auto cmpSize = lossless.compress(buffer, cur_size, cmpData, cmpCap);

    auto compression_ratio = conf.num * sampled_blocks.size() * sizeof(T) * 1.0 / cmpSize;
    
    #ifdef _OPENMP
    #pragma omp parallel for if(conf.openmp)
    #endif
    for (size_t k = 0; k < sample_size; ++k) {
        delete[] quant_inds[k];
    }
    delete[] quant_inds;
    delete[] quant_inds_size;
    delete[] buffer;

    return compression_ratio;

}

template <class T, uint N>
void interp_compress_test_batch(
    const std::vector<std::vector<T>> &sampled_blocks, const Config *confs, int nconf,
    int block_size, double *ratios_out) {
    // Flatten config x block into one task pool so all threads can process sampled blocks.
    size_t B = sampled_blocks.size();
    size_t total = static_cast<size_t>(nconf) * B;
    int16_t **quant_inds = new int16_t*[total];
    uint64_t *quant_inds_size = new uint64_t[total];
    std::vector<std::vector<T>> unpred_vec(total);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(nconf > 0 && confs[0].openmp)
#endif
    for (size_t t = 0; t < total; ++t) {
        int c = static_cast<int>(t / B);
        size_t k = t % B;
        auto falcons = make_decomposition_interpolation<TUNING::ENABLED, T, N>(
            confs[c], LinearQuantizer<T>(confs[c].absErrorBound, confs[c].quantbinCnt / 2));
        auto cur_block = sampled_blocks[k];
        std::tie(quant_inds[t], quant_inds_size[t]) = falcons.compress(confs[c], cur_block.data());
        unpred_vec[t] = falcons.test_unpred();
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(nconf > 0 && confs[0].openmp)
#endif
    for (int c = 0; c < nconf; ++c) {
        size_t base = static_cast<size_t>(c) * B;
        std::vector<size_t> prefix_sum(B << 1);
        size_t cur_size = 0;
        for (size_t k = 0; k < B << 1; ++k) {
            prefix_sum[k] = cur_size;
            if (k < B) {
                cur_size += unpred_vec[base + k].size() * sizeof(T);
            } else {
                cur_size += quant_inds_size[base + k - B] * sizeof(int16_t);
            }
        }
        uchar *buffer = new uchar[cur_size];
        for (size_t k = 0; k < B << 1; ++k) {
            if (k < B) {
                if (!unpred_vec[base + k].empty())
                    memcpy(buffer + prefix_sum[k], reinterpret_cast<const uchar *>(unpred_vec[base + k].data()),
                           unpred_vec[base + k].size() * sizeof(T));
            } else {
                memcpy(buffer + prefix_sum[k], reinterpret_cast<const uchar *>(quant_inds[base + k - B]),
                       quant_inds_size[base + k - B] * sizeof(int16_t));
            }
        }
        size_t cmpCap = cur_size + (cur_size >> 1) + 1024;
        uchar *cmpData = new uchar[cmpCap];
        auto lossless = Lossless_zstd();
        auto cmpSize = lossless.compress(buffer, cur_size, cmpData, cmpCap);
        ratios_out[c] = confs[c].num * B * sizeof(T) * 1.0 / cmpSize;
        delete[] cmpData;
        delete[] buffer;
    }
    for (size_t t = 0; t < total; ++t) delete[] quant_inds[t];
    delete[] quant_inds;
    delete[] quant_inds_size;
}

template <class T, uint N>
double lorenzo_compress_test(
    const std::vector<std::vector<T>> sampled_blocks, const Config &conf, uchar *cmpData,
    size_t cmpCap) {  // test lorenzo cmp on a set of sampled data blocks and return the compression ratio
    std::vector<int> total_quant_bins;
    // if ((N == 3 && !conf.regression2) || (N == 1 && !conf.regression && !conf.regression2)) {
    std::vector<std::shared_ptr<concepts::PredictorInterface<T, N>>> predictors;
    predictors.push_back(std::make_shared<LorenzoPredictor<T, N, 1>>(conf.absErrorBound));
    predictors.push_back(std::make_shared<LorenzoPredictor<T, N, 2>>(conf.absErrorBound));
    auto falcons = make_decomposition_blockwise<T, N>(conf, ComposedPredictor<T, N>(predictors),
                                                 LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2));
    // auto falcons = make_decomposition_lorenzo_regression<T, N>(conf, LinearQuantizer<T>(conf.absErrorBound,
    // conf.quantbinCnt / 2));
    for (size_t k = 0; k < sampled_blocks.size(); k++) {
        auto cur_block = sampled_blocks[k];
        // auto (quant_bins) = falcons.compress(conf, cur_block.data());
        // total_quant_bins.insert(total_quant_bins.end(), quant_bins.begin(),
        //                         quant_bins.end());  // merge the quant bins. Lossless them together
        auto [quant_bins, quant_bins_size] = falcons.compress(conf, cur_block.data());
        total_quant_bins.insert(total_quant_bins.end(), quant_bins, quant_bins + quant_bins_size);
        delete[] quant_bins;   // falcons.compress now returns a new[] array owned by us
    }
    auto encoder = HuffmanEncoder<int>();
    auto lossless = Lossless_zstd();
    encoder.preprocess_encode(total_quant_bins, conf.quantbinCnt);
    size_t bufferSize = std::max<size_t>(1000, 1.2 * (encoder.size_est() + sizeof(T) * total_quant_bins.size()));

    auto buffer = static_cast<uchar *>(malloc(bufferSize));
    uchar *buffer_pos = buffer;
    falcons.save(buffer_pos);
    encoder.save(buffer_pos);

    // store the size of quant_inds is necessary as it is not always equal to conf.num
    write<size_t>(total_quant_bins.size(), buffer_pos);
    encoder.encode(total_quant_bins, buffer_pos);
    encoder.postprocess_encode();
    auto cmpSize = lossless.compress(buffer, buffer_pos - buffer, cmpData, cmpCap);
    free(buffer);
    auto compression_ratio = conf.num * sampled_blocks.size() * sizeof(T) * 1.0 / cmpSize;
    return compression_ratio;

    return 0;
}

template <class T, uint N>
size_t Falcons_compress_Interp_lorenzo(Config &conf, T *data, uchar *cmpData, size_t cmpCap) {
    assert(conf.cmprAlgo == ALGO_INTERP_LORENZO);

#ifdef Falcons_PRINT_TIMINGS
    Timer timer(true);
#endif
    calAbsErrorBound(conf, data);

    if (conf.interpAnchorStride < 0) {  // set default anchor stride
        std::array<size_t, 4> anchor_strides = {4096, 128, 32, 16};
        conf.interpAnchorStride = anchor_strides[N - 1];
    }

    std::array<double, 4> sample_Rates = {0.005, 0.005, 0.005,
                                          0.005};  // default data sample rate. todo: add a config var to control
    auto sampleRate = sample_Rates[N - 1];
    std::array<size_t, 4> sampleBlock_Sizes = {4096, 128, 32,
                                               16};  // default sampled data block rate. Should better be no smaller
                                                     // than the anchor stride. todo: add a config var to control
    size_t sampleBlockSize = sampleBlock_Sizes[N - 1];
    size_t shortest_edge = conf.dims[0];
    for (size_t i = 0; i < N; i++) {
        shortest_edge = conf.dims[i] < shortest_edge ? conf.dims[i] : shortest_edge;
    }
    // Automatically adjust sampleblocksize.
    while (sampleBlockSize >= shortest_edge) sampleBlockSize /= 2;
    while (sampleBlockSize >= 16 && (pow(sampleBlockSize + 1, N) / conf.num) > 1.5 * sampleRate) sampleBlockSize /= 2;
    if (sampleBlockSize < 8) sampleBlockSize = 8;

    bool to_tune = pow(sampleBlockSize + 1, N) <= 0.05 * conf.num;  // to further revise
    for (auto &dim : conf.dims) {
        if (dim < sampleBlockSize) {
            to_tune = false;
            break;
        }
    }

    if (!to_tune) {  // if the sampled data would be too many (currently it is 5% of the input), skip the tuning
        conf.cmprAlgo = ALGO_INTERP;
        return Falcons_compress_Interp<T, N>(conf, data, cmpData, cmpCap);
    }
#ifdef Falcons_PRINT_TIMINGS
    timer.stop("preparation");
    timer.start();
#endif
    std::vector<std::vector<T>> sampled_blocks;
    size_t per_block_ele_num = pow(sampleBlockSize + 1, N);
    size_t sampling_num;
    std::vector<std::vector<size_t>> starts;
    auto profStride = sampleBlockSize / 4;  // larger is faster, smaller is better
    profiling_block<T, N>(data, conf.dims, starts, sampleBlockSize, conf.absErrorBound,
                          profStride, conf.openmp);  // filter out the non-constant data blocks
    size_t num_filtered_blocks = starts.size();
    bool profiling = num_filtered_blocks * per_block_ele_num >= 0.5 * sampleRate * conf.num;  // temp. to refine
    // bool profiling = false;
    sampleBlocks<T, N>(data, conf.dims, sampleBlockSize, sampled_blocks, sampleRate, profiling,
                       starts, 0, conf.openmp);  // sample out same data blocks
    sampling_num = sampled_blocks.size() * per_block_ele_num;

    if (sampling_num == 0 || sampling_num >= conf.num * 0.2) {
        conf.cmprAlgo = ALGO_INTERP;
        return Falcons_compress_Interp<T, N>(conf, data, cmpData, cmpCap);
    }
#ifdef Falcons_PRINT_TIMINGS
    timer.stop("sampling");
    Timer timer_tuning(true);
    timer_tuning.start();
    timer.start();
#endif
    double best_lorenzo_ratio = 0, best_interp_ratio = 0, ratio;
    size_t bufferCap = conf.num * sizeof(T);
    auto buffer = static_cast<uchar *>(malloc(bufferCap));
    Config lorenzo_config = conf;

    {
        // tune interp
        conf.interpDirection = 0;
        if constexpr (N <= 2) {
            conf.interpAlpha = 1.0;
            conf.interpBeta = 1.0;
        } else {
            conf.interpAlpha = 1.25;
            conf.interpBeta = 2.0;
        }
        std::vector<size_t> dims(N, sampleBlockSize + 1);
        uint8_t reverse = factorial(N) - 1;

        const int ablist_size = 3;
        auto alphalist = N <= 2
            ? std::array<double, ablist_size>{1.0, 1.25, 1.5}
            : std::array<double, ablist_size>{1.25, 1.0, 1.5};
        auto betalist = N <= 2
            ? std::array<double, ablist_size>{1.0, 2.0, 2.5}
            : std::array<double, ablist_size>{2.0, 1.0, 2.5};

        int max_t = 1;
#ifdef _OPENMP
        max_t = omp_get_max_threads();
#endif

        if (N >= 3) {
            double ratios[4];
            Config tcs[4];
            for (int i = 0; i < 4; i++) {
                tcs[i] = conf;
                tcs[i].setDims(dims.begin(), dims.end());
                tcs[i].interpAlgo      = (i / 2 == 0) ? INTERP_ALGO_LINEAR : INTERP_ALGO_CUBIC;
                tcs[i].interpDirection = (i % 2 == 0) ? 0 : reverse;
            }
            interp_compress_test_batch<T, N>(sampled_blocks, tcs, 4, sampleBlockSize, ratios);
            for (int i = 0; i < 4; i++) {
                if (ratios[i] > best_interp_ratio) {
                    best_interp_ratio = ratios[i];
                    conf.interpAlgo      = (i / 2 == 0) ? INTERP_ALGO_LINEAR : INTERP_ALGO_CUBIC;
                    conf.interpDirection = (i % 2 == 0) ? 0 : reverse;
                }
            }
        }
        else if (max_t >= 6) {
            // Keep both alpha/beta groups in one OpenMP task pool.
            constexpr int algo_direction_count = 4;
            constexpr int high_thread_ab_count = 2;
            constexpr int candidate_count = algo_direction_count * high_thread_ab_count;
            constexpr std::array<double, high_thread_ab_count> high_thread_alphas{1.0, 1.25};
            constexpr std::array<double, high_thread_ab_count> high_thread_betas{1.0, 2.0};
            std::array<double, candidate_count> candidate_ratios;
            Config candidate_confs[candidate_count];
            for (int i = 0; i < candidate_count; i++) {
                const int algo_direction = i % algo_direction_count;
                const int ab = i / algo_direction_count;
                candidate_confs[i] = conf;
                candidate_confs[i].setDims(dims.begin(), dims.end());
                candidate_confs[i].interpAlgo =
                    (algo_direction / 2 == 0) ? INTERP_ALGO_LINEAR : INTERP_ALGO_CUBIC;
                candidate_confs[i].interpDirection =
                    (algo_direction % 2 == 0) ? 0 : reverse;
                candidate_confs[i].interpAlpha = high_thread_alphas[ab];
                candidate_confs[i].interpBeta = high_thread_betas[ab];
            }
            interp_compress_test_batch<T, N>(sampled_blocks, candidate_confs, candidate_count,
                                               sampleBlockSize, candidate_ratios.data());
            for (int i = 0; i < candidate_count; i++) {
                if (candidate_ratios[i] > best_interp_ratio) {
                    best_interp_ratio = candidate_ratios[i];
                    conf.interpAlgo = candidate_confs[i].interpAlgo;
                    conf.interpDirection = candidate_confs[i].interpDirection;
                    conf.interpAlpha = candidate_confs[i].interpAlpha;
                    conf.interpBeta = candidate_confs[i].interpBeta;
                }
            }
        } else {
            // two-step: 4 algo×dir, then 3 alpha×beta (fewer threads → steps beat fused)
            double ratios[4];
            #ifdef _OPENMP
            #pragma omp parallel for if(conf.openmp)
            #endif
            for (int i = 0; i < 4; i++) {
                auto tc = conf;
                tc.setDims(dims.begin(), dims.end());
                tc.interpAlgo      = (i / 2 == 0) ? INTERP_ALGO_LINEAR : INTERP_ALGO_CUBIC;
                tc.interpDirection = (i % 2 == 0) ? 0 : reverse;
                auto buf = static_cast<uchar *>(malloc(bufferCap));
                ratios[i] = interp_compress_test<T, N>(sampled_blocks, tc, sampleBlockSize, buf, bufferCap);
                free(buf);
            }
            for (int i = 0; i < 4; i++) {
                if (ratios[i] > best_interp_ratio) {
                    best_interp_ratio = ratios[i];
                    conf.interpAlgo      = (i / 2 == 0) ? INTERP_ALGO_LINEAR : INTERP_ALGO_CUBIC;
                    conf.interpDirection = (i % 2 == 0) ? 0 : reverse;
                }
            }
            // best algo×dir chosen → tune alpha×beta on top
            std::array<double, ablist_size> ab_ratios;
            #ifdef _OPENMP
            #pragma omp parallel for if(conf.openmp)
            #endif
            for (int i = 1; i < ablist_size ; i++) {
                auto tc = conf;
                tc.setDims(dims.begin(), dims.end());
                tc.interpAlpha = alphalist[i];
                tc.interpBeta  = betalist[i];
                auto buf = static_cast<uchar *>(malloc(bufferCap));
                ab_ratios[i] = interp_compress_test<T, N>(sampled_blocks, tc, sampleBlockSize, buf, bufferCap);
                free(buf);
            }
            for (int i = 1; i < ablist_size; i++) {
                if (ab_ratios[i] > best_interp_ratio * 1.02) {
                    best_interp_ratio = ab_ratios[i];
                    conf.interpAlpha = alphalist[i];
                    conf.interpBeta  = betalist[i];
                }
            }
        }

    }
#ifdef Falcons_PRINT_TIMINGS
 timer.stop("interp tuning");
#endif
    
#ifdef Falcons_PRINT_TIMINGS
    timer.start();
#endif
    {
        
        // only test lorenzo for 1D
        if (N == 1 && best_interp_ratio < 50) {
            std::vector<size_t> sample_dims(N, sampleBlockSize + 1);
            lorenzo_config.cmprAlgo = ALGO_LORENZO_REG;
            lorenzo_config.setDims(sample_dims.begin(), sample_dims.end());
            lorenzo_config.lorenzo = true;
            lorenzo_config.lorenzo2 = true;
            lorenzo_config.regression = false;
            lorenzo_config.regression2 = false;
            lorenzo_config.openmp = false;
            lorenzo_config.blockSize = 5;
            //        lorenzo_config.quantbinCnt = 65536 * 2;
            best_lorenzo_ratio = lorenzo_compress_test<T, N>(sampled_blocks, lorenzo_config, buffer, bufferCap);
            //            delete[]cmprData;
            //    printf("Lorenzo ratio = %.2f\n", ratio);
        }
    }


    bool useInterp = !(best_lorenzo_ratio >= best_interp_ratio * 1.1 && best_lorenzo_ratio < 50 &&
                       best_interp_ratio < 50);  // 1.1 is a fix coefficient. subject to revise
    size_t cmpSize = 0;
#ifdef Falcons_PRINT_TIMINGS
    timer.stop("lorenzo tuning");
    timer_tuning.stop("total tuning");
    timer.start();
#endif
    if (useInterp) {
        conf.cmprAlgo = ALGO_INTERP;
        cmpSize = Falcons_compress_Interp<T, N>(conf, data, cmpData, cmpCap);
    } else {
        // no need to tune lorenzo for 3D anymore
        // if (N == 3) {
        //     float pred_freq, mean_freq;
        //     T mean_guess;
        //     lorenzo_config.quantbinCnt = optimize_quant_invl_3d<T>(
        //         data, conf.dims[0], conf.dims[1], conf.dims[2], conf.absErrorBound, pred_freq, mean_freq,
        //         mean_guess);
        //     lorenzo_config.pred_dim = 2;
        //     ratio  =
        //         lorenzo_compress_test<T, N>(sampled_blocks, lorenzo_config, buffer, bufferCap);
        //     if (ratio > best_lorenzo_ratio * 1.02) {
        //         best_lorenzo_ratio = ratio;
        //     } else {
        //         lorenzo_config.pred_dim = 3;
        //     }
        // }

        if (conf.relErrorBound < 1.01e-6 && best_lorenzo_ratio > 5 && lorenzo_config.quantbinCnt != 16384) {
            auto quant_num = lorenzo_config.quantbinCnt;
            lorenzo_config.quantbinCnt = 16384;
            ratio = lorenzo_compress_test<T, N>(sampled_blocks, lorenzo_config, buffer, bufferCap);
            if (ratio > best_lorenzo_ratio * 1.02) {
                best_lorenzo_ratio = ratio;
            } else {
                lorenzo_config.quantbinCnt = quant_num;
            }
        }
        lorenzo_config.setDims(conf.dims.begin(), conf.dims.end());
        conf = lorenzo_config;
        //            double tuning_time = timer.stop();
        cmpSize = Falcons_compress_LorenzoReg<T, N>(conf, data, cmpData, cmpCap);
    }

    free(buffer);
    return cmpSize;
}
}  // namespace Falcons
#endif
