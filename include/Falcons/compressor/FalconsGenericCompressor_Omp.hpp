/**
 * @file FalconsGenericCompressor_Omp.hpp
 * @author Bing Lu
 * @brief Falcons parallel compression pipeline.
 *
 * Copyright (c) 2026 Bing Lu.
 */

#ifndef Falcons_COMPRESSOR_TYPE_ONE_OMP_HPP
#define Falcons_COMPRESSOR_TYPE_ONE_OMP_HPP

#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "Falcons/compressor/Compressor.hpp"
#include "Falcons/decomposition/Decomposition.hpp"
#include "Falcons/def.hpp"
#include "Falcons/encoder/Encoder.hpp"
#include "Falcons/lossless/Lossless.hpp"
#include "Falcons/utils/Config.hpp"
#include "Falcons/utils/FileUtil.hpp"
#include "Falcons/utils/Timer.hpp"

namespace Falcons {
/**
 * FalconsGenericCompressor glues together decomposition, encoder, and lossless modules to form the compressor.
 * It only takes Decomposition, not Predictor.
 * @tparam T original data type
 * @tparam N original data dimension
 * @tparam Decomposition decomposition module
 * @tparam Encoder encoder module
 * @tparam Lossless lossless module
 */

template <class T, uint N, class Decomposition, class Encoder, class Lossless>
class FalconsGenericCompressor_OMP : public concepts::CompressorInterface<T> {
   public:
    FalconsGenericCompressor_OMP(Decomposition decomposition, Encoder encoder, Lossless lossless)
        : decomposition(decomposition), encoder(encoder), lossless(lossless) {
        using QuantT = std::remove_pointer_t<std::remove_pointer_t<std::tuple_element_t<0,
            decltype(std::declval<Decomposition&>().compress(std::declval<const Config&>(), std::declval<T*>()))>>>;
        static_assert(std::is_base_of<concepts::DecompositionInterface_OMP<T, QuantT, N>, Decomposition>::value,
                      "must implement the frontend interface");
        static_assert(std::is_base_of<concepts::EncoderInterface<QuantT>, Encoder>::value,
                      "must implement the encoder interface");
        static_assert(std::is_base_of<concepts::LosslessInterface, Lossless>::value,
                      "must implement the lossless interface");
    }

    size_t compress(const Config &conf, T *data, uchar *cmpData, size_t cmpCap) override {
#ifdef Falcons_PRINT_TIMINGS
        Timer timer(true);
#endif
#ifdef _OPENMP
        omp_set_dynamic(0);   // every parallel region must see the same team
#endif
        auto [local_quant_inds, local_quant_index] = decomposition.compress(conf, data);
#ifdef Falcons_PRINT_TIMINGS
        timer.stop("cmp interp");
        timer.start();
#endif


        auto cmpDataPos = cmpData;
        #ifdef _OPENMP
        auto default_nthreads = omp_get_max_threads();
        //std::cout<<default_nthreads<<" "<<quant_inds_size<<std::endl;
        // One stream per thread that really filled one: the decomposition's team can be smaller.
        auto best_num_threads = decomposition.num_threads();
        //std::cout<<best_num_threads<<std::endl;
        if (best_num_threads > 1) {
            // size_t cur_bufferSize = std::max<size_t>(1000, 2 * 1.2 * sizeof(T) * local_quant_index[0].value * best_num_threads);
            // uchar* buffer = new (std::align_val_t(64)) uchar[cur_bufferSize * best_num_threads];
            omp_set_num_threads(best_num_threads);
            //uchar * offset_block_pos = buffer_pos + sizeof(int);
            //size_t bins_per_thread;
            // auto quant_inds_data = quant_inds;
            std::vector<size_t>block_byte_offsets;
            // size_t offset_quant_inds_size;
            // size_t total_huffman_size = 0;
            int nthreads = best_num_threads;
            size_t offset_chunk_size = nthreads * sizeof (size_t);
            size_t total_zstd_size = 0;
            bool fits_cmpCap = true;
            block_byte_offsets.resize(nthreads);
            write<int>(nthreads, cmpDataPos);
            // Timer timer1(true);
            #pragma omp parallel
            {
                // #pragma omp single 
                // {
                //     Timer timer1(true);
                // }   
                Encoder cur_encoder;
                auto tid = omp_get_thread_num();
                // auto temp_buffer_pos0 = buffer_pos + tid * sizeof(int);
                // decomposition.save3(temp_buffer_pos0, tid);
                // write<int>(local_quant_index[tid].value, temp_buffer_pos0);
                // Outliers are stored raw on top of their codes, and anchors have no code at all.
                size_t cur_bufferSize = std::max<size_t>(1000, 1.2 * (cur_encoder.size_est_without_init() + 4096 +
                    (sizeof(T) + 2) * local_quant_index[tid].value + sizeof(T) * decomposition.local_unpred_count(tid)));
                auto cur_buffer = static_cast<uchar *>(malloc(cur_bufferSize)); 
                // uchar* cur_buffer = new (std::align_val_t(64)) uchar[cur_bufferSize];
                // auto cur_buffer = buffer + tid * cur_bufferSize;
                auto cur_buffer_pos = cur_buffer;
                if(tid == best_num_threads - 1)
                    decomposition.save2(cur_buffer_pos);

                decomposition.save3(cur_buffer_pos, tid);
                write<int>(local_quant_index[tid].value, cur_buffer_pos);
                cur_encoder.preprocess_encode(local_quant_inds[tid], local_quant_index[tid].value, 
                    decomposition.get_out_range().second);
                cur_encoder.save(cur_buffer_pos);
                cur_encoder.encode(local_quant_inds[tid], local_quant_index[tid].value, cur_buffer_pos);
                cur_encoder.postprocess_encode();
                auto cur_outSize = cur_buffer_pos - cur_buffer;

                size_t lossless_bufferSize = std::max<size_t>(1000, 1.2 * sizeof(uchar) * cur_outSize);
                auto lossless_buffer = static_cast<uchar *>(malloc(lossless_bufferSize)); 
                auto lossless_buffer_pos = lossless_buffer; 
                auto lossless_outSize = lossless.compress(cur_buffer, cur_outSize, lossless_buffer_pos, lossless_bufferSize);
                block_byte_offsets[tid] = lossless_outSize;
  
                #pragma omp barrier
                #pragma omp single
                {
                    size_t prefix_sum = 0;
                    
                    for (int i = 0; i < nthreads; i++){
                        auto tmp = block_byte_offsets[i];
                        block_byte_offsets[i] = prefix_sum;
                        prefix_sum =  prefix_sum + tmp;
                    }
                    total_zstd_size =  prefix_sum;
                    fits_cmpCap = (cmpDataPos - cmpData) + offset_chunk_size + total_zstd_size <= cmpCap;
                }
                if (fits_cmpCap) {
                    auto temp_cmpData_pos = cmpDataPos + tid * sizeof(size_t);
                    write<size_t>(block_byte_offsets[tid],temp_cmpData_pos);
                    temp_cmpData_pos = cmpDataPos + offset_chunk_size + block_byte_offsets[tid];
                    write<uchar>(lossless_buffer, lossless_outSize,  temp_cmpData_pos);
                }

                free(lossless_buffer);
                free(cur_buffer);

            }
            // free(buffer);
            omp_set_num_threads(default_nthreads);
            if (!fits_cmpCap) throw std::length_error(Falcons_ERROR_COMP_BUFFER_NOT_LARGE_ENOUGH);
            cmpDataPos += offset_chunk_size + total_zstd_size;
        }
            
        else{
            size_t quant_inds_size = local_quant_index[0].value;
            size_t bufferSize = std::max<size_t>(
            1000, 1.2 * (decomposition.size_est() + encoder.size_est_without_init() + sizeof(T) * quant_inds_size));

            auto buffer = static_cast<uchar *>(malloc(bufferSize));
            uchar *buffer_pos = buffer;

            write<int>(1, cmpDataPos); //1 thread
            decomposition.save2(buffer_pos);
            decomposition.save3(buffer_pos, 0);
            write<size_t>(quant_inds_size, buffer_pos);
            // Timer timer1(true);
            encoder.preprocess_encode(local_quant_inds[0], local_quant_index[0].value, 
                decomposition.get_out_range().second);
                    //                         timer1.stop("preprocess ");
                    // timer1.start();
            encoder.save(buffer_pos);
            encoder.encode(local_quant_inds[0], local_quant_index[0].value, buffer_pos);
            //   timer1.stop("encode ");
            encoder.postprocess_encode();
            auto currentSize = buffer_pos - buffer;
            cmpCap-=cmpDataPos-cmpData;

            write<size_t>(0, cmpDataPos); //offset = 0;
            auto zstdSize = lossless.compress(buffer, currentSize, cmpDataPos, cmpCap);
            cmpDataPos+=zstdSize;
            free(buffer);
        }
        #else
            size_t bufferSize = std::max<size_t>(
            1000, 1.2 * (decomposition.size_est() + encoder.size_est_without_init() + sizeof(T) * quant_inds_size));

            auto buffer = static_cast<uchar *>(malloc(bufferSize));
            uchar *buffer_pos = buffer;

            write<int>(1, cmpDataPos); //1 thread
            decomposition.save2(buffer_pos);
            decomposition.save3(buffer_pos, 0);
            write<size_t>(quant_inds_size, buffer_pos);
            encoder.preprocess_encode(local_quant_inds[0], local_quant_index[0].value, decomposition.get_out_range().second);
            encoder.save(buffer_pos);
            encoder.encode(local_quant_inds[0], local_quant_index[0].value, buffer_pos);
            encoder.postprocess_encode();
            auto currentSize = buffer_pos - buffer;
            cmpCap-=cmpDataPos-cmpData;

            write<size_t>(0, cmpDataPos); //offset = 0;
            auto zstdSize = lossless.compress(buffer, currentSize, cmpDataPos, cmpCap);
            cmpDataPos+=zstdSize;
            free(buffer);
        #endif
#ifdef Falcons_PRINT_TIMINGS
        timer.stop("cmp all lossless");
#endif

        return cmpDataPos-cmpData;
    }

    T *decompress(const Config &conf, uchar const *cmpData, size_t cmpSize, T *decData) override {
#ifdef Falcons_PRINT_TIMINGS
        Timer timer(true);
#endif
        auto cmpDataPos = cmpData;
        int compression_thread_num = 0;


        //lossless.decompress(cmpData, cmpSize, buffer, bufferSize);
        // read(huffSize, cmpDataPos);
        read(compression_thread_num, cmpDataPos);
        // The stream fixes the team: every thread owns one code stream, whatever the caller's setting.
        const int caller_threads = omp_get_max_threads();
        omp_set_dynamic(0);
        omp_set_num_threads(compression_thread_num);
        int team = 1;
        #pragma omp parallel
        #pragma omp single
        team = omp_get_num_threads();
        if (team != compression_thread_num) {
            omp_set_num_threads(caller_threads);
            throw std::runtime_error("this Falcons stream needs " + std::to_string(compression_thread_num) +
                                     " OpenMP threads but only " + std::to_string(team) + " are available");
        }
        int16_t** local_quant_inds = new int16_t*[compression_thread_num];
        int* local_quant_index = new int[compression_thread_num];
        decomposition.init_local_unpred(compression_thread_num);
        if(compression_thread_num <= 1){
            uchar *buffer = nullptr;
            size_t bufferSize = 0;
            size_t offset;
            read(offset, cmpDataPos);
            cmpSize -= cmpDataPos - cmpData; 
            lossless.decompress(cmpDataPos, cmpSize, buffer, bufferSize);
            size_t quant_inds_size = 0;
            uchar const *bufferPos = buffer;
            decomposition.load2(bufferPos, bufferSize);
            decomposition.load3(bufferPos, bufferSize, 0);
            read(quant_inds_size, bufferPos);
            local_quant_index[0] = quant_inds_size;
            // std::cout << "quant_inds_size: " << quant_inds_size << std::endl;
            encoder.load(bufferPos, bufferSize);
            local_quant_inds[0] = encoder.decode(bufferPos, quant_inds_size);
            encoder.postprocess_decode();
            // decomposition.decompress(conf, local_quant_inds, decData);
            // return decData;
        }
        else{
            std::vector<size_t>block_byte_offsets(compression_thread_num);
            std::vector<size_t>output_block_byte_offsets(compression_thread_num);

            read(block_byte_offsets.data(),compression_thread_num,cmpDataPos);
            cmpSize -= cmpDataPos - cmpData;
            // #ifdef _OPENMP
            // #pragma omp parallel for 
            // #endif
            // for(int tid=0; tid < compression_thread_num;tid++){
            #pragma omp parallel num_threads(compression_thread_num)
            {
                auto tid = omp_get_thread_num();
                //size_t start_idx = (static_cast<size_t>(tid) * huffSize) / static_cast<size_t>(compression_thread_num), cur_len = (static_cast<size_t>(tid+1) * huffSize) / static_cast<size_t>(compression_thread_num) - start_idx;
                size_t block_byte_offset = block_byte_offsets[tid];
                //std::cout<<tid<<" "<<block_byte_offset<<std::endl;

                size_t block_byte_length = (tid != compression_thread_num-1)? block_byte_offsets[tid + 1] - block_byte_offset : cmpSize - block_byte_offset;
               // #pragma omp critical
                //std::cout<<"tid: "<<tid<<", prefix: "<<block_byte_offset<<std::endl;
                auto temp_cmp_pos = cmpDataPos + block_byte_offset;
                uchar* cur_buffer = nullptr;
                size_t cur_bufferSize = 0;
                lossless.decompress(temp_cmp_pos, block_byte_length, cur_buffer, cur_bufferSize);
                uchar const* cur_buffer_pos = cur_buffer;
                if(tid == compression_thread_num - 1)
                    decomposition.load2(cur_buffer_pos, cur_bufferSize);
                #pragma omp barrier
                decomposition.load3(cur_buffer_pos, cur_bufferSize, tid);
                read<int>(local_quant_index[tid], cur_buffer_pos);

                Encoder cur_encoder;
                cur_encoder.load(cur_buffer_pos, cur_bufferSize);
                // std::cout << "Thread " << tid << " encoded size: " << local_quant_index[tid] << std::endl;
                local_quant_inds[tid] = cur_encoder.decode(cur_buffer_pos, local_quant_index[tid]);
                cur_encoder.postprocess_decode();
                free(cur_buffer);
            }
        }

#ifdef Falcons_PRINT_TIMINGS
        timer.stop("decmp all lossless");
        timer.start();
#endif
        decomposition.decompress(conf, local_quant_inds, decData);
        omp_set_num_threads(caller_threads);
#ifdef Falcons_PRINT_TIMINGS
    timer.stop("decomp interp");
#endif
        return decData;
    }

   private:
    Decomposition decomposition;
    Encoder encoder;
    Lossless lossless;
};

template <class T, uint N, class Decomposition, class Encoder, class Lossless>
std::shared_ptr<FalconsGenericCompressor_OMP<T, N, Decomposition, Encoder, Lossless>> make_compressor_falcons_generic_omp(
    Decomposition decomposition, Encoder encoder, Lossless lossless) {
    return std::make_shared<FalconsGenericCompressor_OMP<T, N, Decomposition, Encoder, Lossless>>(decomposition, encoder,
                                                                                         lossless);
}

}  // namespace Falcons
#endif
