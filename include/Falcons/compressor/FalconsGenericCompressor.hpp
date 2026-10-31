/**
 * @file FalconsGenericCompressor.hpp
 * @author Bing Lu
 * @brief Falcons compression pipeline and entropy-coder integration.
 *
 * Copyright (c) 2026 Bing Lu.
 */

#ifndef Falcons_COMPRESSOR_TYPE_ONE_HPP
#define Falcons_COMPRESSOR_TYPE_ONE_HPP

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
#include "Falcons/quantizer/LinearQuantizer.hpp"
#include "Falcons/decomposition/InterpolationDecomposition.hpp"

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
class FalconsGenericCompressor : public concepts::CompressorInterface<T> {
   public:
    FalconsGenericCompressor(Decomposition decomposition, Encoder encoder, Lossless lossless)
        : decomposition(decomposition), encoder(encoder), lossless(lossless) {
        using QuantT = std::remove_pointer_t<std::tuple_element_t<0,
            decltype(std::declval<Decomposition&>().compress(std::declval<const Config&>(), std::declval<T*>()))>>;
        static_assert(std::is_base_of<concepts::DecompositionInterface<T, QuantT, N>, Decomposition>::value,
                      "must implement the frontend interface");
        static_assert(std::is_base_of<concepts::EncoderInterface<QuantT>, Encoder>::value,
                      "must implement the encoder interface");
        static_assert(std::is_base_of<concepts::LosslessInterface, Lossless>::value,
                      "must implement the lossless interface");
    }
    using TargetDecomposition =
    decltype(make_decomposition_interpolation<TUNING::DISABLED, T, N>(
        std::declval<Config>(),
        std::declval<LinearQuantizer<T>>()
    ));

    size_t compress(const Config &conf, T *data, uchar *cmpData, size_t cmpCap) override {
#ifdef Falcons_PRINT_TIMINGS
        Timer timer(true);
#endif
        auto [quant_inds, quant_inds_size] = decomposition.compress(conf, data);
#ifdef Falcons_PRINT_TIMINGS
        timer.stop("cmp interp");
        timer.start();
#endif

        size_t bufferSize = std::max<size_t>(
            1000, 1.2 * (decomposition.size_est() + encoder.size_est_without_init() + (sizeof(T) + 5) * quant_inds_size));
        auto buffer = static_cast<uchar *>(malloc(bufferSize));
        uchar *buffer_pos = buffer;
        auto cmpDataPos = cmpData;
        {
            decomposition.save(buffer_pos);
            write<uint64_t>(quant_inds_size, buffer_pos);
            encoder.preprocess_encode(quant_inds, quant_inds_size, decomposition.get_out_range().second);
            encoder.save(buffer_pos);
            encoder.encode(quant_inds, quant_inds_size, buffer_pos);
            encoder.postprocess_encode();
#ifdef Falcons_PRINT_TIMINGS
        timer.stop("cmp lossless0");
        timer.start();
#endif
            auto currentSize = buffer_pos - buffer;
            cmpCap -= cmpDataPos-cmpData;

            auto zstdSize = lossless.compress(buffer, currentSize, cmpDataPos, cmpCap);
            cmpDataPos+=zstdSize;
            free(buffer);
        }
        delete[] quant_inds;
#ifdef Falcons_PRINT_TIMINGS
        timer.stop("cmp zstd");
#endif
        return cmpDataPos-cmpData;
    }

    T *decompress(const Config &conf, uchar const *cmpData, size_t cmpSize, T *decData) override {

#ifdef Falcons_PRINT_TIMINGS
        Timer timer(true);
#endif
        auto cmpDataPos = cmpData;
        uint64_t quant_inds_size = 0;

        uchar *buffer = nullptr;
        size_t bufferSize = 0;

        cmpSize -= cmpDataPos - cmpData; 
        lossless.decompress(cmpDataPos, cmpSize, buffer, bufferSize);
#ifdef Falcons_PRINT_TIMINGS
        timer.stop("decomp zstd");
        timer.start();
#endif
        uchar const *bufferPos = buffer;
        decomposition.load(bufferPos, bufferSize);
        read(quant_inds_size, bufferPos);
        encoder.load(bufferPos, bufferSize);
        auto quant_inds = encoder.decode(bufferPos, quant_inds_size);
        free(buffer);
#ifdef Falcons_PRINT_TIMINGS
        timer.stop("decomp lossless0");
        timer.start();
#endif
        decomposition.decompress(conf, quant_inds, decData);
#ifdef Falcons_PRINT_TIMINGS
    timer.stop("decomp interp");
#endif
        delete[] quant_inds;
        return decData;
    }

   private:
    Decomposition decomposition;
    Encoder encoder;
    Lossless lossless;
};

template <class T, uint N, class Decomposition, class Encoder, class Lossless>
std::shared_ptr<FalconsGenericCompressor<T, N, Decomposition, Encoder, Lossless>> make_compressor_falcons_generic(
    Decomposition decomposition, Encoder encoder, Lossless lossless) {
    return std::make_shared<FalconsGenericCompressor<T, N, Decomposition, Encoder, Lossless>>(decomposition, encoder,
                                                                                         lossless);
}

}  // namespace Falcons
#endif
