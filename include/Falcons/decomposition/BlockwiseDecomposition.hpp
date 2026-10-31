#ifndef Falcons_BLOCKWISE_DECOMPOSITION_HPP
#define Falcons_BLOCKWISE_DECOMPOSITION_HPP

#include <cstring>

#include "Decomposition.hpp"
#include "Falcons/def.hpp"
#include "Falcons/predictor/LorenzoPredictor.hpp"
#include "Falcons/predictor/Predictor.hpp"
#include "Falcons/quantizer/LinearQuantizer.hpp"
#include "Falcons/utils/Config.hpp"
#include "Falcons/utils/FileUtil.hpp"
#include "Falcons/utils/BlockwiseIterator.hpp"
#include "Falcons/utils/Timer.hpp"

namespace Falcons {
template <class T, uint N, class Predictor, class Quantizer>
class BlockwiseDecomposition : public concepts::DecompositionInterface<T, int, N> {
   public:
    using Block_iter = typename block_data<T, N>::block_iterator;

    BlockwiseDecomposition(const Config &conf, Predictor predictor, Quantizer quantizer)
        : predictor(predictor), quantizer(quantizer), fallback_predictor(conf.absErrorBound) {
        static_assert(std::is_base_of<concepts::PredictorInterface<T, N>, Predictor>::value,
                      "must implement the Predictor interface");
    }

    std::tuple<int*, uint64_t>  compress(const Config &conf, T *data) override {
        auto data_with_padding = std::make_shared<block_data<T, N>>(data, conf.dims, predictor.get_padding(), true);
        auto block = data_with_padding->block_iter(conf.blockSize);
        int *quant_inds = new int[conf.num];   // caller owns it (delete[]); same contract as InterpolationDecomposition
        uint64_t cnt = 0;
        const int radius = quantizer.get_out_range().second / 2;   // =32768 by default; bias codes by +radius so
                                                                   // signed lorenzo codes become non-negative [0,2*radius),
                                                                   // matching HuffmanEncoder<int>'s [0,1<<16) freq table
        do {
            concepts::PredictorInterface<T, N> *predictor_withfallback = &predictor;
            if (!predictor.precompress(block)) {
                predictor_withfallback = &fallback_predictor;
            }
            predictor_withfallback->precompress_block_commit();
            Block_iter::foreach (block, [&](T *c, const std::array<size_t, N> &index) {
                T pred = predictor_withfallback->predict(block, c, index);
                quant_inds[cnt++] = quantizer.quantize_and_overwrite(*c, pred) + radius;   // +radius bias (escape -radius -> 0)
            });

        } while (block.next());
        return {quant_inds, cnt};
    }

    T *decompress(const Config &conf, int* quant_inds, T *dec_data) override {
        int *quant_inds_pos = quant_inds;
        const int radius = quantizer.get_out_range().second / 2;   // un-bias: stored code - radius -> signed code
                                                                   // (escape 0 -> -radius, triggers recover_unpred)

        auto data_with_padding =
            std::make_shared<block_data<T, N>>(dec_data, conf.dims, predictor.get_padding(), false);
        auto block = data_with_padding->block_iter(conf.blockSize);
        do {
            concepts::PredictorInterface<T, N> *predictor_withfallback = &predictor;
            if (!predictor.predecompress(block)) {
                predictor_withfallback = &fallback_predictor;
            }
            Block_iter::foreach (block, [&](T *c, const std::array<size_t, N> &index) {
                T pred = predictor_withfallback->predict(block, c, index);
                *c = quantizer.recover(pred, *(quant_inds_pos++) - radius);   // -radius un-bias
            });

        } while (block.next());

        return dec_data;
    }

    void save(uchar *&c) override {
        fallback_predictor.save(c);
        predictor.save(c);
        quantizer.save(c);
    }

    void load(const uchar *&c, size_t &remaining_length) override {
        fallback_predictor.load(c, remaining_length);
        predictor.load(c, remaining_length);
        quantizer.load(c, remaining_length);
    }

    std::pair<int, int> get_out_range() override { return quantizer.get_out_range(); }

    // Outliers, plus room for the predictors' saved tables on small inputs.
    size_t size_est() override { return quantizer.size_est() + (1 << 20); }

   private:
    Predictor predictor;
    Quantizer quantizer;
    LorenzoPredictor<T, N, 1> fallback_predictor;
};

template <class T, uint N, class Predictor, class Quantizer>
BlockwiseDecomposition<T, N, Predictor, Quantizer> make_decomposition_blockwise(const Config &conf, Predictor predictor,
                                                                                Quantizer quantizer) {
    return BlockwiseDecomposition<T, N, Predictor, Quantizer>(conf, predictor, quantizer);
}

}  // namespace Falcons
#endif
