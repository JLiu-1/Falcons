#ifndef Falcons_Falcons_HPP
#define Falcons_Falcons_HPP

#include "Falcons/api/impl/FalconsImpl.hpp"
#include "Falcons/version.hpp"

/**
 * API for compression
 * @tparam T source data type
 * @param config compression configuration. Please update the config with 1). data dimension and shape and 2). desired
settings.
 * @param data source data
 * @param cmpData pre-allocated buffer for compressed data
 * @param cmpCap pre-allocated buffer size (in bytes) for compressed data
 * @return compressed data size (in bytes)

The compression algorithms are:
ALGO_INTERP_LORENZO:
 The default hybrid interpolation/Lorenzo algorithm in Falcons.
 The whole dataset will be compressed by interpolation or lorenzo predictor with auto-optimized settings.
ALGO_INTERP:
 The whole dataset will be compressed by interpolation predictor with default settings.
ALGO_LORENZO_REG:
 The whole dataset will be compressed by lorenzo and/or regression based predictors block by block with default
settings. The four predictors ( 1st-order lorenzo, 2nd-order lorenzo, 1st-order regression, 2nd-order regression) can be
enabled or disabled independently by conf settings (lorenzo, lorenzo2, regression, regression2).

Interpolation+lorenzo example:
Falcons::Config conf(100, 200, 300); // 300 is the fastest dimension
conf.cmprAlgo = Falcons::ALGO_INTERP_LORENZO;
conf.errorBoundMode = Falcons::EB_ABS; // refer to def.hpp for all supported error bound mode
conf.absErrorBound = 1E-3; // absolute error bound 1e-3
char *compressedData = Falcons_compress(conf, data, outSize);

Interpolation example:
Falcons::Config conf(100, 200, 300); // 300 is the fastest dimension
conf.cmprAlgo = Falcons::ALGO_INTERP;
conf.errorBoundMode = Falcons::EB_REL; // refer to def.hpp for all supported error bound mode
conf.relErrorBound = 1E-3; // value-rang-based error bound 1e-3
char *compressedData = Falcons_compress(conf, data, outSize);

Lorenzo/regression example :
Falcons::Config conf(100, 200, 300); // 300 is the fastest dimension
conf.cmprAlgo = Falcons::ALGO_LORENZO_REG;
conf.lorenzo = true; // only use 1st order lorenzo
conf.lorenzo2 = false;
conf.regression = false;
conf.regression2 = false;
conf.errorBoundMode = Falcons::EB_ABS; // refer to def.hpp for all supported error bound mode
conf.absErrorBound = 1E-3; // absolute error bound 1e-3
char *compressedData = Falcons_compress(conf, data, outSize);
 */
template <class T>
size_t Falcons_compress(const Falcons::Config &config, const T *data, char *cmpData, size_t cmpCap) {
    using namespace Falcons;
#ifdef Falcons_PRINT_TIMINGS
    Timer timer(true);
#endif
    Config conf(config);
    if (cmpCap < Falcons_compress_size_bound<T>(conf)) {
        throw std::invalid_argument(Falcons_ERROR_COMP_BUFFER_NOT_LARGE_ENOUGH);
    }

    auto confEstSize = conf.size_est();
    auto cmpDataPos = reinterpret_cast<uchar *>(cmpData) + confEstSize;
    memset(cmpData, 0, confEstSize);
    auto cmpDataCap = cmpCap - conf.size_est();
#ifdef Falcons_PRINT_TIMINGS
    timer.stop("cmpData process");
#endif
    size_t cmpDataLen = 0;
    if (conf.N == 1) {
        cmpDataLen = Falcons_compress_impl<T, 1>(conf, data, cmpDataPos, cmpDataCap);
    } else if (conf.N == 2) {
        cmpDataLen = Falcons_compress_impl<T, 2>(conf, data, cmpDataPos, cmpDataCap);
    } else if (conf.N == 3) {
        cmpDataLen = Falcons_compress_impl<T, 3>(conf, data, cmpDataPos, cmpDataCap);
    } else if (conf.N == 4) {
        cmpDataLen = Falcons_compress_impl<T, 4>(conf, data, cmpDataPos, cmpDataCap);
    } else {
        throw std::invalid_argument("Data dimension higher than 4 is not supported.");
    }

#ifdef Falcons_PRINT_TIMINGS
    timer.start();
#endif
    auto cmpConfPos = reinterpret_cast<uchar *>(cmpData);

    auto confSize = conf.save(cmpConfPos);
    if (confSize > confEstSize) {
        throw std::length_error("buffer allocated for config is not large enough.");
    }
#ifdef Falcons_PRINT_TIMINGS
    timer.stop("conf save");
#endif
    return confSize + cmpDataLen;
}

/**
 * API for compression
 * @tparam T  source data type
 * @param config config compression configuration
 * @param data source data
 * @param cmpSize compressed data size (in bytes)
 * @return compressed data, remember to 'delete []' when the data is no longer needed.
 *
 * Similar with Falcons_compress(Falcons::Config &conf, const T *data, char *cmpData, size_t cmpCap)
 * The only difference is this one doesn't need the pre-allocated buffer (thus remember to do 'delete []' yourself)
 */
template <class T>
char *Falcons_compress(const Falcons::Config &config, const T *data, size_t &cmpSize) {
    using namespace Falcons;

    size_t bufferLen = Falcons_compress_size_bound<T>(config);
    auto buffer = new char[bufferLen];
    cmpSize = Falcons_compress(config, data, buffer, bufferLen);

    return buffer;
}

/** Read and validate stream metadata before allocating a decompression buffer. */
inline void Falcons_read_config(Falcons::Config &config, const char *cmpData, size_t cmpSize) {
    using namespace Falcons;
    auto cmpConfPos = reinterpret_cast<const uchar *>(cmpData);
    config.load(cmpConfPos, cmpSize);
    if (config.falconsMagicNumber != Falcons_MAGIC_NUMBER) {
        throw std::invalid_argument("magic number mismatch, the input data is not compressed by Falcons");
    }
    if (!versionReadable(config.falconsDataVer)) {
        std::stringstream ss;
        printf("program v%s , program-data %s , input data v%s\n", Falcons_VER, Falcons_DATA_VER,
               versionStr(config.falconsDataVer).data());
        ss << "Please use Falcons v" << versionStr(config.falconsDataVer) << " to decompress the data" << std::endl;
        std::cerr << ss.str();
        throw std::invalid_argument(ss.str());
    }


    if (cmpSize <= static_cast<size_t>(static_cast<unsigned char>(cmpData[0])))
        throw std::invalid_argument("Missing Falcons compressed payload");
}

/**
 * API for decompression
 * @tparam T decompressed data type
 * @param config configuration placeholder. It will be overwritten by the compression configuration
 * @param cmpData compressed data
 * @param cmpSize compressed data size in bytes
 * @param decData pre-allocated buffer for decompressed data

 example:
 auto decData = new float[100*200*300];
 Falcons::Config conf;
 Falcons_decompress(conf, cmpData, cmpSize, decData);

 */
template <class T>
void Falcons_decompress(Falcons::Config &config, const char *cmpData, size_t cmpSize, T *&decData) {
    using namespace Falcons;
    Falcons_read_config(config, cmpData, cmpSize);
    const size_t headerSize = static_cast<unsigned char>(cmpData[0]);

    auto cmpDataPos = reinterpret_cast<const uchar *>(cmpData) + headerSize;
    auto cmpDataSize = cmpSize - headerSize;

    if (decData == nullptr) {
        decData = new T[config.num];
    }
    if (config.N == 1) {
        Falcons_decompress_impl<T, 1>(config, cmpDataPos, cmpDataSize, decData);
    } else if (config.N == 2) {
        Falcons_decompress_impl<T, 2>(config, cmpDataPos, cmpDataSize, decData);
    } else if (config.N == 3) {
        Falcons_decompress_impl<T, 3>(config, cmpDataPos, cmpDataSize, decData);
    } else if (config.N == 4) {
        Falcons_decompress_impl<T, 4>(config, cmpDataPos, cmpDataSize, decData);
    } else {
        throw std::invalid_argument("Data dimension higher than 4 is not supported.");
    }
}

/** Decompress into a caller-owned buffer with an explicit element capacity. */
template <class T>
void Falcons_decompress(Falcons::Config &config, const char *cmpData, size_t cmpSize,
                        T *&decData, size_t decCapacity) {
    Falcons_read_config(config, cmpData, cmpSize);
    if (decData == nullptr || config.num > decCapacity)
        throw std::invalid_argument("The decompression buffer is too small");
    Falcons_decompress(config, cmpData, cmpSize, decData);
}

/**
 * API for decompression
 * Similar with Falcons_decompress(Falcons::Config &config, char *cmpData, size_t cmpSize, T *&decData)
 * The only difference is this one doesn't need pre-allocated buffer for decompressed data
 *
 * @tparam T decompressed data type
 * @param config configuration placeholder. It will be overwritten by the compression configuration
 * @param cmpData compressed data
 * @param cmpSize compressed data size in bytes
 * @return decompressed data, remember to 'delete []' when the data is no longer needed.

 example:
 Falcons::Config conf;
 float decompressedData = Falcons_decompress(conf, cmpData, cmpSize)
 */
template <class T>
T *Falcons_decompress(Falcons::Config &config, const char *cmpData, size_t cmpSize) {
    using namespace Falcons;
    T *decData = nullptr;
    Falcons_decompress<T>(config, cmpData, cmpSize, decData);
    return decData;
}

#endif
