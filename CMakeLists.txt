cmake_minimum_required(VERSION 3.18)
project(Falcons VERSION 1.0.0)

#data version defines the version of the compressed data format
#it is not always equal to the program version (e.g., Falcons v3.1.0 and Falcons v.3.1.1 may use the same data version of v.3.1.0)
#only update data version if the new version of the program changes compressed data format
set(Falcons_DATA_VERSION 1.0.1)

include(GNUInstallDirs)


option(BUILD_SHARED_LIBS "build shared libraries by default" ON)
option(ENABLE_AVX2 "Enable AVX2 instruction set compilation" OFF )
option(ENABLE_SVE2 "Enable SVE2 instruction set compilation" OFF )
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
else () #-Wunused-parameter
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wreorder -Wsign-compare -Wsuggest-override -Wold-style-cast -Wcast-qual -Wzero-as-null-pointer-constant -Wextra-semi -Wnon-virtual-dtor -Wcast-align -Wunused-variable")
endif ()
set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type was set. Setting build type to ${default_build_type}.")
    set(CMAKE_BUILD_TYPE ${default_build_type} CACHE
            STRING "Choose the type to build" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
            "MinSizeRel" "RelWithDebInfo")
endif ()

find_package(PkgConfig)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Falcons/version.hpp.in
        ${CMAKE_CURRENT_BINARY_DIR}/include/Falcons/version.hpp)

# Standalone FSE is a compiled dependency of the public template API.
# Keep its private symbols hidden so they cannot interpose libzstd internals.
add_library(Falcons_fse STATIC
        include/Falcons/encoder/zfse/entropy_common.c
        include/Falcons/encoder/zfse/error_private.c
        include/Falcons/encoder/zfse/fse_compress.c
        include/Falcons/encoder/zfse/fse_decompress.c
        include/Falcons/encoder/zfse/hist.c)
set_target_properties(Falcons_fse PROPERTIES
        EXPORT_NAME FSE
        POSITION_INDEPENDENT_CODE ON
        C_VISIBILITY_PRESET hidden)

add_library(${PROJECT_NAME} INTERFACE)
add_library(Falcons::Falcons ALIAS Falcons)
target_link_libraries(${PROJECT_NAME} INTERFACE Falcons_fse)
target_include_directories(
        ${PROJECT_NAME} INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(${PROJECT_NAME}
        INTERFACE cxx_std_17
)

find_package(OpenMP)
if (OpenMP_FOUND)
    target_link_libraries(${PROJECT_NAME} INTERFACE OpenMP::OpenMP_CXX)
endif ()

if (MSVC)
    option(Falcons_USE_BUNDLED_ZSTD "prefer the bundled version of Zstd" ON)
else ()
    option(Falcons_USE_BUNDLED_ZSTD "prefer the bundled version of Zstd" OFF)
endif ()

option(Falcons_DEBUG_TIMINGS "print debug timing information" OFF)
if (Falcons_DEBUG_TIMINGS)
    target_compile_definitions(${PROJECT_NAME} INTERFACE Falcons_DEBUG_TIMINGS=1)
else ()
    target_compile_definitions(${PROJECT_NAME} INTERFACE Falcons_DEBUG_TIMINGS=0)
endif ()

option(Falcons_USE_SKA_HASH "use ska hashmap instead of std hashmap, default is ska for better speed" ON)
if (Falcons_USE_SKA_HASH)
    target_compile_definitions(${PROJECT_NAME} INTERFACE Falcons_USE_SKA_HASH=1)
else ()
    target_compile_definitions(${PROJECT_NAME} INTERFACE Falcons_USE_SKA_HASH=0)
endif ()

if (PkgConfig_FOUND)
    pkg_search_module(ZSTD IMPORTED_TARGET libzstd)
endif ()
set(Falcons_USE_SYSTEM_ZSTD OFF)
if (ZSTD_FOUND AND NOT Falcons_USE_BUNDLED_ZSTD)
    set(Falcons_USE_SYSTEM_ZSTD ON)
    target_link_libraries(${PROJECT_NAME} INTERFACE PkgConfig::ZSTD)
else ()
    add_subdirectory(tools/zstd)
    target_link_libraries(${PROJECT_NAME} INTERFACE zstd)
endif ()

option(BUILD_TESTING "Build tests" OFF)
if (BUILD_TESTING)
    enable_testing()
    add_subdirectory(tools/test)
endif ()

find_package(GSL)
if (GSL_FOUND)
    target_compile_definitions(${PROJECT_NAME} INTERFACE -DFalcons_ENABLE_GSL)
    target_link_libraries(${PROJECT_NAME} INTERFACE GSL::gsl)
endif ()

option(BUILD_Falcons_BINARY "build FALCONS executable" ON)
if (BUILD_Falcons_BINARY)
    add_subdirectory(tools/falcons)
endif ()

option(BUILD_H5Z_FILTER "build the H5Z-Falcons filter" OFF)
if (BUILD_H5Z_FILTER)
    add_subdirectory(tools/H5Z-Falcons)
endif ()

if(ENABLE_AVX2)
  message(STATUS "AVX2 compilation enabled.")
  target_compile_options(${PROJECT_NAME} INTERFACE -mavx2 -mfma ) # -mavx512f
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(WARNING "Compiler '${CMAKE_CXX_COMPILER_ID}' might not explicitly support AVX2 flags. Please add them manually if needed.")
  endif()
else()
  message(STATUS "AVX2 compilation disabled.")
endif()

if(ENABLE_SVE2)
  message(STATUS "SVE2 compilation enabled.")
  target_compile_options(${PROJECT_NAME} INTERFACE -march=armv8.6-a+sve2)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(WARNING "Compiler '${CMAKE_CXX_COMPILER_ID}' might not explicitly support SVE2 flags. Please add them manually if needed.")
  endif()
else()
  message(STATUS "SVE2 compilation disabled.")
endif()


install(TARGETS ${PROJECT_NAME} Falcons_fse
        EXPORT FalconsTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/include/Falcons/version.hpp"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Falcons/"
)
install(EXPORT FalconsTargets NAMESPACE Falcons:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Falcons)
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/FalconsConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/FalconsConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Falcons
)
write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/FalconsConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
)
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/FalconsConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/FalconsConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Falcons
)
#export falcons target for external use
export(EXPORT FalconsTargets NAMESPACE Falcons:: FILE Falcons.cmake)


# Retain project and third-party notices in installed distributions.
set(Falcons_LICENSE_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/licenses/Falcons")
install(FILES LICENSE copyright-and-BSD-license.txt
        DESTINATION "${Falcons_LICENSE_INSTALL_DIR}")
install(FILES include/Falcons/encoder/zfse/LICENSE include/Falcons/encoder/zfse/COPYING
        DESTINATION "${Falcons_LICENSE_INSTALL_DIR}/zfse")
install(FILES include/Falcons/utils/ska_hash/LICENSE.txt
        DESTINATION "${Falcons_LICENSE_INSTALL_DIR}/ska_hash")
install(FILES tools/zstd/LICENSE tools/zstd/COPYING
        DESTINATION "${Falcons_LICENSE_INSTALL_DIR}/zstd")
