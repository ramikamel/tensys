cmake_minimum_required(VERSION 3.24)
project(tensys VERSION 1.0.0 LANGUAGES CXX)

include(CTest)
include(FetchContent)

# Use verified local archives when available, for repeat builds without network.
function(tensys_dependency name version url sha256)
    set(archive "${PROJECT_SOURCE_DIR}/build/dependencies/${name}-${version}.tar.gz")
    if(EXISTS "${archive}")
        set(url "${archive}")
    endif()
    FetchContent_Declare(${name} URL "${url}" URL_HASH "SHA256=${sha256}")
    FetchContent_MakeAvailable(${name})
endfunction()

tensys_dependency(json 3.12.0
    https://codeload.github.com/nlohmann/json/tar.gz/refs/tags/v3.12.0
    4b92eb0c06d10683f7447ce9406cb97cd4b453be18d7279320f7b2f025c10187)

add_library(tensys_core
    src/config/config.cpp
    src/tiler/schedule.cpp
    src/sim/simulator.cpp
    src/sim/tile_cost.cpp
    src/sim/writeback.cpp
    src/verify/verify.cpp
)
target_compile_features(tensys_core PUBLIC cxx_std_20)
set_target_properties(tensys_core PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(tensys_core PUBLIC src)
target_link_libraries(tensys_core PUBLIC nlohmann_json::nlohmann_json)

add_executable(tensys src/main.cpp)
target_link_libraries(tensys PRIVATE tensys_core)
target_compile_definitions(tensys PRIVATE TENSYS_VERSION="${PROJECT_VERSION}")
set_target_properties(tensys PROPERTIES CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    foreach(target tensys_core tensys)
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endforeach()
else()
    message(FATAL_ERROR "TenSys requires Clang or GCC for exact wide-integer cycle arithmetic")
endif()

if(BUILD_TESTING)
    set(BUILD_GMOCK OFF CACHE BOOL "Build GoogleMock")
    tensys_dependency(googletest 1.17.0
        https://codeload.github.com/google/googletest/tar.gz/refs/tags/v1.17.0
        65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c)
    add_executable(tensys_tests tests/model_test.cpp)
    target_link_libraries(tensys_tests PRIVATE tensys_core GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(tensys_tests)
endif()
