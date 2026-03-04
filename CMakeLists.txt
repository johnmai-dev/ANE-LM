cmake_minimum_required(VERSION 3.20)
project(ane-lm VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# macOS deployment target
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS deployment target")

# Use new Accelerate LAPACK API
add_compile_definitions(ACCELERATE_NEW_LAPACK)

# Fetch dependencies
include(FetchContent)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)

FetchContent_Declare(
    tokenizers_cpp
    GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(json tokenizers_cpp)

# Include paths
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})

# Source files
set(VENDOR_SOURCES
    vendor/jinja/value.cpp
    vendor/jinja/string.cpp
    vendor/jinja/lexer.cpp
    vendor/jinja/parser.cpp
    vendor/jinja/runtime.cpp
    vendor/jinja/caps.cpp
)

set(CORE_SOURCES
    common.cpp
    core/safetensors.cpp
    core/model_loader.cpp
    core/tokenizer.cpp
    core/cpu_ops.cpp
    core/sampling.cpp
    core/ane_runtime.cpp
    utils.cpp
    generate.cpp
)

set(MODEL_SOURCES
    models/llm/qwen3_5.cpp
    models/llm/qwen3.cpp
)

set(MAIN_SOURCE
    main.cpp
)

# Executable
add_executable(ane-lm
    ${VENDOR_SOURCES}
    ${CORE_SOURCES}
    ${MODEL_SOURCES}
    ${MAIN_SOURCE}
)

# Link frameworks and ObjC runtime
target_link_libraries(ane-lm PRIVATE
    nlohmann_json::nlohmann_json
    tokenizers_cpp
    "-framework Foundation"
    "-framework IOSurface"
    "-framework Accelerate"
    "-lobjc"
)

# Optimization for release
target_compile_options(ane-lm PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:RelWithDebInfo>:-O2 -g>
)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
