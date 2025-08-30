cmake_minimum_required(VERSION 3.18)
project(unified_tree_cpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra -pedantic")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /arch:AVX2")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /W4")
endif()

find_package(Python3 3.12 EXACT COMPONENTS Interpreter Development REQUIRED)

# Find required packages
find_package(pybind11 REQUIRED)

# Create the pybind11 module FIRST
pybind11_add_module(unified_tree unified_tree.cpp)

# Set properties
target_compile_features(unified_tree PRIVATE cxx_std_20)
target_compile_definitions(unified_tree PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# Optional: Find OpenMP for parallel processing
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(unified_tree PRIVATE OpenMP::OpenMP_CXX)
    add_definitions(-DWITH_OPENMP)
endif()

# Optional: Find Intel TBB for advanced parallelism
find_package(TBB)
if(TBB_FOUND)
    target_link_libraries(unified_tree PRIVATE TBB::tbb)
    add_definitions(-DWITH_TBB)
endif()

# Platform-specific settings
if(APPLE)
    set_target_properties(unified_tree PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Enable LTO for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_property(TARGET unified_tree PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()