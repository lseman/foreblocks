cmake_minimum_required(VERSION 3.20)

project(foretree_isolated LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(TREE_BUILD_FORETREE "Build foretree pybind module" ON)
option(TREE_BUILD_FOREFOREST "Build foreforest pybind module" ON)
option(TREE_ENABLE_TBB "Enable TBB if available" ON)
option(TREE_ENABLE_STDEXEC "Enable stdexec include path for foreforest" ON)
option(FORETREE_EXEC_USE_CUDA "Use nvexec CUDA scheduler for foreforest" OFF)
option(TREE_ENABLE_CUDA_BACKEND "Build native CUDA histogram backend" ON)
set(TREE_STDEXEC_INCLUDE_DIR "" CACHE PATH "Path to stdexec include directory")

# Paths to local tree module sources
set(TREE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(TREE_INCLUDE_DIR "${TREE_ROOT_DIR}/include")
set(TREE_SRC_DIR "${TREE_ROOT_DIR}/src")

# ------------------------------------------------------------------
# CPM bootstrap (before any CPMAddPackage)
# ------------------------------------------------------------------
include(cmake/CPM.cmake OPTIONAL)
if(NOT COMMAND CPMAddPackage)
  include(FetchContent)
  FetchContent_Declare(
    CPM
    GIT_REPOSITORY https://github.com/cpm-cmake/CPM.cmake.git
    GIT_TAG v0.42.3
  )
  FetchContent_MakeAvailable(CPM)
  include(${cpm_SOURCE_DIR}/cmake/CPM.cmake)
endif()

# ------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------

if(TREE_ENABLE_STDEXEC)
  message(STATUS "stdexec enabled")
  cpmaddpackage(
    NAME stdexec
    GITHUB_REPOSITORY NVIDIA/stdexec
    GIT_TAG main
    OPTIONS
    "STDEXEC_BUILD_TESTS OFF"
    "STDEXEC_BUILD_EXAMPLES OFF"
    "STDEXEC_BUILD_BENCHMARKS OFF"
    "STDEXEC_BUILD_DOCS OFF"
    "BUILD_TESTING OFF"
    "STDEXEC_ENABLE_CUDA ${FORETREE_EXEC_USE_CUDA}"
  )
else()
  message(STATUS "stdexec disabled")
endif()

if(FORETREE_EXEC_USE_CUDA AND NOT TREE_ENABLE_STDEXEC)
  message(FATAL_ERROR "FORETREE_EXEC_USE_CUDA=ON requires TREE_ENABLE_STDEXEC=ON")
endif()

# Dependencies kept minimal for tree-only builds
find_package(fmt CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

if(TREE_ENABLE_CUDA_BACKEND)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(foretree_cuda STATIC "${TREE_SRC_DIR}/gpu/cuda_histogram.cu")
  target_include_directories(foretree_cuda PUBLIC "${TREE_INCLUDE_DIR}")
  target_link_libraries(foretree_cuda PUBLIC CUDA::cudart)
  target_compile_features(foretree_cuda PUBLIC cxx_std_20)
  set_target_properties(foretree_cuda PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
  )
  target_compile_definitions(foretree_cuda PUBLIC FORETREE_HAS_CUDA=1)

  # enable_testing()
  # add_executable(test_cuda_histogram "${TREE_ROOT_DIR}/tests/test_cuda_histogram.cu")
  # target_include_directories(test_cuda_histogram PRIVATE "${TREE_INCLUDE_DIR}")
  # target_link_libraries(test_cuda_histogram PRIVATE foretree_cuda)
  # set_target_properties(test_cuda_histogram PROPERTIES
  # CUDA_STANDARD 20
  # CUDA_STANDARD_REQUIRED ON
  # )
  # add_test(NAME cuda_histogram COMMAND test_cuda_histogram)
endif()

add_library(foretree_core INTERFACE)
target_include_directories(foretree_core
  INTERFACE
  "${TREE_INCLUDE_DIR}"
)
target_link_libraries(foretree_core
  INTERFACE
  fmt::fmt
  Eigen3::Eigen
)

function(tree_common_setup tgt)
  target_compile_features(${tgt} PRIVATE cxx_std_23)
  target_include_directories(${tgt}
    PRIVATE
    "${TREE_INCLUDE_DIR}"
    "${TREE_SRC_DIR}"
  )
  target_link_libraries(${tgt}
    PRIVATE
    fmt::fmt
    Eigen3::Eigen
    Python3::Module
  )

  if(MSVC)
    target_compile_options(${tgt} PRIVATE /W4)
  else()
    target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wpedantic)
  endif()

  if(TREE_ENABLE_TBB)
    find_package(TBB QUIET COMPONENTS tbb)
    if(TARGET TBB::tbb)
      target_link_libraries(${tgt} PRIVATE TBB::tbb)
    else()
      target_compile_definitions(${tgt} PRIVATE FF_NO_TBB)
    endif()
  else()
    target_compile_definitions(${tgt} PRIVATE FF_NO_TBB)
  endif()
endfunction()

if(TREE_BUILD_FORETREE)
  # Prefer Python 3.12
  if(NOT DEFINED Python3_EXECUTABLE)
    find_program(Python3_EXECUTABLE NAMES python3.12 python3.11 python3)
  endif()
  find_package(Python3 3.12 EXACT COMPONENTS Interpreter Development.Module REQUIRED)
  CPMAddPackage(
    NAME nanobind
    GITHUB_REPOSITORY wjakob/nanobind
    GIT_TAG v2.9.2
    OPTIONS "NANOBIND_INSTALL OFF" "NB_USE_BUILD_SUPPORT ON"
  )
  # Link Python development headers to nanobind build
  if(TARGET nanobind-static)
    target_link_libraries(nanobind-static PRIVATE Python3::Module)
  endif()
  nanobind_add_module(foretree NB_DOMAIN foretree "${TREE_SRC_DIR}/pybind/foretree.cpp")
  tree_common_setup(foretree)
  target_link_libraries(foretree PRIVATE foretree_core)
  if(TREE_ENABLE_CUDA_BACKEND)
    target_link_libraries(foretree PRIVATE foretree_cuda)
  endif()
  if(TREE_ENABLE_STDEXEC)
    target_include_directories(foretree PRIVATE ${stdexec_SOURCE_DIR}/include)
  endif()
endif()

if(TREE_BUILD_FOREFOREST)
  # Prefer Python 3.12
  if(NOT DEFINED Python3_EXECUTABLE)
    find_program(Python3_EXECUTABLE NAMES python3.12 python3.11 python3)
  endif()
  find_package(Python3 3.12 EXACT COMPONENTS Interpreter Development.Module REQUIRED)

  cpmaddpackage(
    NAME nanobind
    GITHUB_REPOSITORY wjakob/nanobind
    GIT_TAG v2.9.2
    OPTIONS "NANOBIND_INSTALL OFF" "NB_USE_BUILD_SUPPORT ON"
  )
  # Link Python development headers to nanobind build
  if(TARGET nanobind-static)
    target_link_libraries(nanobind-static PRIVATE Python3::Module)
  endif()
  nanobind_add_module(foreforest NB_DOMAIN foreforest "${TREE_SRC_DIR}/pybind/foreforest.cpp")
  tree_common_setup(foreforest)
  target_link_libraries(foreforest PRIVATE foretree_core)
  if(TREE_ENABLE_CUDA_BACKEND)
    target_link_libraries(foreforest PRIVATE foretree_cuda)
  endif()

  if(TREE_ENABLE_STDEXEC)
    target_include_directories(foreforest PRIVATE ${stdexec_SOURCE_DIR}/include)
    target_link_libraries(foreforest PRIVATE stdexec)
    target_compile_definitions(foreforest PRIVATE TREE_ENABLE_STDEXEC=1)
  endif()

  if(FORETREE_EXEC_USE_CUDA)
    target_compile_definitions(foreforest PRIVATE FORETREE_EXEC_USE_CUDA=1)
    if(TARGET STDEXEC::nvexec)
      target_link_libraries(foreforest PRIVATE STDEXEC::nvexec)
    else()
      message(FATAL_ERROR "FORETREE_EXEC_USE_CUDA=ON requires STDEXEC::nvexec target. Check CUDA/stdexec setup.")
    endif()
  endif()
endif()

message(STATUS "==== Tree isolated build ====")
message(STATUS "TREE_BUILD_FORETREE   : ${TREE_BUILD_FORETREE}")
message(STATUS "TREE_BUILD_FOREFOREST : ${TREE_BUILD_FOREFOREST}")
message(STATUS "TREE_ENABLE_TBB       : ${TREE_ENABLE_TBB}")
message(STATUS "TREE_ENABLE_STDEXEC   : ${TREE_ENABLE_STDEXEC}")
message(STATUS "FORETREE_EXEC_USE_CUDA: ${FORETREE_EXEC_USE_CUDA}")
message(STATUS "TREE_ENABLE_CUDA_BACKEND: ${TREE_ENABLE_CUDA_BACKEND}")
