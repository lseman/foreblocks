# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/data/solver/ad/tree/build-cuda/_deps/cpm-src")
  file(MAKE_DIRECTORY "/data/solver/ad/tree/build-cuda/_deps/cpm-src")
endif()
file(MAKE_DIRECTORY
  "/data/solver/ad/tree/build-cuda/_deps/cpm-build"
  "/data/solver/ad/tree/build-cuda/_deps/cpm-subbuild/cpm-populate-prefix"
  "/data/solver/ad/tree/build-cuda/_deps/cpm-subbuild/cpm-populate-prefix/tmp"
  "/data/solver/ad/tree/build-cuda/_deps/cpm-subbuild/cpm-populate-prefix/src/cpm-populate-stamp"
  "/data/solver/ad/tree/build-cuda/_deps/cpm-subbuild/cpm-populate-prefix/src"
  "/data/solver/ad/tree/build-cuda/_deps/cpm-subbuild/cpm-populate-prefix/src/cpm-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/data/solver/ad/tree/build-cuda/_deps/cpm-subbuild/cpm-populate-prefix/src/cpm-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/data/solver/ad/tree/build-cuda/_deps/cpm-subbuild/cpm-populate-prefix/src/cpm-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
