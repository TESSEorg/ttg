# prerequisites
- CMake, version 3.9 or more recent
- C++17 compiler
- Boost 1.66 or later; installed, if not found
- PaRSEC or MADNESS runtimes (see below); installed, if not found

# compile
GNU C++ (7.x) and Clang (Apple Clang 9.3 or LLVM Clang 7) compilers work. To avoid polluting the source tree, make a build directory,
and from there do:
- cmake `<cmake args>`

## useful cmake command-line arguments:
- `CMAKE_CXX_COMPILER`, e.g. `-DCMAKE_CXX_COMPILER=clang++` to use clang
- `BOOST_ROOT`, e.g. `-DBOOST_ROOT=path` (to ensure that `path` is correct look for `path/boost/version.hpp` exists)

## MADNESS runtime notes
The following tests/examples are supported with MADNESS
- `test-mad`
- `t9-mad`
- `serialization`

## PaRSEC runtime notes

currently PaRSEC support is being retrofitted to be able to use the latest PaRSEC.

## spmm-* notes
To compile (block-)sparse SUMMA must:
- obtain and install Eigen (header-only) library and pass the path to Eigen source to CMake as `-DEIGEN3_INCLUDE_DIR=<path to Eigen>`
- __block-sparse only__ obtain and install BTAS header-only tensor library from https://github.com/BTAS/BTAS ;
  clone and pass the path to BTAS source to CMake as `-DBTAS_INSTALL_DIR=<path to BTAS>`
- to build element-sparse SUMMA example: `make spmm-mad`
- to build block-sparse SUMMA example: `make bspmm-mad`

