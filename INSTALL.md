# prerequisites
- C++ compiler with support for the [C++17 standard](http://www.iso.org/standard/68564.html), or a more recent standard. This includes the following compilers:
  - [GNU C++](https://gcc.gnu.org/), version 7.0 or higher
  - [Clang](https://clang.llvm.org/), version 5 or higher
  - [Apple Clang](https://en.wikipedia.org/wiki/Xcode), version 9.3 or higher
  - [Intel C++ compiler](https://software.intel.com/en-us/c-compilers), version 19 or higher

  See the current [Travis CI matrix](.travis.yml) for the most up-to-date list of compilers that are known to work.

- [CMake](https://cmake.org/), version 3.10 or higher; if CUDA support is needed, CMake 3.17 or higher is required.
- [Git]() 1.8 or later (required to obtain TiledArray and MADNESS source code from GitHub)
- [Boost](https://boost.org/) version 1.66 or later; installed, if not found
- PaRSEC or MADNESS runtimes (see below); installed, if not found

# build
- `cmake <cmake args>`
- `cmake --build . --target <test or example you want (see below)>`

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

