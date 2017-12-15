# tesse-cxx
Prototype TESSE C++ API, with MADNESS and PaRSEC as backends. These instructions refer to the latest version of the API, Template Task Graph (TTG).

# prerequisites
- C++14 compiler
- Boost 1.66 (__to be released in 12/2017!__ ) ... until the release occurs get boost as follows (more instructions   [here](https://github.com/boostorg/boost/wiki/Getting-Started)
):
  - `git clone --recursive https://github.com/boostorg/boost.git`
  - `cd boost`
  - `./bootstrap.sh`
  - `./b2 headers`
  - make sure that you pass the output of `pwd` to CMake as the argument to `BOOST_ROOT`

# compile
GNU C++ (6.x) and Clang (Apple LLVM 8.0.0) compilers work. To avoid polluting the source tree, make a build directory,
and from there do:
- cmake `<cmake args>`

## useful cmake command-line arguments:
- `CMAKE_CXX_COMPILER`, e.g. `-DCMAKE_CXX_COMPILER=clang++` to use clang
- `BOOST_ROOT`, e.g. `-DBOOST_ROOT=path` (to ensure that `path` is correct look for `path/boost/version.hpp` exists)

## notes on PaRSEC
- run CMake with `-DBUILD_DPLASMA=OFF -DSUPPORT_FORTRAN=OFF -DPARSEC_WITH_DEVEL_HEADERS=ON`, then `make install`

## MADNESS examples
N.B. Must use CMake to configure MADNESS (i.e. autotools builds will not work). The existing examples only use the `world` component of MADNESS, so to save time you only need to build targets `install-world` and `install-config`.
- `cmake <path to the top of tesse-cxx> -DMADNESS_ROOT_DIR=<MADNESS install prefix>`
- `make test-mad t9-mad serialization`
- `./examples/ttg-mad`
- `./examples/t9-mad`
- `./tests/serialization`

## PaRSEC examples

N.B. Distributed memory is not yet supported with PaRSEC backend.

- `export PKG_CONFIG_PATH=<PaRSEC lib prefix>/pkgconfig:${PKG_CONFIG_PATH}`
- `cmake <path to the top of tesse-cxx>
- `make test-parsec t9-parsec`
- `./examples/test-parsec`
- `./examples/t9-parsec`

## spmm-* notes
To compile (block-)sparse SUMMA must:
- obtain and install Eigen (header-only) library and pass the path to Eigen source to CMake as `-DEIGEN3_INCLUDE_DIR=<path to Eigen>`
- __block-sparse only__ obtain and install BTAS header-only tensor library from https://github.com/BTAS/BTAS ;
  clone and pass the path to BTAS source to CMake as `-DBTAS_INSTALL_DIR=<path to BTAS>`
- to build element-sparse SUMMA example: `make spmm-mad`
- to build block-sparse SUMMA example: `make bspmm-mad`

