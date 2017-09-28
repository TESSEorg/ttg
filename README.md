# tesse-cxx
Prototype TESSE C++ API, with MADNESS and PaRSEC as backends. These instructions refer to the latest version of the API, Template Task Graph (TTG).

# prerequisites
- C++14 compiler
- Boost 1.66 (__to be released in 12/2017!__, for now clone boost.org sources as explained [here](https://github.com/boostorg/boost/wiki/Getting-Started)

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
- `make ttgtest-mad t9-wrap-mad mxm-summa-mad spmm-mad`
- `./ttgtest-mad`
- `./t9-wrap-mad`
- `./mxm-summa-ttg`
- `./spmm-mad`

## PaRSEC examples

N.B. Distributed memory is not yet supported with PaRSEC backend.

- `export PKG_CONFIG_PATH=<PaRSEC lib prefix>/pkgconfig:${PKG_CONFIG_PATH}`
- `cmake <path to the top of tesse-cxx>
- `make ttgtest-parsec t9-wrap-parsec spmm-parsec`
- `./ttgtest-parsec`
- `./t9-wrap-parsec`
- `./spmm-parsec`

## spmm-* notes
To use block sparse matrices (instead of element sparse) must manually add additional compile flags (do `make spmm-mad VERBOSE=1` to reveal the compilation command):
`-I/path/to/btas -I/path/to.boost -DBLOCK_SPARSE_GEMM=1`. Obtain (latest) BTAS from https://github.com/BTAS/BTAS .

The block-sparse version of the code is not tested.

