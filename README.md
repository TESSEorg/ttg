# parsec-cxx
Prototype C++ API for MADNESS and PaRSEC. These instructions refer to the latest version of the API, Template Task Graph (TTG).

# compile
GNU C++ (6.x) and Clang (Apple LLVM 8.0.0) compilers work. To avoid polluting the source tree, make a build directory,
and from there do

## useful cmake command-line arguments:
- `CMAKE_CXX_COMPILER`, e.g. `-DCMAKE_CXX_COMPILER=clang++` to use clang

## madness examples
N.B. Must use CMake to configure MADNESS (i.e. autotools builds will not work). The existing examples only use the `world` component of MADNESS, so to save time you only need to build targets `install-world` and `install-config`.
- `cmake <path to the top of parsec-cxx> -DMADNESS_ROOT_DIR=<MADNESS install prefix>`
- `make ttgtest-mad t9-wrap-mad mxm-summa-mad spmm-mad`
- `./ttgtest-mad`
- `./t9-wrap-mad`
- `./mxm-summa-ttg`
- `./spmm-mad`

### spmm-mad notes
To use block sparse matrices (instead of element sparse) must manually add additional compile flags (do `make spmm-mad VERBOSE=1` to reveal the compilation command):
`-I/path/to/btas -I/path/to.boost -DBLOCK_SPARSE_GEMM=1`. Obtain (latest) BTAS from https://github.com/BTAS/BTAS .

The block-sparse version of the code is not tested.

## parsec examples

- `cmake <path to the top of parsec-cxx> -DMADNESS_ROOT_DIR=<install prefix of Madness compiled with cmake>`
- `make ttgtest-parsec`
- `./ttgtest-parsec`

