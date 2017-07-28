# parsec-cxx
Prototype C++ API for MADNESS and PaRSEC. These instructions refer to the latest version of the API, Template Task Graph (TTG).

# compile
GNU C++ (6.x) and Clang (Apple LLVM 8.0.0) compilers work. To avoid polluting the source tree, make a build directory,
and from there do

## madness examples
- `cmake <path to the top of parsec-cxx> -DMADNESS_ROOT_DIR=<install prefix of Madness compiled with cmake>`
- `make ttgtest-mad t9-wrap-mad mxm-summa-mad spmm-mad`
- `./ttgtest-mad`
- `./t9-wrap-mad`
- `./mxm-summa-ttg`
- `./spmm-mad`

### spmm-mad notes
To use block sparse matrices (instead of element sparse) must manually add additional compile flags (do `make spmm-mad VERBOSE=1` to reveal the compilation command):
`-I/path/to/btas -I/path/to.boost -DBLOCK_SPARSE_GEMM=1`. Obtain (latest) BTAS from https://github.com/BTAS/BTAS .

Block-sparse version of the code is not tested.

## parsec examples

- `cmake <path to the top of parsec-cxx> -DMADNESS_ROOT_DIR=<install prefix of Madness compiled with cmake>`
- `make ttgtest-parsec`
- `./ttgtest-parsec`

optional cmake command-line arguments:
- `-DCMAKE_CXX_COMPILER=clang++`
