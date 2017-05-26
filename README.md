# parsec-cxx
prototype C++ API for PaRSEC

# compile
GNU C++ (6.x) and Clang (Apple LLVM 8.0.0) compilers work. To avoid polluting the source tree, make a build directory,
and from there do

- `cmake <path to the top of parsec-cxx> -DMADNESS_ROOT_DIR=<install prefix of Madness compiled with cmake>`
- `make ttgtest t9-wrap-ttg`
- `./ttgtest`
- `./t9-wrap-ttg`

optional cmake command-line arguments:
- `-DCMAKE_CXX_COMPILER=clang++`
