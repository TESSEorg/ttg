# synopsis

```.sh
$ git clone https://github.com/TESSEorg/ttg.git
$ cmake -S ttg -B ttg/build -DCMAKE_INSTALL_PREFIX=/path/to/ttg/install [optional cmake args]
$ cmake --build ttg/build
(optional, but recommended): $ cmake --build ttg/build --target check-ttg
$ cmake --build ttg/build --target install
```

# prerequisites

## mandatory prerequisites
- [CMake](https://cmake.org/), version 3.14 or higher
- C++ compiler with support for the [C++17 standard](http://www.iso.org/standard/68564.html), or a more recent standard. This includes the following compilers:
  - [GNU C++](https://gcc.gnu.org/), version 7.0 or higher
  - [Clang](https://clang.llvm.org/), version 5 or higher
  - [Apple Clang](https://en.wikipedia.org/wiki/Xcode), version 9.3 or higher
  - [Intel C++ compiler](https://software.intel.com/en-us/c-compilers), version 19 or higher
- one or more of the following runtimes:
  - [PaRSEC](https://bitbucket.org/icldistcomp/parsec): this distributed-memory runtime is the primary runtime intended for high-performance implementation of TTG
  - [MADNESS](https://github.org/m-a-d-n-e-s-s/madness): this distributed-memory runtime is to be used primarily for developmental purposes

While the list of prerequisites is short, note that the runtimes have many more mandatory prerequisites; these are discussed under `transitive prerequisites` below.
Also: it is _strongly_ recommended that the runtimes are built as parts of the TTG build process (this requires some of the optional prerequisites, listed below). This will make sure that the correct versions of the runtimes are used.

## optional prerequisites
- [Git](https://git-scm.com) 1.8 or later: needed to obtain the source code for PaRSEC or MADNESS runtimes
- [Boost](https://boost.org/) version 1.66 or later: needed to use TTG with classes serializable by the [Boost.Serialization](https://www.boost.org/doc/libs/master/libs/serialization/doc/index.html) library.
  - The [Boost.Serialization](https://www.boost.org/doc/libs/master/libs/serialization/doc/index.html) library is not header-only, i.e., it must be compiled.
  - If the Boost package is not detected TTG can download and build Boost as part of its build process; to do that configure TTG with the CMake cache variable `TTG_FETCH_BOOST` set to `ON` (e.g., by adding `-DTTG_FETCH_BOOST=ON` to the CMake executable command line)

## transitive prerequisites

### PaRSEC
see [here](https://bitbucket.org/icldistcomp/parsec/src/master/INSTALL.rst#rst-header-id1)

### MADNESS
- An implementation of Message Passing Interface version 2 or 3, with support for `MPI_THREAD_MULTIPLE`.
- a Pthreads library
- (optional) Intel Thread Building Blocks (TBB), available in a [commercial](software.intel.com/tbb) or an [open-source](https://www.threadingbuildingblocks.org/) form

## prerequisites for building examples

TTG includes several examples that may require additional prerequisites. These are listed here:
- SPMM: (block-)sparse matrix multiplication example
  - [Eigen](https://eigen.tuxfamily.org/) library, version 3
  - [BTAS](https://github.com/ValeevGroup/BTAS) library: for the _block_-sparse case only
    - BTAS' prerequisites are listed [here](https://github.com/ValeevGroup/BTAS#prerequisites)

# configure + build
- `cmake -S /path/to/ttg/source/directory -B /path/to/ttg/build/directory <cmake args>`
- `cmake --build /path/to/ttg/build/directory [--target <check-ttg | install>]`

## useful cmake cache variables:

|Variable                         |Default             | Description   |
|---------------------------------|--------------------|---------------|
| `BUILD_TESTING`                 | `ON`               | whether target `check-ttg` and its relatives will actually build and run unit tests |
| `TTG_EXAMPLES`                  | `OFF`              | whether target `check-ttg` and its relatives will actually build and run examples; setting this to `ON` will cause detection of several optional prerequisites, and (if missing) building from source |
| `TTG_ENABLE_TRACE`              | `OFF`              | setting this to `ON` will enable the ability to instrument TTG code for tracing (see `ttg::trace()`, etc.); if this is set to `OFF`, `ttg::trace()` is a no-op |
