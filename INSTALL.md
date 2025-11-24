# synopsis

```sh
$ git clone https://github.com/TESSEorg/ttg.git
$ cmake -S ttg -B ttg/build -DCMAKE_INSTALL_PREFIX=/path/to/ttg/install [optional cmake args]
$ cmake --build ttg/build
(optional, but recommended): $ cmake --build ttg/build --target check-ttg
$ cmake --build ttg/build --target install
```

# prerequisites

TTG is usable only on POSIX systems.

## mandatory prerequisites
- [CMake](https://cmake.org/), version 3.14 or higher; version 3.21 or higher is required to support execution on HIP/ROCm-capable devices.
- C++ compiler with support for the [C++20 standard](http://www.iso.org/standard/68564.html), or a more recent standard. This includes the following compilers:
  - [GNU C++](https://gcc.gnu.org/), version 10.0 or higher; GCC is the only compiler that can be used for accelerator programming.
  - [Clang](https://clang.llvm.org/), version 10 or higher
  - [Apple Clang](https://en.wikipedia.org/wiki/Xcode), version 10.0 or higher
  - [Intel C++ compiler](https://software.intel.com/en-us/c-compilers), version 2021.1 or higher
- one or more of the following runtimes:
  - [PaRSEC](https://bitbucket.org/icldistcomp/parsec): this distributed-memory runtime is the primary runtime intended for high-performance implementation of TTG, including on accelerators
  - [MADNESS](https://github.org/m-a-d-n-e-s-s/madness): this distributed-memory runtime is to be used primarily for developmental purposes (host-only)
- [Umpire C++ allocator](github.com/ValeevGroup/umpire-cxx-allocator) -- a C++ allocator for [LLNL/Umpire](https://github.com/LLNL/Umpire), a portable memory manager. Umpire itself is a prerequisite of this, and can be built from source.

While the list of prerequisites is short, note that the runtimes have many more mandatory prerequisites; these are discussed under `transitive prerequisites` below.
Also: it is _strongly_ recommended that the runtimes and the memory manager are built as parts of the TTG build process (this requires some of the optional prerequisites, listed below). This will make sure that the correct versions of these prerequisites and their dependents are used.

## optional prerequisites
- [Git](https://git-scm.com): needed to obtain the source code for any prerequisite built from source code as part of TTG, such as PaRSEC or MADNESS runtimes
- [Boost](https://boost.org/) version 1.81 or later. If the Boost package is not detected TTG can download and build Boost as part of its build process, but this is NOT recommended, you should obtain Boost via the system or third-party package manager. Experts may try to build Boost from source as part of TTG by configuring it with the CMake cache variable `TTG_FETCH_BOOST` set to `ON` (e.g., by adding `-DTTG_FETCH_BOOST=ON` to the CMake executable command line). The following primary Boost libraries/modules (and their transitive dependents) are used:
  - (required) [Boost.CallableTraits](): used to introspect generic callables given to `make_tt`. P.S. TTG has a bundled copy of `Boost.CallableTraits` which is used and installed if Boost is not found or built from source. To avoid the installation and use of the bundled Boost.CallableTraits configure TTG with the CMake cache variable `TTG_IGNORE_BUNDLED_EXTERNALS` set to `ON`.
  - (optional) [Boost.Serialization](https://www.boost.org/doc/libs/master/libs/serialization/doc/index.html): needed to use TTG with classes serializable by the [Boost.Serialization](https://www.boost.org/doc/libs/master/libs/serialization/doc/index.html) library. Note that `Boost.Serialization` is not header-only, i.e., it must be compiled. This is only required if TTG is configured with CMake cache variable `TTG_PARSEC_USE_BOOST_SERIALIZATION` set to `ON`.
- ([Doxygen](http://www.doxygen.nl/), version 1.9.6 or later: needed for building documentation.
- for execution on GPGPUs and other accelerators, the following are required:
  - [CUDA compiler and runtime](https://developer.nvidia.com/cuda-zone) -- for execution on NVIDIA's CUDA-enabled accelerators. CUDA 11 or later is required.
  - [HIP/ROCm compiler and runtime](https://developer.nvidia.com/cuda-zone) -- for execution on AMD's ROCm-enabled accelerators.
  - [oneAPI DPC++/SYCL/LevelZero compiler and runtime](https://developer.nvidia.com/cuda-zone) -- for execution on Intel accelerators.

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

# build
- configure: `cmake -S /path/to/ttg/source/directory -B /path/to/ttg/build/directory <cmake args>`
- build+test: `cmake --build /path/to/ttg/build/directory --target check-ttg`
- generate HTML dox: `cmake --build /path/to/ttg/build/directory --target html-ttg`
- install: `cmake --build /path/to/ttg/build/directory --target install`

## useful cmake cache variables:

| Variable                             | Default | Description                                                                                                                                                                                                                                  |
|--------------------------------------|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `TTG_ENABLE_CUDA`                    | `OFF`   | whether to enable CUDA device support                                                                                                                                                                                                        |
| `TTG_ENABLE_HIP`                     | `OFF`   | whether to enable HIP/ROCm device support                                                                                                                                                                                                    |
| `TTG_ENABLE_LEVEL_ZERO`              | `OFF`   | whether to enable Intel oneAPI Level Zero device support                                                                                                                                                                                     |
| `BUILD_TESTING`                      | `ON`    | whether target `check-ttg` and its relatives will actually build and run unit tests                                                                                                                                                          |
| `TTG_EXAMPLES`                       | `OFF`   | whether target `check-ttg` and its relatives will actually build and run examples; setting this to `ON` will cause detection of several optional prerequisites, and (if missing) building from source                                        |
| `TTG_ENABLE_TRACE`                   | `OFF`   | setting this to `ON` will enable the ability to instrument TTG code for tracing (see `ttg::trace()`, etc.); if this is set to `OFF`, `ttg::trace()` is a no-op                                                                               |
| `TTG_PARSEC_USE_BOOST_SERIALIZATION` | `OFF`   | whether to use Boost.Serialization for serialization for the PaRSEC backend; if this is set to `OFF`, PaRSEC backend will only be able to use trivially-copyable data types or, if MADNESS backend is available, MADNESS-serializable types. |
| `TTG_FETCH_BOOST`                    | `ON`    | whether to download and build Boost automatically, if missing                                                                                                                                                                                |
| `TTG_IGNORE_BUNDLED_EXTERNALS`       | `OFF`   | whether to install and use bundled external dependencies (currently, only Boost.CallableTraits)                                                                                                                                              |
