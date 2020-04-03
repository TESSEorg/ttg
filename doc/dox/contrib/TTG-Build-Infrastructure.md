# TTG Build Infrastructure {#TTG-Build-Infrastructure}

TTG uses CMake metabuild system. Each dependency can be either:
- found via `find_package` (thus it must be discoverable by e.g. adding its installation prefix to `CMAKE_PREFIX_PATH`), or
- downloaded and built from source as a [CMake subproject](https://github.com/ttroy50/cmake-examples/tree/master/02-sub-projects/A-basic) of the TTG repo.

Due to the fragility/sensitivity
of ABI of C++ code to seemingly trivial toolchain/platform details (and even the API of preprocessor-infected C/C++ code)
it is stronly recommended to _let TTG download and build all prerequisites from source_! Pre-building dependencies
should be left to the package maintainers.

## Managing subprojects in TTG

Dependency source code deployment is managed by the [FetchContent module](https://cmake.org/cmake/help/latest/module/FetchContent.html). It pulls the source from the origin (git repo, tarball URL, etc.) to the
`<BUILDDIR>/_deps/<SUBPROJECT>-src/` directory and sets up the code for building in the
`<BUILD_DIR>/_deps/<SUBPROJECT>-build` directory. Although these directories are located in the build tree,
they are made part of the source tree by `add_subdirectory`. Thus in effect this mechanism
is equivalent to manually copying the source tree of the dependency into the TTG source tree.
The only difference from building the dependency as a part of TTG and standalone is
the CMake state: when configured as a subproject
the dependency CMake code will see the cache and non-cache variables and targets
defined by TTG itself and the prior dependency subprojects. Thus to be usable as a subproject
dependency CMake code needs to be designed to avoid variable and target name clashes
with the host project.

### Keeping the subprojects in sync

In order to keep TTG in sync with each subproject, one needs to update the following files:

1. `cmake/modules/FindOrFetch*.cmake` : these files decide what is the
source of the git repository for each project

2. `cmake/modules/ExternalDependenciesVersions.cmake`: define, for
each subproject the {branch, tag, revision hash} to use.

The build system should update the subproject source when
these files are changed. However, for clean rebuild you should, in addition to wiping out `CMakeCache.txt`,
remove the `build/_deps` directory.


