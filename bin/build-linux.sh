#! /bin/sh

#
# install prerequisites
#

# need more recent cmake than available
if [ ! -d "${INSTALL_PREFIX}/cmake" ]; then
  CMAKE_VERSION=3.17.0
  CMAKE_URL="https://cmake.org/files/v${CMAKE_VERSION%.[0-9]}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz"
  mkdir ${INSTALL_PREFIX}/cmake && wget --no-check-certificate -O - ${CMAKE_URL} | tar --strip-components=1 -xz -C ${INSTALL_PREFIX}/cmake
fi
export PATH=${INSTALL_PREFIX}/cmake/bin:${PATH}
cmake --version

${TRAVIS_BUILD_DIR}/bin/build-boost-$TRAVIS_OS_NAME.sh
${TRAVIS_BUILD_DIR}/bin/build-eigen3-$TRAVIS_OS_NAME.sh
${TRAVIS_BUILD_DIR}/bin/build-mpich-$TRAVIS_OS_NAME.sh

# Exit on error
set -ev

# download latest Doxygen
if [ "$DEPLOY" = "1" ]; then
  DOXYGEN_VERSION=1.9.1
  if [ ! -d ${INSTALL_PREFIX}/doxygen-${DOXYGEN_VERSION} ]; then
    cd ${BUILD_PREFIX} && wget http://doxygen.nl/files/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz
    cd ${INSTALL_PREFIX} && tar xzf ${BUILD_PREFIX}/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz
  fi
  export PATH=${INSTALL_PREFIX}/doxygen-${DOXYGEN_VERSION}/bin:$PATH
  which doxygen
  doxygen --version
fi

# use Ninja for Debug builds and Make for Release
if [ "$BUILD_TYPE" = "Debug" ]; then
    export CMAKE_GENERATOR="Ninja"
else
    export CMAKE_GENERATOR="Unix Makefiles"
fi

#
# Environment variables
#
export CXX_FLAGS="-mno-avx -ftemplate-depth=1024"
if [ "$CXX" = "g++" ]; then
    export CC=/usr/bin/gcc-$GCC_VERSION
    export CXX=/usr/bin/g++-$GCC_VERSION
    # ggc-min params to try to reduce peak memory consumption ... typical ICE under Travis is due to insufficient memory
    export EXTRAFLAGS="--param ggc-min-expand=20 --param ggc-min-heapsize=2048000"
else
    export CC=/usr/bin/clang-$CLANG_VERSION
    export CXX=/usr/bin/clang++-$CLANG_VERSION
    export EXTRAFLAGS="-Wno-unused-command-line-argument -stdlib=libc++"
fi

echo $($CC --version)
echo $($CXX --version)

# list the prebuilt prereqs
ls ${INSTALL_PREFIX}

# where to install
export INSTALL_DIR=${INSTALL_PREFIX}/ttg

# make build dir
cd ${BUILD_PREFIX}
mkdir -p ttg
cd ttg

# configure CodeCov only for g++-8 debug build
if [ "$COMPUTE_COVERAGE" = "1" ]; then
    export CODECOVCXXFLAGS="--coverage -O0"
fi

cmake ${TRAVIS_BUILD_DIR} -G "${CMAKE_GENERATOR}" \
    -DCMAKE_TOOLCHAIN_FILE=${TRAVIS_BUILD_DIR}/cmake/toolchains/travis.cmake \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_PREFIX_PATH="${INSTALL_PREFIX}/eigen3" \
    -DBOOST_ROOT="${INSTALL_PREFIX}/boost" \
    -DCMAKE_CXX_FLAGS="${CXX_FLAGS} ${EXTRAFLAGS} ${CODECOVCXXFLAGS}"

### examples
cmake --build .
export MPI_HOME=${INSTALL_PREFIX}/mpich
# run examples
for RUNTIME in mad parsec
do
for EXAMPLE in test t9 spmm bspmm
do
for NPROC in 1 2
do
  examples/$EXAMPLE-$RUNTIME
  setarch `uname -m` -R ${MPI_HOME}/bin/mpirun -n $NPROC examples/$EXAMPLE-$RUNTIME
done
done
done

### tests
cmake --build . --target serialization
tests/serialization

# print ccache stats
ccache -s

