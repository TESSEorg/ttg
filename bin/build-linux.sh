#! /bin/sh

${TRAVIS_BUILD_DIR}/bin/build-boost-$TRAVIS_OS_NAME.sh
${TRAVIS_BUILD_DIR}/bin/build-eigen3-$TRAVIS_OS_NAME.sh
${TRAVIS_BUILD_DIR}/bin/build-btas-$TRAVIS_OS_NAME.sh
${TRAVIS_BUILD_DIR}/bin/build-mpich-$TRAVIS_OS_NAME.sh
${TRAVIS_BUILD_DIR}/bin/build-madness-$TRAVIS_OS_NAME.sh
#${TRAVIS_BUILD_DIR}/bin/build-parsec-$TRAVIS_OS_NAME.sh

# Exit on error
set -ev

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
if [ "$BUILD_TYPE" = "Debug" ] && [ "$GCC_VERSION" = 8 ]; then
    export CODECOVCXXFLAGS="--coverage -O0"
fi

cmake ${TRAVIS_BUILD_DIR} \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DBOOST_ROOT="${INSTALL_PREFIX}/boost" \
    -DMADNESS_ROOT_DIR="${INSTALL_PREFIX}/madness" \
    -DEIGEN3_INCLUDE_DIR="${INSTALL_PREFIX}/eigen3" \
    -DBTAS_INSTALL_DIR="${INSTALL_PREFIX}/btas" \
    -DCMAKE_CXX_FLAGS="${CXX_FLAGS} ${EXTRAFLAGS} ${CODECOVCXXFLAGS}"

### test
#make -j2 check VERBOSE=1

### examples
make test-mad t9-mad spmm-mad bspmm-mad serialization
./examples/ttg-mad
./examples/t9-mad
./examples/spmm-mad
./examples/bspmm-mad
./tests/serialization

# print ccache stats
ccache -s

