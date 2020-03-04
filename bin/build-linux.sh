#! /bin/sh

${TRAVIS_BUILD_DIR}/bin/build-boost-$TRAVIS_OS_NAME.sh
${TRAVIS_BUILD_DIR}/bin/build-eigen3-$TRAVIS_OS_NAME.sh
${TRAVIS_BUILD_DIR}/bin/build-btas-$TRAVIS_OS_NAME.sh
${TRAVIS_BUILD_DIR}/bin/build-mpich-$TRAVIS_OS_NAME.sh
#${TRAVIS_BUILD_DIR}/bin/build-madness-$TRAVIS_OS_NAME.sh
#${TRAVIS_BUILD_DIR}/bin/build-parsec-$TRAVIS_OS_NAME.sh

# Exit on error
set -ev

# download latest Doxygen
if [ "$DEPLOY" = "1" ]; then
  DOXYGEN_VERSION=1.8.17
  if [ ! -d ${INSTALL_PREFIX}/doxygen-${DOXYGEN_VERSION} ]; then
    cd ${BUILD_PREFIX} && wget http://doxygen.nl/files/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz
    cd ${INSTALL_PREFIX} && tar xzf ${BUILD_PREFIX}/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz
  fi
  export PATH=${INSTALL_PREFIX}/doxygen-${DOXYGEN_VERSION}/bin:$PATH
  which doxygen
  doxygen --version
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

cmake ${TRAVIS_BUILD_DIR} \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_PREFIX_PATH="${INSTALL_PREFIX}/eigen3" \
    -DBOOST_ROOT="${INSTALL_PREFIX}/boost" \
    -DMADNESS_ROOT_DIR="${INSTALL_PREFIX}/madness" \
    -DBTAS_INSTALL_DIR="${INSTALL_PREFIX}/BTAS" \
    -DCMAKE_CXX_FLAGS="${CXX_FLAGS} ${EXTRAFLAGS} ${CODECOVCXXFLAGS}"

### test
make serialization
tests/serialization

### examples
make test-mad t9-mad spmm-mad bspmm-mad
export MPI_HOME=${INSTALL_PREFIX}/mpich
for PROG in test-mad t9-mad spmm-mad bspmm-mad
do
  examples/$PROG
  setarch `uname -m` -R ${MPI_HOME}/bin/mpirun -n 2 examples/$PROG
done

# print ccache stats
ccache -s

