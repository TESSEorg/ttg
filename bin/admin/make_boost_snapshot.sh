#!/bin/sh

set -e

VERSION=1.77.0
VERSION_="$(echo ${VERSION} | sed 's/\./_/g')"
TARBALL=boost_${VERSION_}.tar.bz2

#### download
if ! test -f ${TARBALL}; then
  BOOST_URL=https://boostorg.jfrog.io/artifactory/main/release/${VERSION}/source/${TARBALL}
  echo ${BOOST_URL}
  curl -o ${TARBALL} -L ${BOOST_URL} 
fi

#### unpack
if ! test -d boost_${VERSION_}; then
  tar -xvjf ${TARBALL}
fi

#### build bcp
cd boost_${VERSION_}
./bootstrap.sh && ./b2 tools/bcp

#### extract boost/callable_traits.hpp and dependencies
mkdir -p ../../../ttg/ttg/external
dist/bin/bcp --unix-lines boost/callable_traits.hpp ../../../ttg/ttg/external

#### cleanup
rm -rf boost_${VERSION_}
