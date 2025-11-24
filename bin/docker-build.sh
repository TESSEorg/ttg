#!/bin/sh
# SPDX-License-Identifier: BSD-3-Clause

# this script builds a TTG docker image
# usage: docker-build.sh <arch>
# where <arch> is one of the following: amd64 arm64

# to run bash in the image: docker run --privileged -i -t ttg-dev:latest /sbin/my_init -- bash -l
# see docker.md for further instructions
# locations:
#   - source dir: /home/tesse/ttg
#   - build dir: /home/tesse/ttg-build
#   - build dir: /home/tesse/ttg-install

if [ "$#" -ne 1 ] || ([ "$1" != "amd64" ] && [ "$1" != "arm64" ]); then
  echo "Usage: $0 <arch> , with <arch> either amd64 or arm64" >&2
  exit 1
fi

export CLANG_VERSION=18
export DIRNAME=`dirname $0`
export ABSDIRNAME=`pwd $DIRNAME`

# Get the TTG source root directory (parent of bin/)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TTG_SOURCE=$(cd "$SCRIPT_DIR/.." && pwd)

if [ "$1" = "amd64" ]; then
  export ARCH=amd64
  export ARCH_CMAKE=x86_64
fi
if [ "$1" = "arm64" ]; then
  export ARCH=arm64
  export ARCH_CMAKE=aarch64
fi

##############################################################
# make a script to disable ASLR to make MADWorld happy
disable_aslr="$SCRIPT_DIR/disable_aslr.sh"
cat > $disable_aslr << END
#!/bin/sh
echo 0 > /proc/sys/kernel/randomize_va_space
END
chmod +x $disable_aslr

##############################################################
# make Dockerfile
dockerfile="$SCRIPT_DIR/Dockerfile"
cat > $dockerfile << END
# Use phusion/baseimage as base image. To make your builds
# reproducible, make sure you lock down to a specific version, not
# to 'latest'! See
# https://github.com/phusion/baseimage-docker/blob/master/Changelog.md
# for a list of version numbers.
FROM phusion/baseimage:noble-1.0.2

# Use baseimage-docker's init system.
CMD ["/sbin/my_init"]

# update the OS
RUN apt-get update && apt-get upgrade -y -o Dpkg::Options::="--force-confold"

# build TTG
# 1. basic prereqs
# N.B. use libboost-all-dev to ensure that all possible boost libs are available
RUN apt-get update && apt-get install -y cmake ninja-build libopenblas-dev bison flex mpich libboost-all-dev libeigen3-dev git wget libtbb-dev clang-${CLANG_VERSION} libc++-${CLANG_VERSION}-dev libc++abi-${CLANG_VERSION}-dev && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# 2. copy local TTG source and build
RUN mkdir -p /home/tesse
COPY . /home/tesse/ttg
RUN cd /home/tesse && mkdir ttg-build && cd ttg-build && cmake ../ttg -GNinja -DCMAKE_CXX_COMPILER=clang++-${CLANG_VERSION} -DCMAKE_C_COMPILER=clang-${CLANG_VERSION} -DCMAKE_INSTALL_PREFIX=/home/tesse/ttg-install -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=ON && cmake --build . --target install

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# disable ASLR to make MADWorld happy
RUN mkdir -p /etc/my_init.d
ADD bin/disable_aslr.sh /etc/my_init.d/disable_aslr.sh

# for further info ...
ARG CACHEBUST=1
RUN echo "\e[92mDone! For info on how to use the image refer to $ABSDIRNAME/docker.md\e[0m"

END

clean_up() {
  rm -f "$disable_aslr" "$dockerfile"
  exit
}

trap clean_up SIGHUP SIGINT SIGTERM

##############################################################
# build a dev image from the TTG source root
cd "$TTG_SOURCE"
docker build --platform linux/${ARCH} -t ttg-dev --build-arg CACHEBUST=$(date +%s) -f bin/Dockerfile .

##############################################################
# extra admin tasks, uncomment as needed
# docker save ttg-dev | bzip2 > ttg-dev.docker.tar.bz2

##############################################################
# done
clean_up
