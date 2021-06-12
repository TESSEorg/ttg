#!/bin/sh

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

export CMAKE_VERSION=3.19.5
export DIRNAME=`dirname $0`
export ABSDIRNAME=`pwd $DIRNAME`
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
disable_aslr=disable_aslr.sh
cat > $disable_aslr << END
#!/bin/sh
echo 0 > /proc/sys/kernel/randomize_va_space
END
chmod +x $disable_aslr

##############################################################
# make Dockerfile, if missing
cat > Dockerfile << END
# Use phusion/baseimage as base image. To make your builds
# reproducible, make sure you lock down to a specific version, not
# to 'latest'! See
# https://github.com/phusion/baseimage-docker/blob/master/Changelog.md
# for a list of version numbers.
FROM phusion/baseimage:master-${ARCH}

# Use baseimage-docker's init system.
CMD ["/sbin/my_init"]

# update the OS
RUN apt-get update && apt-get upgrade -y -o Dpkg::Options::="--force-confold"

# build MPQC4
# 1. basic prereqs
RUN apt-get update && apt-get install -y cmake ninja-build libblas-dev liblapack-dev liblapacke-dev bison flex mpich libboost-dev libeigen3-dev git wget libtbb-dev clang-8 libc++-8-dev libc++abi-8-dev && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# 2. recent cmake
RUN CMAKE_URL="https://cmake.org/files/v${CMAKE_VERSION%.[0-9]}/cmake-${CMAKE_VERSION}-Linux-${ARCH_CMAKE}.tar.gz" && wget --no-check-certificate -O - \$CMAKE_URL | tar --strip-components=1 -xz -C /usr/local
ENV CMAKE=/usr/local/bin/cmake
# 3. clone and build TTG
RUN mkdir -p /home/tesse && cd /home/tesse && git clone https://github.com/TESSEorg/ttg.git && mkdir ttg-build && cd ttg-build && \$CMAKE ../ttg -GNinja -DCMAKE_CXX_COMPILER=clang++-8 -DCMAKE_C_COMPILER=clang-8 -DCMAKE_INSTALL_PREFIX=/home/tesse/ttg-install -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF && \$CMAKE --build . --target install

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# disable ASLR to make MADWorld happy
RUN mkdir -p /etc/my_init.d
ADD $disable_aslr /etc/my_init.d/disable_aslr.sh

# for further info ...
ARG CACHEBUST=1
RUN echo "\e[92mDone! For info on how to use the image refer to $ABSDIRNAME/docker.md\e[0m"

END

function clean_up {
  rm -f $disable_aslr Dockerfile
  exit
}

trap clean_up SIGHUP SIGINT SIGTERM

##############################################################
# build a dev image
docker build -t ttg-dev --build-arg CACHEBUST=$(date +%s) .

##############################################################
# extra admin tasks, uncomment as needed
# docker save ttg-dev | bzip2 > ttg-dev.docker.tar.bz2

##############################################################
# done
clean_up
