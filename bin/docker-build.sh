#!/bin/sh

# this script builds a TTG docker image

# to run bash in the image: docker run --privileged -i -t ttg-dev:latest /sbin/my_init -- bash -l
# see docker.md for further instructions
# locations:
#   - source dir: /home/tesse/ttg
#   - build dir: /home/tesse/ttg-build
#   - build dir: /home/tesse/ttg-install

export CMAKE_VERSION=3.16.3

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
FROM phusion/baseimage:0.11

# Use baseimage-docker's init system.
CMD ["/sbin/my_init"]

# update the OS
RUN apt-get update && apt-get upgrade -y -o Dpkg::Options::="--force-confold"

# build MPQC4
# 1. basic prereqs
RUN apt-get update && apt-get install -y cmake ninja-build libblas-dev liblapack-dev liblapacke-dev flex mpich libboost-dev libeigen3-dev git wget libtbb-dev clang-8 libc++-8-dev libc++abi-8-dev && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# 2. recent cmake
RUN CMAKE_URL="https://cmake.org/files/v${CMAKE_VERSION%.[0-9]}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz" && wget --no-check-certificate -O - \$CMAKE_URL | tar --strip-components=1 -xz -C /usr/local
ENV CMAKE=/usr/local/bin/cmake
# 3. clone and build TTG
RUN mkdir -p /home/tesse && cd /home/tesse && git clone https://github.com/TESSEorg/ttg.git && mkdir ttg-build && cd ttg-build && \$CMAKE ../ttg -G Ninja -DCMAKE_CXX_COMPILER=clang++-8 -DCMAKE_C_COMPILER=clang-8 -DCMAKE_INSTALL_PREFIX=/home/tesse/ttg-install -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF && \$CMAKE --build . --target test-madness && \$CMAKE --build . --target test-parsec

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# disable ASLR to make MADWorld happy
RUN mkdir -p /etc/my_init.d
ADD $disable_aslr /etc/my_init.d/disable_aslr.sh
END

function clean_up {
  rm -f $disable_aslr Dockerfile
  exit
}

trap clean_up SIGHUP SIGINT SIGTERM

##############################################################
# build a dev image
docker build -t ttg-dev .

##############################################################
# extra admin tasks, uncomment as needed
# docker save ttg-dev | bzip2 > ttg-dev.docker.tar.bz2

##############################################################
# done
clean_up
