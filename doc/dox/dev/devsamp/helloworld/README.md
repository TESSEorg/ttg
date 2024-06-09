# TTG "Hello World"

This directory contains the TTG "Hello World" program

## Build

After TTG has been installed to `/path/to/ttg`, do this:

- configure: `cmake -S . -B build -DCMAKE_PREFIX_PATH="/path/to/ttg"`
- build: `cmake --build build`
- run: `./build/helloworld-parsec` or `./build/helloworld-mad`
