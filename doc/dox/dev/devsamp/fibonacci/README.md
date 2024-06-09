# Largest Fibonacci number

This directory contains TTG programs computing the largest Fibonacci number smaller than $N$:

- CPU version: `fibonacci.cc`
- Device version: `fibonacci_device.cc`
  - CUDA kernel: `fibonacci_cuda_kernel.{cu,h}`

## Build

After TTG has been installed to `/path/to/ttg`, do this:

- configure: `cmake -S . -B build -DCMAKE_PREFIX_PATH="/path/to/ttg"`
- build:
  - CPU version: `cmake --build build --target fibonacci`
  - CUDA version (TTG must have been configured with CUDA support): `cmake --build build --target fibonacci_cuda`
- run: `./build/fibonacci N` or `./build/fibonacci_cuda N`
