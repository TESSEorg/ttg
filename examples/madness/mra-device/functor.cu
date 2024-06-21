#include <iostream>
#include <cuda_runtime.h>

// Note: To be added to TTG
#define N 2048
#define THREADS_PER_BLOCK 512

template <typename T>
void random_ints(T *arr, int size) {
    for (int i = 0; i < size; ++i) {
        arr[i] = rand() % 100; // Generate random int between 0 and 99
    }
}

template <typename T>
__device__ float cexp(T a) {
    return std::exp(a);
}

class functor {
    public:
        template <typename T>
        __device__ T operator()(T x) const{
            return cexp(x);
        }
};

template <typename T, class F>
__global__ void kernel(const F &f, T *x, T *y, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) y[index] = f(x[index]);
}

int main(){
    float *a, *c, *da, *dc;
    functor f, *d_f;

    a = (float*)malloc(sizeof(float)*N); random_ints(a, N);
    c = (float*)malloc(sizeof(float)*N);

    cudaMalloc(&d_f, sizeof(functor));
    cudaMalloc(&da, sizeof(float)*N);
    cudaMalloc(&dc, sizeof(float)*N);
    cudaMemcpy(d_f, &f, sizeof(functor), cudaMemcpyHostToDevice);
    cudaMemcpy(da, a, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, c, sizeof(float)*N, cudaMemcpyHostToDevice);

    kernel<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(*d_f, da, dc, N);

    cudaMemcpy(c, dc, sizeof(float)*N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        std::cout << c[i] << std::endl;
    }

    free(a); free(c);
    cudaFree(da); cudaFree(dc); cudaFree(d_f);

    return 0;

}
