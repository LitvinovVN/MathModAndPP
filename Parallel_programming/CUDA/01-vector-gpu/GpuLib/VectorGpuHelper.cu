#pragma once

#include <iostream>
#include "VectorGpu.cu"

template<typename T>
__global__
void kernel_vector(VectorGpu<T> vectorGpu)
{
    int th_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_i == 0)
    {
        printf("GPU: vectorGpu._size = %d\n", vectorGpu.getSize());
        T* _dev_data_pointer = vectorGpu.get_dev_data_pointer();
        for(size_t i=0; i<vectorGpu.getSize(); i++)
        {
            printf("%lf ", _dev_data_pointer[i]);
        }
        printf("\n");
    }
}

////////////////////////////////
template<typename T>
__global__
void print_kernel(VectorGpu<T> vectorGpu)
{
    int th_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_i == 0)
    {
        printf("GPU: print_kernel() vectorGpu._size = %d\n", vectorGpu.getSize());
        T* _dev_data_pointer = vectorGpu.get_dev_data_pointer();
        for(size_t i=0; i<vectorGpu.getSize(); i++)
        {
            printf("%lf ", _dev_data_pointer[i]);
        }
        printf("\n");
    }
}

template<typename T>
class VectorGpuHelper
{
public:
    static void print(VectorGpu<T>& vectorGpu)
    {
        print_kernel<double><<<1,1>>>(vectorGpu);
        cudaDeviceSynchronize();
    }
};