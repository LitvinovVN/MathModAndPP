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

// cuda-ядро для вывода вектора в консоль
template<typename T>
__global__
void print_kernel(VectorGpu<T> vectorGpu, size_t indStart, size_t length)
{
    int th_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_i == 0)
    {
        //printf("GPU: print_kernel() vectorGpu._size = %d\n", vectorGpu.getSize());
        T* _dev_data_pointer = vectorGpu.get_dev_data_pointer();
        auto indEnd = indStart + length - 1;
        if(indEnd > vectorGpu.getSize())
        {
            printf("Error! indEnd > vectorGpu.getSize()\n");
            return;
        }

        printf("[%ld..", indStart);
        printf("%ld]: ", indEnd);
        for(size_t i = indStart; i <= indEnd; i++)
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
    static void Print(VectorGpu<T>& vectorGpu, size_t indStart, size_t length = 1)
    {
        auto indEnd = indStart + length;
        
        if(indEnd > vectorGpu.getSize())
        {            
            throw std::range_error("indEnd > vectorGpu.getSize()");                
        }

        print_kernel<double><<<1,1>>>(vectorGpu, indStart, length);
        cudaDeviceSynchronize();
    }

    static void Print(VectorGpu<T>& vectorGpu)
    {        
        Print(vectorGpu, 0, vectorGpu.getSize());        
    }
};