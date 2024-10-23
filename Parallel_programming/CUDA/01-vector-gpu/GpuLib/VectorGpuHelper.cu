#pragma once

#include <iostream>
#include "VectorGpu.cu"

// cuda-ядро для вывода вектора в консоль
template<typename T>
__global__
void print_kernel(VectorGpu<T> vectorGpu, size_t indStart, size_t length)
{
    int th_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_i == 0)
    {
        //printf("GPU: print_kernel() vectorGpu._size = %d\n", vectorGpu.GetSize());
        T* _dev_data_pointer = vectorGpu.Get_dev_data_pointer();
        auto indEnd = indStart + length - 1;
        if(indEnd > vectorGpu.GetSize())
        {
            printf("Error! indEnd > vectorGpu.GetSize()\n");
            return;
        }

        printf("[%d..", (long)indStart);
        printf("%d]: ", (long)indEnd);
        for(size_t i = indStart; i <= indEnd; i++)
        {
            printf("%f ", _dev_data_pointer[i]);
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
        
        if(indEnd > vectorGpu.GetSize())
        {            
            throw std::range_error("indEnd > vectorGpu.GetSize()");                
        }

        print_kernel<double><<<1,1>>>(vectorGpu, indStart, length);
        cudaDeviceSynchronize();
    }

    static void Print(VectorGpu<T>& vectorGpu)
    {        
        Print(vectorGpu, 0, vectorGpu.getSize());        
    }
};