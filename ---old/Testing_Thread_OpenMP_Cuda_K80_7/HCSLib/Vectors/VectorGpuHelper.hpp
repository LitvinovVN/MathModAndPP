#pragma once

#include <iostream>
#include "../FuncResult.hpp"
#include "VectorGpu.hpp"


class VectorGpuHelper
{
public:
    template<typename T>
    static
    FuncResult<T> SumCuda(VectorGpu<T>& v, size_t indStart, size_t indEnd, unsigned NumBlocks, unsigned Nthreads)
    {
        bool calcStatus = true;
        auto start = high_resolution_clock::now();
        T result = ArrayHelper::SumCuda(v._dev_data, indStart, indEnd, NumBlocks, Nthreads);
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);        
        auto t = duration.count();

        return FuncResult<T>(calcStatus, result, t);
    }

    template<typename T>
    static
    FuncResult<T> SumCuda(VectorGpu<T>& v, unsigned NumBlocks, unsigned Nthreads)
    {
        return SumCuda(v, 0, v.Size() - 1, NumBlocks, Nthreads);
    }

    /////////////
    // Суммирование на одном GPU с двумя видеочипами
    template<typename T>
    static
    FuncResult<T> SumCudaDevNum1GpuNum2(VectorGpu<T>& v, size_t indStart, size_t indEnd, unsigned NumBlocks, unsigned Nthreads)
    {
        bool calcStatus = true;
        auto start = high_resolution_clock::now();
        T result = ArrayHelper::SumCudaDevNum1GpuNum2(v._dev_data, indStart, indEnd, NumBlocks, Nthreads);
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);        
        auto t = duration.count();

        return FuncResult<T>(calcStatus, result, t);
    }

    // Суммирование на одном GPU с двумя видеочипами
    template<typename T>
    static
    FuncResult<T> SumCudaDevNum1GpuNum2(VectorGpu<T>& v, unsigned NumBlocks, unsigned Nthreads)
    {
        return SumCudaDevNum1GpuNum2(v, 0, v.Size() - 1, NumBlocks, Nthreads);
    }
};

