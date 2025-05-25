#pragma once

#include <iostream>
#include "../CommonHelpers/FuncResult.hpp"
#include "VectorRam.hpp"

class VectorRamHelper
{
public:
    template<typename T>
    static
    FuncResult<T> Sum(VectorRam<T>& v, size_t indStart, size_t indEnd)
    {
        bool calcStatus = true;
        auto start = high_resolution_clock::now();
        T result = ArrayHelper::Sum(v.data, indStart, indEnd);
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);        
        auto t = duration.count();

        return FuncResult<T>(calcStatus, result, t);
    }

    template<typename T>    
    static
    FuncResult<T> Sum(VectorRam<T>& v)
    {
        return Sum(v, 0, v.Length() - 1);
    }

    template<typename T>
    static
    FuncResult<T> Sum(VectorRam<T>& v, size_t indStart, size_t indEnd, unsigned threadsNum)
    {
        bool calcStatus = true;
        auto start = high_resolution_clock::now();
        T result = ArrayHelper::Sum(v.data, indStart, indEnd, threadsNum);
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);        
        auto t = duration.count();

        return FuncResult<T>(calcStatus, result, t);
    }

    template<typename T>
    static
    FuncResult<T> Sum(VectorRam<T>& v, unsigned threadsNum)
    {
        return Sum(v, 0, v.Length() - 1, threadsNum);
    }

    /////////////////// OpenMP ////////////////////
    template<typename T>
    static
    FuncResult<T> SumOpenMP(VectorRam<T>& v, size_t indStart, size_t indEnd, unsigned threadsNum)
    {
        bool calcStatus = true;
        auto start = high_resolution_clock::now();
        T result = ArrayHelper::SumOpenMP(v.data, indStart, indEnd, threadsNum);
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);        
        auto t = duration.count();

        return FuncResult<T>(calcStatus, result, t);
    }

    template<typename T>
    static
    FuncResult<T> SumOpenMP(VectorRam<T>& v, unsigned threadsNum)
    {
        return SumOpenMP(v, 0, v.Length() - 1, threadsNum);
    }
};
