#pragma once

#include "../CommonHelpers/FuncResult.hpp"
#include "ArrayHelper.hpp"

struct ArrayHelperFuncResult
{


template<typename T>
static
FuncResult<T> SumOpenMP(T* data, size_t size, unsigned threadsNum)
{    
    try
    {
        auto start = high_resolution_clock::now();
        T result = ArrayHelper::SumOpenMP(data, size, threadsNum);
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);        
        auto time_mks = duration.count();

        return FuncResult<T>(true, result, time_mks);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return FuncResult<T>();
    }
}

};