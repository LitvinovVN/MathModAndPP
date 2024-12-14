#pragma once

#include <vector>
#include "FuncResult.hpp"
#include "VectorRam.hpp"
#include "VectorGpu.hpp"
#include "TestParams.hpp"


/// @brief Вспомогательный класс для запуска численных экспериментов
class TestHelper
{
public:
    template<typename T>
    static std::vector<FuncResult<T>> LaunchSum(VectorRam<T>& v, TestParams p)
    {
        std::cout << "-------LaunchSum(VectorRam<T>& v) Start ------" << std::endl;
        auto iterNum = p.IterNum;
        std::vector<FuncResult<T>> results;
        for(unsigned i{0}; i < iterNum; i++)
        {
            FuncResult<T> res = VectorRamHelper::Sum(v);
            results.push_back(res);
        }
        
        std::cout << "-------LaunchSum(VectorRam<T>& v) End --------" << std::endl;
        return results;
    }

    template<typename T>
    static std::vector<FuncResult<T>> LaunchSum(VectorRam<T>& v, unsigned Nthreads, TestParams p)
    {
        std::cout << "-------LaunchSum(VectorRam<T>& v, unsigned Nthreads) Start ------" << std::endl;
        auto iterNum = p.IterNum;
        std::vector<FuncResult<T>> results;
        for(unsigned i{0}; i < iterNum; i++)
        {
            FuncResult<T> res = VectorRamHelper::Sum(v, Nthreads);
            results.push_back(res);
        }
        
        std::cout << "-------LaunchSum(VectorRam<T>& v, unsigned Nthreads) End --------" << std::endl;
        return results;
    }

    template<typename T>
    static std::vector<FuncResult<T>> LaunchSumOpenMP(VectorRam<T>& v, unsigned Nthreads, TestParams p)
    {
        std::cout << "-------LaunchSumOpenMP(VectorRam<T>& v, unsigned Nthreads) Start ------" << std::endl;
        auto iterNum = p.IterNum;
        std::vector<FuncResult<T>> results;

        #ifdef _OPENMP

        for(unsigned i{0}; i < iterNum; i++)
        {
            FuncResult<T> res = VectorRamHelper::SumOpenMP(v, Nthreads);
            results.push_back(res);
        }

        #endif
        
        std::cout << "-------LaunchSumOpenMP(VectorRam<T>& v, unsigned Nthreads) End --------" << std::endl;
        return results;
    }

    
    template<typename T>
    static std::vector<FuncResult<T>> LaunchSumCuda(VectorGpu<T>& v, unsigned NumBlocks, unsigned Nthreads, TestParams p)
    {
        std::cout << "-------LaunchSumCuda(VectorGpu<T>& v, unsigned NumBlocks, unsigned Nthreads, TestParams p) Start ------" << std::endl;
        std::vector<FuncResult<T>> results;

        #ifdef __NVCC__

        auto iterNum = p.IterNum;
        for(unsigned i{0}; i < iterNum; i++)
        {
            FuncResult<T> res = VectorGpuHelper::SumCuda(v, NumBlocks, Nthreads);
            results.push_back(res);
        }

        #endif
        
        std::cout << "-------LaunchSumCuda(VectorGpu<T>& v, unsigned NumBlocks, unsigned Nthreads, TestParams p) End --------" << std::endl;
        return results;
    }
};

