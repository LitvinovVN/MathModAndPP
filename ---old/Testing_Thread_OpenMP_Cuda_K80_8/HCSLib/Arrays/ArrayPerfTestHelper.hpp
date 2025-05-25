#pragma once

#include <iostream>

struct ArrayPerfTestHelper
{

static PerfTestResults PerfTest_SumOpenMP(PerfTestParams perfTestParams)
{
    std::cout << "ArrayPerfTestHelper::PerfTest_SumOpenMP()\n";

    size_t arrLengthMin  = perfTestParams.perfTestParamsData.arrayLengthMin;
    size_t arrLengthMax  = perfTestParams.perfTestParamsData.arrayLengthMax;
    size_t arrLengthStep = perfTestParams.perfTestParamsData.arrayLengthStep;

    for (size_t dataLength = arrLengthMin;
            dataLength <= arrLengthMax;
            dataLength += arrLengthStep)
    {// Цикл по размеру массива (начало)
        std::cout << "---------- dataLength: " << dataLength << std::endl;

        auto array = ArrayHelper::CreateArrayRam<double>(dataLength);
        ArrayHelper::InitArrayRam(array, dataLength, 0.001);
        
        auto cpuThreadNumMin = perfTestParams.perfTestParamsCpu.cpuThreadsNumMin;
        auto cpuThreadNumMax = perfTestParams.perfTestParamsCpu.cpuThreadsNumMax;
        auto cpuThreadNumStep = perfTestParams.perfTestParamsCpu.cpuThreadsNumStep;
        for (auto cpuThreadsNum = cpuThreadNumMin;
            cpuThreadsNum <= cpuThreadNumMax;
            cpuThreadsNum += cpuThreadNumStep)
        {// Цикл по количеству потоков CPU (начало)
            std::cout << "----- cpuThreadsNum: " << cpuThreadsNum << std::endl;
            std::vector<FuncResult<double>> results;
            
            unsigned iterNumber = perfTestParams.iterNumber;
            for (auto iterCnt = 0; iterCnt < iterNumber; iterCnt += 1)
            {// Цикл по количеству итераций (начало)
                auto result = ArrayHelperFuncResult::SumOpenMP(array, dataLength, cpuThreadsNum);
                results.push_back(result);
                result.Print();
            }// Цикл по количеству итераций (конец)
            CalculationStatistics stat(results);
            stat.Print();
            ParallelCalcIndicators parallelCalcIndicators{};
            parallelCalcIndicators.Print();

            std::cout << "-----" << std::endl;    
        }// Цикл по количеству потоков CPU (конец)
        

        ArrayHelper::DeleteArrayRam(array);

        std::cout << "----------" << std::endl;
    }// Цикл по размеру массива (конец)
    



    PerfTestResults results;

    return results;
}

static PerfTestResults PerfTest_SumCublas(PerfTestParams perfTestParams)
{
    std::cout << "ArrayPerfTestHelper::PerfTest_SumCublas()\n";
    cublasHandle_t cublasH = CublasHelper::CublasCreate();

    size_t arrLengthMin  = perfTestParams.perfTestParamsData.arrayLengthMin;
    size_t arrLengthMax  = perfTestParams.perfTestParamsData.arrayLengthMax;
    size_t arrLengthStep = perfTestParams.perfTestParamsData.arrayLengthStep;

    for (size_t dataLength = arrLengthMin;
            dataLength <= arrLengthMax;
            dataLength += arrLengthStep)
    {// Цикл по размеру массива (начало)
        std::cout << "---------- dataLength: " << dataLength << std::endl;

        auto array = ArrayHelper::CreateArrayGpu<double>(dataLength);
        ArrayHelper::InitArrayGpu(array, dataLength, 0.001);
        std::vector<FuncResult<double>> results;
        
        unsigned iterNumber = perfTestParams.iterNumber;
        for (auto iterCnt = 0; iterCnt < iterNumber; iterCnt += 1)
        {// Цикл по количеству итераций (начало)
            auto result = ArrayHelperFuncResult::SumCublas(cublasH, array, dataLength);
            results.push_back(result);
            result.Print();
        }// Цикл по количеству итераций (конец)
        CalculationStatistics stat(results);
        stat.Print();
        ParallelCalcIndicators parallelCalcIndicators{};
        parallelCalcIndicators.Print();

        std::cout << "-----" << std::endl;    
        
        ArrayHelper::DeleteArrayGpu(array);

        std::cout << "----------" << std::endl;
    }// Цикл по размеру массива (конец)
    

    CublasHelper::CublasDestroy(cublasH);

    PerfTestResults results;

    return results;
}


};



