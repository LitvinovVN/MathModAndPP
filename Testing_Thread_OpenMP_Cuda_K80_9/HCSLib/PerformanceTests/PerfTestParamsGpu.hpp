#pragma once

/// @brief Параметры варьирования блоков и потоков GPU
struct PerfTestParamsGpu
{
    // Минимальное количество блоков GPU    
    unsigned gpuBlockNumMin = 1;
    // Максимальное количество блоков GPU
    unsigned gpuBlockNumMax = 1;
    // Шаг изменения количества блоков GPU
    unsigned gpuBlockNumStep = 1;

    // Минимальное количество потоков GPU    
    unsigned gpuThreadNumMin = 1;
    // Максимальное количество потоков GPU
    unsigned gpuThreadNumMax = 1;
    // Шаг изменения количества потоков GPU
    unsigned gpuThreadNumStep = 1;

    PerfTestParamsGpu()
    {}
    
    PerfTestParamsGpu(unsigned gpuBlockNumMin,
        unsigned gpuBlockNumMax,
        unsigned gpuBlockNumStep,
        unsigned gpuThreadNumMin,
        unsigned gpuThreadNumMax,
        unsigned gpuThreadNumStep) :
            gpuBlockNumMin(gpuBlockNumMin),
            gpuBlockNumMax(gpuBlockNumMax),
            gpuBlockNumStep(gpuBlockNumStep),
            gpuThreadNumMin(gpuThreadNumMin),
            gpuThreadNumMax(gpuThreadNumMax),
            gpuThreadNumStep(gpuThreadNumStep)
    {}

    void Print(PrintParams pp = PrintParams{})
    {
        std::cout << pp.startMes;        
        std::cout << "gpuBlockNumMin"   << pp.splitterKeyValue << gpuBlockNumMin;
        std::cout << pp.splitter;
        std::cout << "gpuBlockNumMax"   << pp.splitterKeyValue << gpuBlockNumMax;
        std::cout << pp.splitter;
        std::cout << "gpuBlockNumStep"  << pp.splitterKeyValue << gpuBlockNumStep;
        std::cout << pp.splitter;
        std::cout << "gpuThreadNumMin"  << pp.splitterKeyValue << gpuThreadNumMin;
        std::cout << pp.splitter;
        std::cout << "gpuThreadNumMax"  << pp.splitterKeyValue << gpuThreadNumMax;
        std::cout << pp.splitter;
        std::cout << "gpuThreadNumStep" << pp.splitterKeyValue << gpuThreadNumStep;
        std::cout << pp.endMes;

        if(pp.isEndl)
            std::cout << std::endl;
    }
};