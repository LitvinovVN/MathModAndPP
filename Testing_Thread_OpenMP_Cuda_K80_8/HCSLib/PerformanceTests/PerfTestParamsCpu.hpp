#pragma once

/// @brief Параметры варьирования потоков CPU
struct PerfTestParamsCpu
{
    // Минимальное количество потоков CPU    
    unsigned cpuThreadsNumMin = 1;
    // Максимальное количество потоков CPU
    unsigned cpuThreadsNumMax = 20;
    // Шаг изменения количества потоков CPU
    unsigned cpuThreadsNumStep = 1;

    PerfTestParamsCpu()
    {}
    
    PerfTestParamsCpu(unsigned cpuThreadsNumMin,
        unsigned cpuThreadsNumMax,
        unsigned cpuThreadsNumStep) :
            cpuThreadsNumMin(cpuThreadsNumMin),
            cpuThreadsNumMax(cpuThreadsNumMax),
            cpuThreadsNumStep(cpuThreadsNumStep)
    {}

    void Print(PrintParams pp = PrintParams{})
    {
        std::cout << pp.startMes;        
        std::cout << "cpuThreadsNumMin" << pp.splitterKeyValue << cpuThreadsNumMin;
        std::cout << pp.splitter;
        std::cout << "cpuThreadsNumMax" << pp.splitterKeyValue << cpuThreadsNumMax;
        std::cout << pp.splitter;
        std::cout << "cpuThreadsNumStep" << pp.splitterKeyValue << cpuThreadsNumStep;
        std::cout << pp.endMes;

        if(pp.isEndl)
            std::cout << std::endl;
    }
};