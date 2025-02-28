#pragma once

/// @brief Параметры выполнения тестов производительности
struct PerfTestParams
{
    // Количество итераций
    unsigned iterNumber;
    // Параметры варьирования данных    
    PerfTestParamsData perfTestParamsData;
    // Параметры варьирования количества потоков CPU
    PerfTestParamsCpu perfTestParamsCpu;
    // Параметры варьирования параметров GPU
    PerfTestParamsGpu perfTestParamsGpu;

    PerfTestParams()
    {}
    
    PerfTestParams(unsigned iterNumber,
        PerfTestParamsData perfTestParamsData,
        PerfTestParamsCpu perfTestParamsCpu,
        PerfTestParamsGpu perfTestParamsGpu) :
            iterNumber(iterNumber),
            perfTestParamsData(perfTestParamsData),
            perfTestParamsCpu(perfTestParamsCpu),
            perfTestParamsGpu(perfTestParamsGpu)
    {}

    PerfTestParams(unsigned iterNumber,
        PerfTestParamsData perfTestParamsData,
        PerfTestParamsCpu perfTestParamsCpu) :
            iterNumber(iterNumber),
            perfTestParamsData(perfTestParamsData),
            perfTestParamsCpu(perfTestParamsCpu)
    {}

    PerfTestParams(unsigned iterNumber,
        PerfTestParamsData perfTestParamsData) :
            iterNumber(iterNumber),
            perfTestParamsData(perfTestParamsData)
    {}

    void Print(PrintParams pp = PrintParams{})
    {
        std::cout << pp.startMes;
        std::cout << "iterNumber" << pp.splitterKeyValue << iterNumber;
        std::cout << pp.splitter;
        std::cout << "perfTestParamsData" << pp.splitterKeyValue;
        perfTestParamsData.Print();
        std::cout << pp.splitter;
        std::cout << "perfTestParamsCpu" << pp.splitterKeyValue;
        perfTestParamsCpu.Print();
        std::cout << pp.splitter;
        std::cout << "perfTestParamsGpu" << pp.splitterKeyValue;
        perfTestParamsGpu.Print();
        std::cout << pp.endMes;

        if(pp.isEndl)
            std::cout << std::endl;
    }
};