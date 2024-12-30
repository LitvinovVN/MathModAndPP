#pragma once

#include <iostream>

#include "../PrintParams.hpp"

/// @brief Метрики алгоритма
struct AlgorithmMetrics
{
    // Объём дополнительной памяти ОЗУ
    size_t allocRam;
    // Объём дополнительной памяти Gpu
    size_t allocGpu;
    // количество считываний из памяти ОЗУ
    size_t readRam;
    // количество считываний из глобальной памяти Gpu
    size_t readGpu;
    // количество суммирований и вычитаний
    size_t arifmSumSub;
    // количество умножений и делений
    size_t arifmMultDiv;

    void Print(PrintParams pp)
    {
        std::cout << pp.startMes;

        std::cout << "allocRam"     << pp.splitterKeyValue << allocRam << pp.splitter;
        std::cout << "allocGpu"     << pp.splitterKeyValue << allocGpu << pp.splitter;
        std::cout << "readRam"      << pp.splitterKeyValue << readRam << pp.splitter;
        std::cout << "readGpu"      << pp.splitterKeyValue << readGpu << pp.splitter;
        std::cout << "arifmSumSub"  << pp.splitterKeyValue << arifmSumSub << pp.splitter;
        std::cout << "arifmMultDiv" << pp.splitterKeyValue << arifmMultDiv << pp.splitter;

        std::cout << pp.endMes;
        if(pp.isEndl)
            std::cout << std::endl;
    }
};
