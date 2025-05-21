#pragma once

#include "../CommonHelpers/PrintParams.hpp"

/// @brief Параметры центрального процессора
struct RamParams
{    
    /// @brief Объём доступной для вычислений RAM, Гб
    unsigned RamSizeGb{0};
    
    /// @brief Пропускная способность RAM, Гб/c
    double RamBandwidthGbS{0};

    void Print(PrintParams pp)
    {
        //std::cout << "CpuParams::Print(PrintParams pp)" << std::endl;
        std::cout << pp.startMes;
        
        std::cout << "RamSizeGb"        << pp.splitterKeyValue << RamSizeGb;
        std::cout << pp.splitter;
        std::cout << "RamBandwidthGbS"  << pp.splitterKeyValue << RamBandwidthGbS;
        
        std::cout << pp.endMes;

        if(pp.isEndl)
            std::cout << std::endl;
    }
};