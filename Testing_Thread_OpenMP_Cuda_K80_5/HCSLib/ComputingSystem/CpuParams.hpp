#pragma once

#include "../PrintParams.hpp"

/// @brief Параметры центрального процессора
struct CpuParams
{
    /// @brief УИД CPU
    unsigned id{0};
    /// @brief Наименование CPU
    std::string name{""};    
    /// @brief Количество поток
    unsigned ThreadsNumber{0};
    

    void Print(PrintParams pp)
    {
        //std::cout << "CpuParams::Print(PrintParams pp)" << std::endl;
        std::cout << pp.startMes;

        std::cout << "id"              << pp.splitterKeyValue << id;
        std::cout << pp.splitter;
        std::cout << "name"            << pp.splitterKeyValue << name;
        std::cout << pp.splitter;
        std::cout << "ThreadsNumber"   << pp.splitterKeyValue << ThreadsNumber;
        
        std::cout << pp.endMes;

        if(pp.isEndl)
            std::cout << std::endl;
    }
};