#pragma once

#include <iostream>

/// @brief Структура для хранения поддерживаемых библиотек
struct LibSupport
{
    bool IsOpenMP = false;// Поддержка OpenMP
    bool IsCuda   = false;// Поддержка CUDA

    LibSupport()
    {
        // OpenMP
        #ifdef _OPENMP
        IsOpenMP = true;
        #endif

        // CUDA        
        #ifdef __NVCC__
        IsCuda = true;
        #endif
    }

    void Print()
    {
        std::cout << "Supported libs: ";
        if (IsOpenMP) 
            std::cout << "OpenMP ";
        if (IsCuda) 
            std::cout << "CUDA ";
        std::cout << std::endl;
    }
};

