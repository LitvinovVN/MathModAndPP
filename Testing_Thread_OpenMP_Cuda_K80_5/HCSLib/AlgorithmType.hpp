#pragma once

#include <iostream>

/// @brief Тип алгоритма:
enum class AlgorithmType
{
    None,        // Неинициализировано
    SeqCpu,      // 1 - последовательный CPU
    SeqGpuCuda,  // 2 - последовательный GPU CUDA
    ParCpuThread,// 3 - параллельный CPU std::thread
    ParCpuOpenMP,// 4 - параллельный CPU OpenMP
    ParGpuCuda   // 5 - параллельный GPU CUDA
};

std::ostream& operator<<(std::ostream& os, AlgorithmType algType)
{
    switch (algType)
    {
    case AlgorithmType::None:
        os << "None";
        break;
    case AlgorithmType::SeqCpu:
        os << "SeqCpu";
        break;
    case AlgorithmType::SeqGpuCuda:
        os << "SeqGpuCuda";
        break;
    case AlgorithmType::ParCpuThread:
        os << "ParCpuThread";
        break;
    case AlgorithmType::ParCpuOpenMP:
        os << "ParCpuOpenMP";
        break;
    case AlgorithmType::ParGpuCuda:
        os << "ParGpuCuda";
        break;
    default:
        break;
    }

    return os;
}