#pragma once

#include <iostream>

/// @brief Место расположения данных
enum class AlgorithmDataLocation
{
    None,   // 0 - Неинициализировано
    Ram,    // 1 - ОЗУ
    Gpu,    // 2 - видеопамять GPU
    RamGpu  // 3 - ОЗУ + видеопамять GPU
};

std::ostream& operator<<(std::ostream& os, AlgorithmDataLocation dataLocation)
{
    switch (dataLocation)
    {
    case AlgorithmDataLocation::None:
        os << "None";
        break;
    case AlgorithmDataLocation::Ram:
        os << "Ram";
        break;
    case AlgorithmDataLocation::Gpu:
        os << "Gpu";
        break;
    case AlgorithmDataLocation::RamGpu:
        os << "RamGpu";
        break;
    default:
        break;
    }

    return os;
}