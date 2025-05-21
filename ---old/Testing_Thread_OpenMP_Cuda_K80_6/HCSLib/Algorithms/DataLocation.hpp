#pragma once

#include <iostream>

/// @brief Место расположения данных
enum class DataLocation
{
    None,   // 0 - Неинициализировано
    Ram,    // 1 - ОЗУ
    Gpu,    // 2 - видеопамять GPU
    RamGpu  // 3 - ОЗУ + видеопамять GPU
};

std::ostream& operator<<(std::ostream& os, DataLocation dataLocation)
{
    switch (dataLocation)
    {
    case DataLocation::None:
        os << "None";
        break;
    case DataLocation::Ram:
        os << "Ram";
        break;
    case DataLocation::Gpu:
        os << "Gpu";
        break;
    case DataLocation::RamGpu:
        os << "RamGpu";
        break;
    default:
        break;
    }

    return os;
}