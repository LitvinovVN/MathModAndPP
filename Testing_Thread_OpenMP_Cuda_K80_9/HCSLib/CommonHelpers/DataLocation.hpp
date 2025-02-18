#pragma once

#include <iostream>

/// @brief Место хранения данных (векторов, матриц и пр.)
enum class DataLocation
{
    None = -2,     // Данные нигде не хранятся (нулевая, единичная матрицы и пр.)
    RAM  = -1,      
    GPU0 = 0, // Видеопамять GPU0
    GPU1 = 1, // Видеопамять GPU1
    GPU2 = 2, // Видеопамять GPU2
    GPU3 = 3  // Видеопамять GPU3
};

std::ostream& operator<<(std::ostream& os, DataLocation dl)
{
    switch (dl)
    {
    case DataLocation::None:
        os << "DataLocation::None";
        break;
    case DataLocation::RAM:
        os << "DataLocation::RAM";
        break;
    case DataLocation::GPU0:
        os << "DataLocation::GPU0";
        break;
    case DataLocation::GPU1:
        os << "DataLocation::GPU1";
        break;
    case DataLocation::GPU2:
        os << "DataLocation::GPU2";
        break;
    case DataLocation::GPU3:
        os << "DataLocation::GPU3";
        break;
            
    default:
        break;
    }

    return os;
}