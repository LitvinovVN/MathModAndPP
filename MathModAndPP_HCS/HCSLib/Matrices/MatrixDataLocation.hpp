#pragma once

#include <iostream>

/// @brief Место хранения матрицы
enum class MatrixDataLocation
{
    None = -2,     // Данные нигде не хранятся (нулевая, единичная матрицы и пр.)
    RAM  = -1,      
    GPU0 = 0, // Видеопамять GPU0
    GPU1 = 1, // Видеопамять GPU1
    GPU2 = 2, // Видеопамять GPU2
    GPU3 = 3  // Видеопамять GPU3
};

std::ostream& operator<<(std::ostream& os, MatrixDataLocation fdt)
{
    switch (fdt)
    {
    case MatrixDataLocation::None:
        os << "MatrixDataLocation::None";
        break;
    case MatrixDataLocation::RAM:
        os << "MatrixDataLocation::RAM";
        break;
    case MatrixDataLocation::GPU0:
        os << "MatrixDataLocation::GPU0";
        break;
    case MatrixDataLocation::GPU1:
        os << "MatrixDataLocation::GPU1";
        break;
    case MatrixDataLocation::GPU2:
        os << "MatrixDataLocation::GPU2";
        break;
    case MatrixDataLocation::GPU3:
        os << "MatrixDataLocation::GPU3";
        break;
            
    default:
        break;
    }

    return os;
}