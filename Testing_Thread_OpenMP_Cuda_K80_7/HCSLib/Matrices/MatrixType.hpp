#pragma once

#include <iostream>

/// @brief Тип матрицы
enum class MatrixType
{
    Zero,   // Нулевая матрица
    E       // Единичная матрица
};

std::ostream& operator<<(std::ostream& os, MatrixType fdt)
{
    switch (fdt)
    {
    case MatrixType::Zero:
        os << "MatrixType::Zero";
        break;
    case MatrixType::E:
        os << "MatrixType::E";
        break;
            
    default:
        break;
    }

    return os;
}