#pragma once

#include <iostream>

/// @brief Тип матрицы
enum class MatrixType
{
    Zero,     // Нулевая матрица
    E,        // Единичная матрица
    Diagonal, // Диагональная матрица
    MatrixBlockRamGpus // Блочная матрица с размещением данных в RAM и нескольких GPU на одном вычислительном узле
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
    case MatrixType::Diagonal:
        os << "MatrixType::Diagonal";
        break;
    case MatrixType::MatrixBlockRamGpus:
        os << "MatrixType::MatrixBlockRamGpus";
        break;
            
    default:
        break;
    }

    return os;
}