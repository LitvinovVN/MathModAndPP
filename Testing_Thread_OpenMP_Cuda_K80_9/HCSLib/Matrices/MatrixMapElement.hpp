#pragma once

#include <iostream>
#include "MatrixType.hpp"
#include "MatrixDataLocation.hpp"
#include "IMatrix.hpp"

/// @brief Элемент карты блочной матрицы
struct MatrixMapElement
{
    // Индекс столбца блочной матрицы
    unsigned   columnIndex = 0;
    // Тип матрицы
    MatrixType matrixType  = MatrixType::Zero;
    // Место хранения данных матрицы
    MatrixDataLocation matrixDataLocation = MatrixDataLocation::None;    
    // Указатель на объект матрицы
    IMatrix*   matrixPtr   = nullptr;
};