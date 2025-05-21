#pragma once

#include <iostream>

/// @brief Перечисление "Типы векторов"
enum class VectorType
{
    VectorRow,   // Вектор-строка
    VectorColumn // Вектор-столбец
};

std::ostream& operator<<(std::ostream& os, VectorType vectorType)
{
    switch (vectorType)
    {
    case VectorType::VectorRow:
        os << "VectorType::VectorRow";
        break;
    case VectorType::VectorColumn:
        os << "VectorType::VectorColumn";
        break;
            
    default:
        break;
    }

    return os;
}