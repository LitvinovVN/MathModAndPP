#pragma once

#include <iostream>

/// @brief Размерность для геометрии и пр.
enum class PhysicalQuantityEnum
{
    None,  // Безразмерная величина, о.е.
    T = 1, // Температура, град. Цельсия 
    P = 2  // Давление, Па
};

std::ostream& operator<<(std::ostream& os, PhysicalQuantityEnum dim)
{
    switch (dim)
    {
    case PhysicalQuantityEnum::None:
        os << "None";
        break;
    case PhysicalQuantityEnum::T:
        os << "T";
        break;
    case PhysicalQuantityEnum::P:
        os << "P";
        break;
            
    default:
        break;
    }

    return os;
}