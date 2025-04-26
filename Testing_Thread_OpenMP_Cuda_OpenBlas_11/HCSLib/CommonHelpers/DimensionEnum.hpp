#pragma once

#include <iostream>

/// @brief Размерность для геометрии и пр.
enum class Dimension
{   
    D1 = 1, // 1D 
    D2 = 2, // 2D
    D3 = 3  // 3D
};

std::ostream& operator<<(std::ostream& os, Dimension dim)
{
    switch (dim)
    {
    case Dimension::D1:
        os << "1D";
        break;
    case Dimension::D2:
        os << "2D";
        break;
    case Dimension::D3:
        os << "3D";
        break;
            
    default:
        break;
    }

    return os;
}