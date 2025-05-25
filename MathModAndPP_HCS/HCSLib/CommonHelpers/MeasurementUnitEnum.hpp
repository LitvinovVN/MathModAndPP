#pragma once

#include <iostream>

/// @brief Размерность для геометрии и пр.
enum class MeasurementUnitEnum
{   
    Meter = 1, // Метры 
    Pascal = 2, // Паскали
    Gramm = 3  // Граммы
};

std::ostream& operator<<(std::ostream& os, MeasurementUnitEnum mu)
{
    switch (mu)
    {
    case MeasurementUnitEnum::Meter:
        os << "m.";
        break;
    case MeasurementUnitEnum::Pascal:
        os << "Pa.";
        break;
    case MeasurementUnitEnum::Gramm:
        os << "g.";
        break;
            
    default:
        break;
    }

    return os;
}