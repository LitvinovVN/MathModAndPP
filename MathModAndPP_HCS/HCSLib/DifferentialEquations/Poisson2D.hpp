#pragma once

#include <iostream>
#include "../CommonHelpers/_IncludeCommonHelpers.hpp"

/// @brief Уравнение Пуассона для двумерной области
class Poisson2D
{
    // Физическая величина
    PhysicalQuantityEnum physicalQuantity;
    // Указатель на функцию правой части
    // double (*f)(double, double);
    IDiffEqFunction* f;
public:
    Poisson2D(PhysicalQuantityEnum physicalQuantity,
        IDiffEqFunction* f)
        : physicalQuantity(physicalQuantity), f(f)
    {
    }

    /// @brief Возвращает размерность объекта функции
    Dimension GetDimension() const
    {
        return Dimension::D2;
    }

    void Print() const
    {
        std::cout << "Poisson2D:" << std::endl;
        std::cout << "physicalQuantity: " << physicalQuantity << std::endl;
        std::cout << "f address: " << f << std::endl;
    }
};