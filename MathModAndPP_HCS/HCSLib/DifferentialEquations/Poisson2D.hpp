#pragma once

#include <iostream>
#include "../CommonHelpers/_IncludeCommonHelpers.hpp"

/// @brief Уравнение Пуассона для двумерной области
class Poisson2D
{
    // Физическая величина
    PhysicalQuantityEnum physicalQuantity;
    // Указатель на функцию правой части
    double (*f)(double, double);
public:
    Poisson2D(PhysicalQuantityEnum physicalQuantity,
        double (*f)(double, double))
        : physicalQuantity(physicalQuantity),f(f)
    {

    }

    void Print() const
    {
        std::cout << "Poisson2D:" << std::endl;
        std::cout << "physicalQuantity: " << physicalQuantity << std::endl;
        std::cout << "f address: " << f << std::endl;
        std::cout << "f(10, 20): " << f(5.0,  10.0) << std::endl;
        std::cout << "f(20, 20): " << f(20.0, 20.0) << std::endl;
    }
};