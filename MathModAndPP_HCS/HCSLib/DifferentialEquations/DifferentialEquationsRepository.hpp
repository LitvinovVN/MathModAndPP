#pragma once

#include <iostream>
#include "_IncludeDifferentialEquations.hpp"

/// @brief Репозиторий дифференциальных уравнений
class DifferentialEquationsRepository
{
    

public:
    static Poisson2D GetPoisson2D(PhysicalQuantityEnum physicalQuantity,
        IDiffEqFunction* f)
    {
         return Poisson2D(physicalQuantity, f);
    }
    

    /// @brief Выводит в консоль сведения об объекте
    void Print() const
    {
        std::cout << "DifferentialEquationsRepository" << std::endl;
        
    }

};