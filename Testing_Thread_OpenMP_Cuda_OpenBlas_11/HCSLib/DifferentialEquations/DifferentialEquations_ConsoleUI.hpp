#pragma once

#include <iostream>

struct DifferentialEquations_ConsoleUI
{
    static void Poisson2D_ConsoleUI()
    {
        std::cout << "Poisson2D_ConsoleUI()\n";
        auto f = [](double x, double y)
        {
            if (x < 10 )
                return 111.1;
            
            return 0.0;
        };
        auto poisson2D = DifferentialEquationsRepository::GetPoisson2D(PhysicalQuantityEnum::T, f);
        poisson2D.Print();
    }
};