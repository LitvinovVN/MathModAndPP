#pragma once

#include <iostream>

struct DifferentialEquations_ConsoleUI
{
    static void Poisson2D_ConsoleUI()
    {
        std::cout << "Poisson2D_ConsoleUI()\n";
        /*auto f = [](double x, double y)
        {
            if (x < 10 )
                return 111.1;
            
            return 0.0;
        };*/
        auto f = new DiffEqFunc2DPointSources();
        f->AddPointSource(10,20,100);
        f->AddPointSource(20,10,80);
        auto poisson2D = DifferentialEquationsRepository::GetPoisson2D(PhysicalQuantityEnum::T, f);
        poisson2D.Print();
    }

    static void DiffEqFunc2DPointSources_ConsoleUI()
    {
        std::cout << "DiffEqFunc2DPointSources_ConsoleUI()\n";

        auto f = new DiffEqFunc2DPointSources();
        f->AddPointSource(10,20,100);
        f->AddPointSource(20,10,80);
        f->Print();
        ((IDiffEqFunction*)f)->Print();
    }

    static void DiffEqFunc2D_ConsoleUI()
    {
        std::cout << "DiffEqFunc2D_ConsoleUI()\n";

        auto f = [](double x, double y)
        {
            return 10 * sin(x) + 5 * cos(y);
        };
        IDiffEqFunction* idf = new DiffEqFunc2D(f);        
        idf->Print();
        std::cout << "f(0.1, 0.2) [10 * sin(x) + 5 * cos(y)] = "
                  << idf->GetValue(std::vector<double>{0.1, 0.2}) << std::endl;
    }
};