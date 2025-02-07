#pragma once

#include <iostream>
#include "MathHelper.hpp"

template<class E>
void f(const Expression<E> &expr0)
{
    const E &expr = expr0.Self();
    std::cout << "expr(3): " << expr(3) << std::endl;
    std::cout << "expr(1.5): " << expr(1.5) << std::endl;
}

/// @brief Вспомогательный класс для работы с модулем Math
struct MathHelper_ConsoleUI
{
    /// @brief Работа с классом MathObject
    static void MathObject_ConsoleUI()
    {
        std::cout << "--- void MathObject_ConsoleUI() ---" << std::endl;
        
        /*struct Scalar : MathObject<Scalar>
        {
            int value = 10;
        };

        Scalar a;
        std::cout << "Scalar a; a.value: " << a.value << std::endl;
        std::cout << "a.Self().value: " << a.Self().value << std::endl;
        std::cout << "a.GetProxy().value: " << a.GetProxy().value << std::endl;*/

        

        Variable x;

        std::cout << "f(x) = x" << std::endl;
        f(x);

        std::cout << "f(x) = x+1.5" << std::endl;
        auto expr1 = x + 1.5;
        f(expr1);
        double res = expr1(10);
        std::cout << "double res = expr1(10): " << res << std::endl;

        f(sin(x * x + M_PI));

        auto expr2 = 5 * cos(-x * (x + 1));
        f(expr2);

        //Variable y;
        //auto expr_x_y = 2*x - y/3;
        //std::cout<< "f(x,y) = 2*x - y/3; f(10, 30) = " << expr_x_y(10, 30) << std::endl;//err

    }

};

