#pragma once

#include <iostream>
#include "MathHelper.hpp"

/// @brief Вспомогательный класс для работы с модулем Math
class MathHelper_ConsoleUI
{
public:
    /// @brief Работа с классом MathObject
    static void MathObject_ConsoleUI()
    {
        std::cout << "--- void MathObject_ConsoleUI() ---" << std::endl;
        
        struct Scalar : MathObject<Scalar>
        {
            int value = 10;
        };

        Scalar a;
        std::cout << "Scalar a; a.value: " << a.value << std::endl;
        std::cout << "a.Self().value: " << a.Self().value << std::endl;
        std::cout << "a.GetProxy().value: " << a.GetProxy().value << std::endl;
    }

};

