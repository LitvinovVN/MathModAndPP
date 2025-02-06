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

        class TestA{};
        class TestB : TestA{};
        std::cout << "sizeof(TestA{}): " << sizeof(TestA{}) << std::endl;
        std::cout << "sizeof(TestB{}): " << sizeof(TestB{}) << std::endl;

        MathObject<TestA, TestB> mo;
        std::cout << "sizeof(MathObject<TestA, TestB>): " << sizeof(mo) << std::endl;

        //mo.Self();
        //mo.GetProxy();
    }

};

