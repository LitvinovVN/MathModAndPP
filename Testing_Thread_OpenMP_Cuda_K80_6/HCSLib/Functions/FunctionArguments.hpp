#pragma once

#include <vector>
#include "../Functions/FunctionArgument.hpp"
#include "../CommonHelpers/PrintParams.hpp"

/// @brief Аргументы функции
class FunctionArguments
{
    std::vector<FunctionArgument> functionArguments;

public:
    void Add(FunctionArgument arg)
    {
        functionArguments.push_back(arg);
    }

    void Print(PrintParams pp)
    {
        for (size_t i = 0; i < functionArguments.size(); i++)
        {
            std::cout << i << ": " << functionArguments[i] << "; ";
        }
        std::cout << std::endl;
    }

};