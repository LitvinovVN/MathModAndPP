#pragma once

#include <vector>
#include "../Functions/FunctionArgument.hpp"
#include "../CommonHelpers/PrintParams.hpp"

/// @brief Аргументы функции
class FunctionArguments
{
    std::vector<FunctionArgument> functionArguments;

public:

    FunctionDataTypes GetFunctionArgumentsDataTypes() const
    {
        FunctionDataTypes argDataTypes;
        for (size_t i = 0; i < functionArguments.size(); i++)
        {
            argDataTypes.Add(functionArguments[i].dataType);
        }
        return argDataTypes;
    }

    void Add(FunctionArgument arg)
    {
        functionArguments.push_back(arg);
    }

    FunctionArgument Get(unsigned index) const
    {
        return functionArguments[index];
    }

    template<typename T>
    T GetArgumentValue(unsigned index)
    {
        FunctionArgument arg = Get(index);
        T argValue = arg.GetValue<T>();
        return argValue;
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