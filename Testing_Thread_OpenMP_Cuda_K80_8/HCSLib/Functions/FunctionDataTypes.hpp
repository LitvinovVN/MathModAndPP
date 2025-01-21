#pragma once

#include <vector>
#include "../Functions/FunctionDataType.hpp"
#include "../CommonHelpers/PrintParams.hpp"

/// @brief Список типов аргументов функции
class FunctionDataTypes
{
    std::vector<FunctionDataType> functionDataTypes;

public:
    void Add(FunctionDataType dataType)
    {
        functionDataTypes.push_back(dataType);
    }

    void Print(PrintParams pp = PrintParams{}) const
    {
        for (size_t i = 0; i < functionDataTypes.size(); i++)
        {
            std::cout << i << ": " << functionDataTypes[i] << "; ";
        }
        std::cout << std::endl;
    }

    unsigned Count() const
    {
        return functionDataTypes.size();
    }

    FunctionDataType operator[](unsigned index) const
    {
        if(index >= Count())
        {
            std::cout << "\nError! Index out of range\n";
            throw std::runtime_error("Error in FunctionDataTypes::operator[]. Out of range!");
        }

        return functionDataTypes[index];
    }

};

std::ostream& operator<<(std::ostream& os, FunctionDataTypes fdts)
{
    std::cout << "(";
    for(unsigned i = 0; i < fdts.Count(); i++)
    {
        std::cout << fdts[i];
        if (i < fdts.Count() - 1)
            std::cout << ", ";
    }
    std::cout << ")";

    return os;
}