#pragma once

#include <vector>
#include "DataType.hpp"
#include "PrintParams.hpp"

/// @brief Список типов данных
class DataTypes
{
    std::vector<DataType> dataTypes;

public:
    void Add(DataType dataType)
    {
        dataTypes.push_back(dataType);
    }

    void Print(PrintParams pp = PrintParams{}) const
    {
        std::cout << pp.startMes;
        for (size_t i = 0; i < dataTypes.size(); i++)
        {
            std::cout << i << pp.splitterKeyValue << dataTypes[i];
            if(i<dataTypes.size()-1)
                std::cout << "; ";
        }
        std::cout << pp.endMes;
        if(pp.isEndl)
            std::cout << std::endl;
    }

    unsigned Count() const
    {
        return dataTypes.size();
    }

    DataType operator[](unsigned index) const
    {
        if(index >= Count())
        {
            std::cout << "\nError! Index out of range\n";
            throw std::runtime_error("Error in FunctionDataTypes::operator[]. Out of range!");
        }

        return dataTypes[index];
    }

};

std::ostream& operator<<(std::ostream& os, DataTypes dts)
{
    std::cout << "(";
    for(unsigned i = 0; i < dts.Count(); i++)
    {
        std::cout << dts[i];
        if (i < dts.Count() - 1)
            std::cout << ", ";
    }
    std::cout << ")";

    return os;
}