#pragma once

#include <iostream>
#include <vector>

class ArraysIndexMap
{
    std::vector<std::vector<unsigned long long>> indexMap;
public:
    /// @brief Добавляет строку индексов
    /// @param indStart Индекс первого элемента
    /// @param indEnd Индекс последнего элемента
    void AddIndexes(unsigned long long indStart, unsigned long long indEnd)
    {
        std::vector<unsigned long long> row;
        row.push_back(indStart);
        row.push_back(indEnd);

        indexMap.push_back(row);
    }

    void Print()
    {
        std::cout << "ArraysIndexMap::Print()" << std::endl;
        for (size_t i = 0; i < indexMap.size(); i++)
        {
            auto& row = indexMap[i];
            std::cout << row[0] << " " << row[1] << std::endl;            
        }
        std::cout << "-----------------------" << std::endl;
    }
};