#pragma once

#include <iostream>
#include <vector>
#include "ArrayBlockIndexes.hpp"

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

    /// @brief Возвращает объект, содержащий индексы блока, размер блока и локальный индекс
    /// @param globalIndex Глобальный индекс элемента
    /// @return ArrayBlockIndexes
    ArrayBlockIndexes GetArrayBlockIndexes(unsigned long long globalIndex) const
    {
        ArrayBlockIndexes arrayBlockIndexes;

        for (size_t bi = 0; bi < indexMap.size(); bi++)
        {
            auto indStart = indexMap[bi][0];
            auto indEnd   = indexMap[bi][1];
            if (globalIndex < indStart || globalIndex > indEnd)
                continue;
            
            arrayBlockIndexes.blockIndex = bi;
            arrayBlockIndexes.blockLength = indEnd - indStart + 1;
            arrayBlockIndexes.localIndex = globalIndex - indStart;
            break;
        }
        
        return arrayBlockIndexes;
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