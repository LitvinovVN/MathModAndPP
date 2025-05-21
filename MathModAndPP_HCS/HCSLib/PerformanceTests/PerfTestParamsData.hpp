#pragma once

#include "../CommonHelpers/DataTypes.hpp"

/// @brief Параметры варьирования диапазона данных
struct PerfTestParamsData
{
    // Типы данных
    DataTypes dataTypes;

    // Минимальное количество элементов в контейнере
    unsigned long long arrayLengthMin = 100000000ull;
    // Максимальное количество элементов в контейнере
    unsigned long long arrayLengthMax = 1000000000ull;
    // Шаг изменения количества элементов в контейнере
    unsigned long long arrayLengthStep = 100000000ull;

    PerfTestParamsData()
    {}

    PerfTestParamsData(DataTypes dataTypes,
        unsigned long long arrayLengthMin,
        unsigned long long arrayLengthMax,
        unsigned long long arrayLengthStep) :
            dataTypes(dataTypes),
            arrayLengthMin(arrayLengthMin),
            arrayLengthMax(arrayLengthMax),
            arrayLengthStep(arrayLengthStep)
    {}

    void Print(PrintParams pp = PrintParams{})
    {
        std::cout << pp.startMes;
        std::cout << "dataTypes" << pp.splitterKeyValue;
        dataTypes.Print();
        std::cout << pp.splitter;
        std::cout << "arrayLengthMin" << pp.splitterKeyValue << arrayLengthMin;
        std::cout << pp.splitter;
        std::cout << "arrayLengthMax" << pp.splitterKeyValue << arrayLengthMax;
        std::cout << pp.splitter;
        std::cout << "arrayLengthStep" << pp.splitterKeyValue << arrayLengthStep;
        std::cout << pp.endMes;

        if(pp.isEndl)
            std::cout << std::endl;
    }
};