#pragma once

#include <iostream>
#include "../CommonHelpers/PrintParams.hpp"

/// @brief Индекс и размер блока, локальный индекс элемента в блоке
struct ArrayBlockIndexes
{
    // Индекс блока
    unsigned blockIndex{};
    // Размер блока
    unsigned long long blockLength{};
    // Локальный индекс элемента в блоке
    unsigned long long localIndex{};

    void Print(PrintParams pp = PrintParams{})
    {        
        pp.PrintStartMessage();
        pp.PrintKeyValue("blockIndex", blockIndex);
        pp.PrintSplitter();
        pp.PrintKeyValue("blockLength", blockLength);
        pp.PrintSplitter();
        pp.PrintKeyValue("localIndex", localIndex);
        pp.PrintEndMessage();
        pp.PrintIsEndl();
        
    }

    /// @brief Возвращает флаг инициализации объекта
    /// @return Успех, если размер блока > 0
    bool IsInitialized()
    {
        return (bool)blockLength;
    }
};