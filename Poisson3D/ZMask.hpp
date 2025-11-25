#pragma once

#include <iostream>
#include <vector>

/// @brief Маска для оси Oz (массив пар: индекс сегмента, длина сегмента)
struct ZMask
{
    std::vector<std::pair<size_t, size_t>> mask;

    void Print() const
    {
        for (const auto& elem : mask)
        {
            std::cout << elem.first << elem.second << std::endl;
        }
    }
};