#pragma once
#include <iostream>
#include <vector>
#include "ZMaskRepository.hpp"

/// @brief Маска для XY плоскости
/// @param data Массив значений
struct XYMask
{
    size_t nx = 0; // Размер по x
    size_t ny = 0; // Размер по y
    // Массив значений (0 - нет расчетных узлов вдоль Oz,
    // 1 - Oz частично заполнена,
    // 2 - Oz заполнена полностью)
    std::vector<std::vector<unsigned short>> data;

    XYMask(size_t nx, size_t ny)
        : nx(nx), ny(ny)
    {
        data.resize(nx, std::vector<unsigned short>(ny, 2));
    }

    bool IsPoinOutside(size_t i, size_t j) const
    {
        if (data[i][j] == 0)
            return true;
        else
            return false;
    }

    void Print() const
    {
        std::cout << "XYMask: " << std::endl;
        for (size_t i = 0; i < nx; i++)
        {
            std::cout << "i = " << i << ": ";
            for (size_t j = 0; j < ny; j++)
            {
                std::cout << data[i][j] << " ";
            }
            std::cout << std::endl;
        }        
    }
};