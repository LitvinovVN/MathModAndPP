#pragma once

#include <iostream>
#include <fstream>

#include "PrintParams.hpp"

/// @brief Размерности задачи
struct TaskDimensions
{
    unsigned dim = 1;// 1 - 1D, 2 - 2D, 3 - 3D, 4 - 3D+t
    size_t x = 1;// Количество элементов по x
    size_t y = 1;// Количество элементов по y
    size_t z = 1;// Количество элементов по z
    size_t t = 1;// Количество элементов по t

    /// @brief Возвращает суммарный размер задачи
    /// @return 
    size_t GetFullSize()
    {
        return x * y * z * t;
    }

    /// @brief Является ли задача стационарной
    /// @return true - стационарная, false - нестационарная
    bool IsStationaryProblem()
    {
        if (t > 1)
            return false;
        
        return true;
    }

    /// @brief Является ли задача нестационарной
    /// @return true - нестационарная, false - стационарная
    bool IsNonStationaryProblem()
    {
        return !IsStationaryProblem();
    }

    /// @brief Является ли задача стационарной
    /// @return true - стационарная, false - нестационарная
    bool Is1DProblem()
    {
        if (y == 1 && z == 1)
            return true;
        
        return false;
    }

    void Print(PrintParams pp)
    {
        std::cout << pp.startMes;

        std::cout << "dim" << pp.splitterKeyValue << dim << pp.splitter;
        std::cout << "x"   << pp.splitterKeyValue << x   << pp.splitter;
        std::cout << "y"   << pp.splitterKeyValue << y   << pp.splitter;
        std::cout << "z"   << pp.splitterKeyValue << z   << pp.splitter;
        std::cout << "t"   << pp.splitterKeyValue << t   << pp.splitter;
                    
        std::cout << pp.endMes;
        if(pp.isEndl)
            std::cout << std::endl;
    }

    friend std::ofstream& operator<<(std::ofstream& fout, const TaskDimensions& data)
    {
        fout << data.dim << " "
             << data.x << " "
             << data.y << " "
             << data.z << " "
             << data.t;

        return fout;
    }

};