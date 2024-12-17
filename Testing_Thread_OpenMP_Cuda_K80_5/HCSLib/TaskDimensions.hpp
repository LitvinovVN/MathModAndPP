#pragma once

#include <iostream>
#include <fstream>

/// @brief Размерность задачи
struct TaskDimensions
{
    unsigned dim = 1;// 1 - 1D, 2 - 2D, 3 - 3D, 
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

    void Print()
    {
        std::cout   << "dim:    " << dim    << "; "
                    << "size_x: " << x << "; "
                    << "size_y: " << y << "; "
                    << "size_z: " << z << "; "
                    << "size_t: " << t << "; "
                    << std::endl;
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