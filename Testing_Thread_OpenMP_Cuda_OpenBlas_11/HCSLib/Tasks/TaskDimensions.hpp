#pragma once

#include <iostream>
#include <fstream>

#include "../CommonHelpers/PrintParams.hpp"

/// @brief Размерности задачи
struct TaskDimensions
{
    // Используется ли пространственная ось Ox
    bool is_used_X = false;
    // Используется ли пространственная ось Oy
    bool is_used_Y = false;
    // Используется ли пространственная ось Oz
    bool is_used_Z = false;
    // Используется ли ось времени Ot
    bool is_used_t = false;

    /// @brief Возвращает суммарное количество измерений задачи
    /// @return 
    unsigned GetDimensionsNumber() const
    {
        return (unsigned)is_used_X + (unsigned)is_used_Y + (unsigned)is_used_Z + (unsigned)is_used_t;
    }

    /// @brief Является ли задача стационарной
    /// @return true - стационарная, false - нестационарная
    bool IsStationaryProblem()
    {        
        return !is_used_t;
    }

    /// @brief Является ли задача нестационарной
    /// @return true - нестационарная, false - стационарная
    bool IsNonStationaryProblem()
    {
        return is_used_t;
    }

    /// @brief Является ли задача одномерной
    /// @return true - одномерная, false - неодномерная
    bool Is1DProblem()
    {
        if (GetDimensionsNumber() == 1)
            return true;
        
        return false;
    }

    void Print(PrintParams pp)
    {
        std::cout << pp.startMes;

        std::cout << "DimensionsNumber" << pp.splitterKeyValue << GetDimensionsNumber() << pp.splitter;
        std::cout << "is_used_X"   << pp.splitterKeyValue << is_used_X   << pp.splitter;
        std::cout << "is_used_Y"   << pp.splitterKeyValue << is_used_Y   << pp.splitter;
        std::cout << "is_used_Z"   << pp.splitterKeyValue << is_used_Z   << pp.splitter;
        std::cout << "is_used_t"   << pp.splitterKeyValue << is_used_t   << pp.splitter;
                    
        std::cout << pp.endMes;
        if(pp.isEndl)
            std::cout << std::endl;
    }

    friend std::ofstream& operator<<(std::ofstream& fout, const TaskDimensions& data)
    {
        fout << data.GetDimensionsNumber() << " "
             << data.is_used_X << " "
             << data.is_used_Y << " "
             << data.is_used_Z << " "
             << data.is_used_t;

        return fout;
    }

};

/*
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
*/