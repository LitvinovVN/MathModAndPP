#pragma once

#include <iostream>
#include <vector>
#include "../CommonHelpers/_IncludeCommonHelpers.hpp"

/// @brief Функция правой части дифф. уравнения (заданная аналитически)
class DiffEqFunc2D : public IDiffEqFunction
{
    /// @brief Указатель на функцию
    double (*f)(double, double);
public:

    DiffEqFunc2D(double (*f)(double, double))
        : f(f)
    {

    }

    /// @brief Возвращает размерность объекта функции
    Dimension GetDimension() const override
    {
        return Dimension::D2;
    }

    
    double GetValue(double x, double y) const
    {
        if(!f)
            throw std::runtime_error("f not allowed!");
        return f(x, y);
    }

    /// @brief Возвращает значение функции в точке
    double GetValue(std::vector<double> coordinates) const override
    {
        if(coordinates.size()<2)
            return 0;
        return GetValue(coordinates[0], coordinates[1]);
    }

    void Print() const override
    {
        std::cout << "function address: " << f << std::endl;
        std::cout << "dimension: " << GetDimension() << std::endl;
    }
};