#pragma once

#include <vector>
#include "../CommonHelpers/_IncludeCommonHelpers.hpp"

// Интерфейс функции правой части
class IDiffEqFunction
{
public:
    /// @brief Возвращает значение функции в точке
    virtual double GetValue(std::vector<double>) const = 0;

    /// @brief Возвращает размерность объекта функции
    virtual Dimension GetDimension() const = 0;

    /// @brief Выводит в консоль сведения об объекте
    virtual void Print() const = 0;
};