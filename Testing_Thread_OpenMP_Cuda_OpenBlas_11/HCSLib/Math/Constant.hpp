#pragma once

#include "Expression.hpp"

/// @brief Константа (в выражении)
struct Constant : Expression<Constant>
{
    Constant(double value) : value(value){}

    double operator()(double x) const
    {
        return value; // Возвращаемое значение не зависит от значения переменной.
    }
    
    private:
    double value;
};