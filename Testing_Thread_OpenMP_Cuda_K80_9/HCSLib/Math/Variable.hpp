#pragma once

#include "Expression.hpp"

/// @brief Переменная (в выражении)
struct Variable : Expression<Variable>
{
    double operator()(double x) const
    {
        return x;
    }
};