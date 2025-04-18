#pragma once

#include "Expression.hpp"

/// @brief Отрицание (в выражении)
/// @tparam E 
template<class E>
struct Negate : Expression<Negate<E> >
{
    Negate(const Expression<E> &expr)
        : expr(expr.Self()){}
    
    double operator()(double x) const
    {
        return -expr(x);
    }

private:
    const E expr;
};

template<class E>
Negate<E> operator-(const Expression<E> &expr)
{
    return Negate<E>(expr);
}
