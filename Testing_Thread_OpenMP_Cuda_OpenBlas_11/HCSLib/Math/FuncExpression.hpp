#pragma once

#include "Expression.hpp"

/// @brief Функция (в выражении)
/// @tparam E 
template<class E>
struct FuncExpression : Expression<FuncExpression<E>>
{
    typedef double (*func_t)(double);

    FuncExpression(const Expression<E> &expr, func_t func) :
        expr(expr.Self()), func(func)
    {}

    double operator()(double x) const
    {
        return func(expr(x));
    }

private:
    const E expr;
    func_t func;
};

#define DEFINE_FUNC(func) \
\
template<class E> \
FuncExpression<E> func(const Expression<E> &expr) \
{ \
    return FuncExpression<E>(expr, std::func); \
}

DEFINE_FUNC(sin)
DEFINE_FUNC(cos)
DEFINE_FUNC(tan)
DEFINE_FUNC(atan)
DEFINE_FUNC(exp)
DEFINE_FUNC(log)
DEFINE_FUNC(sqrt)