#pragma once

#include "Expression.hpp"

/// @brief Выражение с двумя операндами
/// @tparam E1 
/// @tparam OP +, -, *, /
/// @tparam E2 
template<class E1, class OP, class E2>
struct BinaryExpression :
    Expression<BinaryExpression<E1, OP, E2> >
{
    BinaryExpression(const Expression<E1> &expr1,
        const OP &op,
        const Expression<E2> &expr2)
          : expr1(expr1.Self()),
            op(op),
            expr2(expr2.Self())
    {}

    double operator()(double x) const
    {
        return op(expr1(x), expr2(x));
    }
private:
    const E1 expr1;
    const OP op;
    const E2 expr2;
};

#define DEFINE_BIN_OP(oper, OP) \
 \
template<class E1, class E2> \
BinaryExpression<E1, std::OP<double>, E2> operator oper \
    (const Expression<E1> &expr1, const Expression<E2> &expr2) \
{ \
        return BinaryExpression<E1, std::OP<double>, E2> \
            (expr1, std::OP<double>(), expr2); \
} \
 \
template<class E> \
BinaryExpression<E, std::OP<double>, Constant> operator oper \
    (const Expression<E> &expr, double value) \
{ \
    return BinaryExpression<E, std::OP<double>, Constant> \
        (expr, std::OP<double>(), Constant(value)); \
} \
 \
template<class E> \
BinaryExpression<Constant, std::OP<double>, E> operator oper \
    (double value, const Expression<E> &expr)\
{ \
    return BinaryExpression<Constant, std::OP<double>, E> \
        (Constant(value), std::OP<double>(), expr); \
}

DEFINE_BIN_OP(+, plus)
DEFINE_BIN_OP(-, minus)
DEFINE_BIN_OP(*, multiplies)
DEFINE_BIN_OP(/, divides)