#pragma once

#include "MathObject.hpp"

// Предварительное определение GridOperatorEvaluator
template<class GO, class EO>
struct GridOperatorEvaluator;



template<class GO, class Proxy>
struct GridOperator : MathObject<GO, Proxy>
{
    template<typename T>
    struct GetValueType
    {
        typedef T type;
    };

    template<class EO>
    GridOperatorEvaluator<GO, EO>
    operator()(const GridEvaluableObject<EO, typename EO::proxy_type>& eobj) 
    {
        return GridOperatorEvaluator<GO, EO>(*this, eobj);
    }
};

#define REIMPLEMENT_GRID_EVAL_OPERATOR() \
template<class EO> \
GridOperatorEvaluator<type, EO> \
operator()(const GridEvaluableObject<EO, typename EO::proxy_type>& eobj) const \
{ \
    return base_type::operator()(eobj); \
}
