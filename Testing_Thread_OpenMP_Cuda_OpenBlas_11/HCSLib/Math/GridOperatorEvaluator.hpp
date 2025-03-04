#pragma once

#include "GridEvaluableObject.hpp"

/// @brief Вычислитель
/// @tparam GO 
/// @tparam EO 
template<class GO, class EO>
struct GridOperatorEvaluator : GridEvaluableObject< GridOperatorEvaluator<GO, EO> >
{
    typedef GridOperatorEvaluator type;
    typedef GridEvaluableObject<type> base_type;
    typedef typename GO::template get_value_type<typename EO::value_type>::type value_type;
    typedef GridOperator<GO, typename GO::proxy_type> op_type;
    typedef GridEvaluableObject<EO, typename EO::proxy_type> eobj_type;
    
    GridOperatorEvaluator(const op_type& op, const eobj_type& eobj)
      : op_proxy(op.get_proxy()),
        eobj_proxy(eobj.get_proxy())
    {}

    template<class GC>
    value_type operator()(size_t i, size_t j, size_t k,
    const GridContext<GC>& context) const
    {
        return op_proxy(i, j, k, eobj_proxy, context);
    }

    value_type operator()(size_t i, size_t j, size_t k) const
    {
        return op_proxy(i, j, k, eobj_proxy);
    }

private:
    const typename GO::proxy_type op_proxy;
    const typename EO::proxy_type eobj_proxy;
};