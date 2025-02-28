#pragma once

#include "MathObject.hpp"

/// @brief Объект, вычисляемый на сетке. Маркерный класс.
/// @tparam EO Тип вычисляемого объекта
/// @tparam Proxy Прокси-объект (облегченный)
template<class EO, class Proxy = EO>
struct GridEvaluableObject : MathObject<EO, Proxy>{};