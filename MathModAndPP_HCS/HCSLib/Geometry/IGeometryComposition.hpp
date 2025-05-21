#pragma once

#include <vector>

#include "../CommonHelpers/_IncludeCommonHelpers.hpp"
#include "IGeometryLocation.hpp"

/// @brief Абстрактный класс, моделирующий размещение объектов геометрии в пространстве
class IGeometryComposition
{
    std::vector<IGeometryLocation> elements;
public:        
    /// @brief Выводит в консоль сведения об объекте и его значение
    virtual void Print() const = 0;

    /// @brief Возвращает размерность объекта геометрии
    virtual Dimension GetDimension() const = 0;

    /// @brief Возвращает единицу измерения, используемую для описания объекта геометрии
    /// @return MeasurementUnitEnum
    MeasurementUnitEnum GetMeasurementUnitEnum() const
    {
        return MeasurementUnitEnum::Meter;
    }
};
