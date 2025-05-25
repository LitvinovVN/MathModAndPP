#pragma once

#include <vector>

#include "../CommonHelpers/_IncludeCommonHelpers.hpp"
#include "IGeometryLocation.hpp"

/// @brief Размещение объектов геометрии в двумерном пространстве
class GeometryComposition2D : public IGeometryComposition
{
    
public:        
    /// @brief Выводит в консоль сведения об объекте и его значение
    void Print() const override
    {
        std::cout << "GeometryComposition2D::Print()\n";
    }

    /// @brief Возвращает размерность объектов геометрии
    Dimension GetDimension() const override
    {
        return Dimension::D2;
    }

    /// @brief Возвращает единицу измерения, используемую для описания объекта геометрии
    /// @return MeasurementUnitEnum
    MeasurementUnitEnum GetMeasurementUnitEnum() const
    {
        return MeasurementUnitEnum::Meter;
    }
};
