#pragma once

#include <vector>

#include "../CommonHelpers/_IncludeCommonHelpers.hpp"
#include "IGeometryLocation.hpp"

/// @brief Абстрактный класс, моделирующий размещение объектов геометрии в пространстве
class IGeometryComposition
{
    std::vector<IGeometryLocation*> elements;
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

    /// @brief Добавляет объект геометрии в расчетную область по заданнй координате
    /// @param geometry 
    /// @param x 
    /// @param y 
    void Add(IGeometry* geometry, double x, double y)
    {
        IGeometryLocation* geometryLocation = new GeometryLocation2D(geometry, x, y);
        elements.push_back(geometryLocation);
    }

    /// @brief Вывод сведений об объект в консоль
    void Print()
    {
        std::cout << "IGeometryComposition address: " << this << std::endl;
        std::cout << "Geometry elements count: " << elements.size() << std::endl;

        for(auto i = 0ull; i < elements.size(); i++)
        {
            elements[i]->Print();
        }
    }
};
