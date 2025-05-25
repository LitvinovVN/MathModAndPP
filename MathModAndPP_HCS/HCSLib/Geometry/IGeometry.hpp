#pragma once

/// @brief Абстрактный класс, моделирующий геометрию расчетной области
class IGeometry
{
public:
    /// @brief Виртуальный деструктор
    virtual ~IGeometry() = default;

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
