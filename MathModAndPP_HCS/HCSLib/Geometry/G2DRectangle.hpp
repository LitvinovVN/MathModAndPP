#pragma once

/// @brief Прямоугольник
class G2DRectangle : public IGeometry
{
    /// @brief Длина
    double Lx;
    /// @brief Ширина
    double Ly;

public:
    /// @brief 
    /// @param Lx 
    /// @param Ly 
    G2DRectangle(double Lx, double Ly)
        : Lx(Lx), Ly(Ly)
    {}

    ~G2DRectangle()
    {
        //std::cout << "G2DRectangle::~G2DRectangle()\n";
    }

    /// @brief Возвращает размерность объекта геометрии
    Dimension GetDimension() const override
    {
        return Dimension::D2;
    }

    /// @brief Выводит в консоль сведения об объекте
    void Print() const override
    {
        std::cout << "ScalarRam object description:" << std::endl;
        std::cout << "type name: " << typeid(this).name() << std::endl;
        std::cout << "address: "   << this << std::endl;
        std::cout << "dimension: " << GetDimension()  << std::endl;
        std::cout << "Lx: " << Lx  << " " << GetMeasurementUnitEnum() << std::endl;
        std::cout << "Ly: " << Ly  << " " << GetMeasurementUnitEnum() << std::endl;
    }

};