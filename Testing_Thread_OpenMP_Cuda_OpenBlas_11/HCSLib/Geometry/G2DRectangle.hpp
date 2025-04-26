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

    /// @brief Выводит в консоль сведения об объекте
    void Print() const override
    {
        std::cout << "ScalarRam object description:" << std::endl;
        std::cout << "type name: " << typeid(this).name() << std::endl;
        std::cout << "address: " << this << std::endl;
        std::cout << "dimension: " << dimension  << std::endl;
        std::cout << "Lx: " << Lx  << " " << GetMeasurementUnitEnum() << std::endl;
        std::cout << "Ly: " << Ly  << " " << GetMeasurementUnitEnum() << std::endl;
    }

};