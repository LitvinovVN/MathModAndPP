#pragma once

/// @brief Координаты расположения объекта геометрии в пространстве
class Location2D : public ILocation
{
public:
    double x{0};
    double y{0};

    Location2D()
    {}

    Location2D(double x, double y)
        : x(x), y(y)
    {}

    void Print() const override
    {
        std::cout << "Location2D: ";
        std::cout << this << "; ";
        std::cout << "x = " << x << "; ";
        std::cout << "y = " << y << ".";
        std::cout << std::endl;
    }
};