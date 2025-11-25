#pragma once
#include <iostream>

struct Coord3D
{
    double x = 0;
    double y = 0;
    double z = 0;

    Coord3D(double x, double y, double z)
        : x(x), y(y), z(z)
    {    }

    void Print() const
    {
        std::cout << "("<< x << ", " << y << ", " << z << ")";
    }

};