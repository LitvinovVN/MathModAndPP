#pragma once

#include "IGeometry.hpp"
#include "Location2D.hpp"
#include "IGeometryLocation.hpp"

class GeometryLocation2D : public IGeometryLocation
{
    Location2D location;
public:
    GeometryLocation2D(IGeometry* geometry, double x, double y)
    {
        this->geometry = geometry;
        location.x = x;
        location.y = y;
    }

    ~GeometryLocation2D()
    {
        delete this->geometry;
    }

    ILocation* GetLocation() override
    {
        return &location;
    }
};