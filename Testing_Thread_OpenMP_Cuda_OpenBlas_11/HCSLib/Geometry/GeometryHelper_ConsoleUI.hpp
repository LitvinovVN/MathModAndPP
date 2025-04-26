#pragma once

#include <iostream>

#include "_IncludeGeometry.hpp"

struct GeometryHelper_ConsoleUI
{
    static void Geometry2DRectangle_Console_UI()
    {
        std::cout << "Geometry2DRectangle_Console_UI()\n";
        IGeometry* g2dRectangle = new G2DRectangle(2, 1);
        g2dRectangle->Print();
    }
};