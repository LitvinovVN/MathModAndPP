#pragma once
#include <iostream>

// Параметры сетки
struct Grid3DParams
{
    size_t nx, ny, nz;
    double hx, hy, hz;

    Grid3DParams(size_t nx, size_t ny, size_t nz, double hx, double hy, double hz)
        : nx(nx), ny(ny), nz(nz), hx(hx), hy(hy), hz(hz)
    {}
};