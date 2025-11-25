// g++ main.cpp -o app -O3 -fopenmp
// ./app > res.txt
#include "Includes.hpp"

int main()
{
    std::cout << "--- Poisson Equation 3D ---" << std::endl;
    std::cout << "omp_get_max_threads(): " << omp_get_max_threads() << std::endl;

    size_t nx = 3;
    size_t ny = 4;
    size_t nz = 5;
    double hx = 0.1;
    double hy = 0.2;
    double hz = 0.3;    
    Coord3D coord3D(10,20,30);
    Grid3DParams grid3DParams(nx, ny, nz, hx, hy, hz);
    XYMask xyMask(nx, ny);
    xyMask.data[0][0] = 0;
    xyMask.data[0][1] = 0;
    xyMask.data[0][2] = 1;

    ScalarField3D c(coord3D, grid3DParams, xyMask);
    c.PrintInfo();
    c.xyMask.Print();
    c.zMaskRepository.Print();

    // Инициализируем поле функцией
    auto func = [](double x, double y, double z) { return x + 2*y + 3*z; };
    ScalarField3DAlg::Init(c, func);
    c.PrintData();

    std::cout << "--- Poisson Equation 3D End ---" << std::endl;
}