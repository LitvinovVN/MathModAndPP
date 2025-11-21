// g++ main.cpp -o app -O3 -fopenmp
// ./app
#include <iostream>
#include <chrono>
#include <thread>

#include "PhysField3D.hpp"

using namespace std::chrono;
using namespace std::chrono_literals;

inline void InitByGlobalIndex(IPhysField3D *physField3D)
{
    size_t Nx = physField3D->GetNx();
    size_t Ny = physField3D->GetNy();
    size_t Nz = physField3D->GetNz();

#pragma omp parallel for
    // for (size_t k = 0; k < Nz; k++) // Медленно
    for (size_t i = 0; i < Nx; i++)
    {
        //size_t offsetNyNz = i * Nz * Ny;
        for (size_t j = 0; j < Ny; j++)
        {
            //size_t offsetNyNzNz = offsetNyNz + j * Nz;
            for (size_t k = 0; k < Nz; k++) // Быстро
            {
                double val = k + j * Nz + i * Nz * Ny;
                //double val = k + offsetNyNzNz;
                physField3D->SetValue(i, j, k, val);
                // std::cout << physField3D->GetValue(i, j, k) << " ";
            }
        }
    }
    // std::cout << std::endl;
}

void TestInitArray(size_t N)
{
    std::cout << "TestInitArray(" << TestInitArray << "): ";
    double *arr = new double[N];
    auto start = high_resolution_clock::now();
#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        arr[i] = i;
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() << " mks\n"
              << std::endl;
    delete[] arr;
}

void CalcGradCx(IPhysField3D *C, IPhysField3D *gradCx)
{
    auto start = high_resolution_clock::now();
    //std::this_thread::sleep_for(1000ms);
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Duration of CalcGradCx, mks: " << duration.count() << std::endl;
}

int main()
{
    std::cout << "Pole 3D\n";
    size_t Nx = 1000;
    size_t Ny = 1000;
    size_t Nz = 1000;
    size_t N = Nx * Ny * Nz;

    TestInitArray(N);

    IPhysField3D *C = (IPhysField3D *)new PhysField3D(Nx, Ny, Nz);

    auto start = high_resolution_clock::now();
    InitByGlobalIndex(C);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Duration of InitByGlobalIndex, mks: " << duration.count() << std::endl;

    C->PrintValue(1, 2, 3);
    C->PrintDataArray(0, N < 5 ? N : 5);

    double newVal = -C->GetValue(2, 3, 4);
    C->SetValue(2, 3, 4, newVal);
    C->PrintValue(2, 3, 4);

    IPhysField3D *gradCx = (IPhysField3D*)new PhysField3D(Nx, Ny, Nz);
    CalcGradCx(C, gradCx);
    delete[] gradCx;


    delete[] C;    
}