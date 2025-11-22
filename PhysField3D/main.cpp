// g++ main.cpp -o app -O3 -fopenmp
// ./app
#include <iostream>
#include <chrono>
#include <thread>

#include "PhysField3D.hpp"
#include "PhysField3DZ.hpp"

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
                double val = k;
                //double val = k + j * Nz + i * Nz * Ny;
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
    size_t Nx = C->GetNx();
    size_t Ny = C->GetNy();
    size_t Nz = C->GetNz();
    double hx = C->GetHx();
    double hy = C->GetHy();
    double hz = C->GetHz();
    //std::this_thread::sleep_for(1000ms);
#pragma omp parallel for
    for (size_t k = 0; k < Nz; k++)
    {
        for (size_t j = 0; j < Ny; j++)
        {
            size_t i = 0;
            double C0 = C->GetValue(i,j,k);
            double Cp1 = C->GetValue(i+1,j,k);
            double res = (Cp1-C0)/hx;
            gradCx->SetValue(i,j,k,res);

            for (i = 1; i < Nx-1; i++)
            {
                double Cm1 = C->GetValue(i-1,j,k);
                double Cp1 = C->GetValue(i+1,j,k);
                double res = (Cp1-Cm1)/(2*hx);
                gradCx->SetValue(i,j,k,res);
            }

            double Cm1 = C->GetValue(i-1,j,k);
            C0 = C->GetValue(i,j,k);
            res = (C0-Cm1)/(hx);
            gradCx->SetValue(i,j,k,res);
        }
        
    }
    

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Duration of CalcGradCx, mks: " << duration.count() << std::endl;
}


void CalcGradCy(IPhysField3D *C, IPhysField3D *gradCy)
{
    auto start = high_resolution_clock::now();
    size_t Nx = C->GetNx();
    size_t Ny = C->GetNy();
    size_t Nz = C->GetNz();    
    double hy = C->GetHy();
    //std::this_thread::sleep_for(1000ms);
#pragma omp parallel for
    for (size_t i = 0; i < Nx; i++)
    {
        for (size_t k = 0; k < Nz; k++)
        {
            size_t j = 0;
            double C0 = C->GetValue(i,j,k);
            double Cp1 = C->GetValue(i,j+1,k);
            double res = (Cp1-C0)/hy;
            gradCy->SetValue(i,j,k,res);

            for (j = 1; j < Ny-1; j++)
            {
                double Cm1 = C->GetValue(i,j-1,k);
                double Cp1 = C->GetValue(i,j+1,k);
                double res = (Cp1-Cm1)/(2*hy);
                gradCy->SetValue(i,j,k,res);
            }

            double Cm1 = C->GetValue(i,j-1,k);
            C0 = C->GetValue(i,j,k);
            res = (C0-Cm1)/(hy);
            gradCy->SetValue(i,j,k,res);
        }
        
    }
    

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Duration of CalcGradCx, mks: " << duration.count() << std::endl;
}


void CalcGradCz(IPhysField3D *C, IPhysField3D *gradCz)
{
    auto start = high_resolution_clock::now();
    size_t Nx = C->GetNx();
    size_t Ny = C->GetNy();
    size_t Nz = C->GetNz();    
    double hz = C->GetHz();
    //std::this_thread::sleep_for(1000ms);
#pragma omp parallel for
    for (size_t i = 0; i < Nx; i++)
    {
        for (size_t j = 0; j < Ny; j++)
        {
            size_t k = 0;
            double C0 = C->GetValue(i,j,k);
            double Cp1 = C->GetValue(i,j,k+1);
            double res = (Cp1-C0)/hz;
            gradCz->SetValue(i,j,k,res);

            for (k = 1; k < Nz-1; k++)
            {
                double Cm1 = C->GetValue(i,j,k-1);
                double Cp1 = C->GetValue(i,j,k+1);
                double res = (Cp1-Cm1)/(2*hz);
                gradCz->SetValue(i,j,k,res);
            }

            double Cm1 = C->GetValue(i,j,k-1);
            C0 = C->GetValue(i,j,k);
            res = (C0-Cm1)/(hz);
            gradCz->SetValue(i,j,k,res);
        }
        
    }
    

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Duration of CalcGradCx, mks: " << duration.count() << std::endl;
}

void TestGradPhysField3D(size_t Nx,size_t Ny,size_t Nz,size_t N,double Hx,double Hy,double Hz)
{
    IPhysField3D *C = (IPhysField3D *)new PhysField3D(Nx, Ny, Nz, Hx, Hy, Hz);

    auto start = high_resolution_clock::now();
    InitByGlobalIndex(C);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Duration of InitByGlobalIndex, mks: " << duration.count() << std::endl;

    C->PrintValue(1, 2, 3);
    C->PrintDataArray(0, N < 5 ? N : 5);
    //C->PrintDataArray();

    double newVal = -C->GetValue(2, 3, 4);
    C->SetValue(2, 3, 4, newVal);
    C->PrintValue(2, 3, 4);

    std::cout << "--- CalcGradCx ---\n";
    IPhysField3D *gradCx = (IPhysField3D*)new PhysField3D(Nx, Ny, Nz, Hx, Hy, Hz);
    CalcGradCx(C, gradCx);
    gradCx->PrintDataArray(0, N < 5 ? N : 5);
    //gradCx->PrintDataArray();
    delete gradCx;

    std::cout << "--- CalcGradCy ---\n";
    IPhysField3D *gradCy = (IPhysField3D*)new PhysField3D(Nx, Ny, Nz, Hx, Hy, Hz);
    CalcGradCy(C, gradCy);
    gradCy->PrintDataArray(0, N < 5 ? N : 5);
    //gradCy->PrintDataArray();
    delete gradCy;

    std::cout << "--- CalcGradCz ---\n";
    IPhysField3D *gradCz = (IPhysField3D*)new PhysField3D(Nx, Ny, Nz, Hx, Hy, Hz);
    CalcGradCz(C, gradCz);
    gradCz->PrintDataArray(0, N < 5 ? N : 5);
    //gradCz->PrintDataArray();
    delete gradCz;

    delete C;
}

void TestGradPhysField3DZ(size_t Nbx,size_t Nby,size_t Nbz,
    size_t Nnbx,size_t Nnby,size_t Nnbz,
    size_t Nx,size_t Ny,size_t Nz,size_t N,double Hx,double Hy,double Hz)
{
    std::cout << "--- TestGradPhysField3DZ START ---\n";
    IPhysField3D *C = (IPhysField3D *)new PhysField3DZ(Nbx, Nby, Nbz,
        Nnbx, Nnby, Nnbz, Hx, Hy, Hz, "C");
    C->Print();

    auto start = high_resolution_clock::now();
    InitByGlobalIndex(C);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Duration of InitByGlobalIndex, mks: " << duration.count() << std::endl;

    C->PrintValue(1, 2, 3);
    
    if(N < 200)
        C->PrintDataArray();
    else
        C->PrintDataArray(0, 10);

    double newVal = -C->GetValue(2, 3, 4);
    C->SetValue(2, 3, 4, newVal);
    C->PrintValue(2, 3, 4);

    std::cout << "--- CalcGradCx ---\n";
    IPhysField3D *gradCx = (IPhysField3D*)new PhysField3D(Nx, Ny, Nz, Hx, Hy, Hz);
    CalcGradCx(C, gradCx);
    gradCx->PrintDataArray(0, N < 5 ? N : 5);
    //gradCx->PrintDataArray();
    delete gradCx;

    std::cout << "--- CalcGradCy ---\n";
    IPhysField3D *gradCy = (IPhysField3D*)new PhysField3D(Nx, Ny, Nz, Hx, Hy, Hz);
    CalcGradCy(C, gradCy);
    gradCy->PrintDataArray(0, N < 5 ? N : 5);
    //gradCy->PrintDataArray();
    delete gradCy;

    std::cout << "--- CalcGradCz ---\n";
    IPhysField3D *gradCz = (IPhysField3D*)new PhysField3D(Nx, Ny, Nz, Hx, Hy, Hz);
    CalcGradCz(C, gradCz);
    gradCz->PrintDataArray(0, N < 5 ? N : 5);
    //gradCz->PrintDataArray();
    delete gradCz;

    delete C;//*/
    std::cout << "--- TestGradPhysField3DZ END ---\n";
}

int main()
{
    std::cout << "----- Pole 3D -----\n";
    //size_t Nbx = 2, Nby = 2, Nbz = 3;// Количество блоков сетки
    size_t Nbx = 500, Nby = 500, Nbz = 500;// Количество блоков сетки
    std::cout << "Nbx: " << Nbx << "; " << "Nby: " << Nby << "; "<< "Nbz: " << Nbz << "\n";
    size_t Nnbx = 2, Nnby = 2, Nnbz = 2;// Количество узлов в блоке
    std::cout << "Nnbx: " << Nnbx << "; " << "Nnby: " << Nnby << "; "<< "Nnbz: " << Nnbz << "\n";
    size_t Nx = Nbx*Nnbx, Ny = Nby*Nnby, Nz = Nbz*Nnbz;
    std::cout << "Nx: " << Nx << "; " << "Ny: " << Ny << "; " << "Nz: " << Nz << "\n";
    //size_t Nx = 1000, Ny = 1000, Nz = 1000;
    size_t N = Nx * Ny * Nz;
    std::cout << "N: " << N << "\n";
    double Hx = 0.1;
    double Hy = 0.2;
    double Hz = 0.3;    
    std::cout << "Hx: " << Hx << "; ";
    std::cout << "Hy: " << Hy << "; ";
    std::cout << "Hz: " << Hz << "\n";

    TestInitArray(N);
    TestGradPhysField3D(Nx,Ny,Nz,N,Hx,Hy,Hz);
    TestGradPhysField3DZ(Nbx,Nby,Nbz,Nnbx,Nnby,Nnbz,Nx,Ny,Nz,N,Hx,Hy,Hz);
    
    std::cout << "-----Success Exit -----\n";
}