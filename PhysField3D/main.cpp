// g++ main.cpp -o app -O3 -fopenmp
// ./app
#include <iostream>
#include <chrono>
#include <thread>

#include "PhysField3D.hpp"
#include "PhysField3DZ.hpp"
#include "PhysField3DAlgGrad.hpp"

using namespace std::chrono;
using namespace std::chrono_literals;

void TestInitArray(size_t N)
{
    std::cout << "----- TestInitArray(" << N << ") START -----\n";
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
    std::cout << "----- TestInitArray(" << N << ") END -----\n\n";
}

void PrintArray3D(double* arr, size_t Nx, size_t Ny, size_t Nz)
{
    for (size_t i = 0; i < Nx; i++)
    {
        std::cout << "i = " << i << "\n";
        for (size_t j = 0; j < Ny; j++)
        {
            std::cout << "j = " << j << " | ";
            for (size_t k = 0; k < Nz; k++)
            {
                size_t m0 = k + j*Nz + i*Nz*Ny;  
                std::cout << arr[m0] << " ";
            }
            std::cout << "\n";
        }
    }    
}

void TestGradArray(size_t Nx, size_t Ny, size_t Nz, size_t N,
    double Hx, double Hy, double Hz)
{
    std::cout << "----- TestGradArray(" << N << ") START -----\n";
    double *arr = new double[N];
    double *arr_grad = new double[N];    
    double khx = 1/Hx;
    double k2hx = 1/(2*Hx);
    double khy = 1/Hy;
    double k2hy = 1/(2*Hy);
    double khz = 1/Hz;
    double k2hz = 1/(2*Hz);

    auto start = high_resolution_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        arr[i] = i;
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "arr init:" << duration.count() << " mks\n";
    if(N < 200)
        PrintArray3D(arr,Nx,Ny,Nz);    

    // Градиент для 1D
    /*start = high_resolution_clock::now();    
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        if(i==0)
        {
            arr_grad[0] = khx*(arr[1]-arr[0]);
        }
        else if(i==N-1)
        {
            arr_grad[N-1] = khx*(arr[N-1]-arr[N-2]);
        }
        else
        {
            arr_grad[i] = k2hx*(arr[i+1] - arr[i-1]);
        }        
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "arr gradient 1D:" << duration.count() << " mks\n";
    if(N < 200)
        PrintArray3D(arr_grad,Nx,Ny,Nz);//*/

    // Градиент для 3D
    // du/dx
    start = high_resolution_clock::now();    
    #pragma omp parallel for
    for (size_t i = 1; i < Nx-1; i++)
    for (size_t j = 1; j < Ny-1; j++)
    for (size_t k = 0; k < Nz; k++)
    {
        size_t m0 = k + j*Nz + i*Nz*Ny;
        size_t m6 = m0 - Nz*Ny;
        size_t m5 = m0 + Nz*Ny;
        arr_grad[m0] = k2hx*(arr[m5] - arr[m6]);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "arr gradient 3D by Ox:" << duration.count() << " mks\n";
    std::cout << "grad("<< N-1 <<"): "<< arr_grad[N-1] <<"\n";
    if(N < 200)
        PrintArray3D(arr_grad,Nx,Ny,Nz);

    // du/dy
    start = high_resolution_clock::now();    
    #pragma omp parallel for
    for (size_t i = 1; i < Nx-1; i++)
    for (size_t j = 1; j < Ny-1; j++)
    for (size_t k = 1; k < Nz-1; k++)
    {
        size_t m0 = k + j*Nz + i*Nz*Ny;
        size_t m4 = m0 - Nz;
        size_t m3 = m0 + Nz;
        arr_grad[m0] = k2hy*(arr[m3] - arr[m4]);
    }    
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "arr gradient 3D by Oy:" << duration.count() << " mks\n";
    std::cout << "grad("<< N-1 <<"): "<< arr_grad[N-1] <<"\n";
    if(N < 200)
        PrintArray3D(arr_grad,Nx,Ny,Nz);

    // du/dz
    start = high_resolution_clock::now();    
    #pragma omp parallel for
    for (size_t i = 1; i < Nx-1; i++)
    for (size_t j = 1; j < Ny-1; j++)
    for (size_t k = 1; k < Nz-1; k++)
    {
        size_t m0 = k + j*Nz + i*Nz*Ny;
        size_t m2 = m0 - 1;
        size_t m1 = m0 + 1;
        arr_grad[m0] = k2hz*(arr[m1] - arr[m2]);
    }    
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "arr gradient 3D by Ox:" << duration.count() << " mks\n";
    std::cout << "grad("<< N-1 <<"): "<< arr_grad[N-1] <<"\n";
    if(N < 200)
        PrintArray3D(arr_grad,Nx,Ny,Nz);

    ////////////////////////
    delete[] arr;
    delete[] arr_grad;
    std::cout << "----- TestGradArray(" << N << ") END -----\n\n";
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
    PhysField3DAlgGrad::InitByGlobalIndex(C);
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
    PhysField3DAlgGrad::InitByGlobalIndex(C);
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
    IPhysField3D *gradCx = (IPhysField3D*)new PhysField3DZ(Nbx, Nby, Nbz,
        Nnbx, Nnby, Nnbz, Hx, Hy, Hz, "gradCx");
    start = high_resolution_clock::now();
    //CalcGradCx(C, gradCx);// Неэффективная реализация
    PhysField3DAlgGrad::GradX((PhysField3DZ*)C, (PhysField3DZ*)gradCx);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Duration of PhysField3DAlgGrad::GradX((PhysField3DZ*)C, (PhysField3DZ*)gradCx), mks: " << duration.count() << std::endl;
        
    if(N < 200)
        gradCx->PrintDataArray();
    else
        gradCx->PrintDataArray(0, 10);
    delete gradCx;

    std::cout << "--- CalcGradCy ---\n";
    IPhysField3D *gradCy = (IPhysField3D*)new PhysField3D(Nx, Ny, Nz, Hx, Hy, Hz);
    //CalcGradCy(C, gradCy);// Неэффективная реализация
    gradCy->PrintDataArray(0, N < 5 ? N : 5);
    //gradCy->PrintDataArray();
    delete gradCy;

    std::cout << "--- CalcGradCz ---\n";
    IPhysField3D *gradCz = (IPhysField3D*)new PhysField3D(Nx, Ny, Nz, Hx, Hy, Hz);
    //CalcGradCz(C, gradCz);// Неэффективная реализация
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

    //TestInitArray(N);
    TestGradArray(Nx,Ny,Nz,N,Hx,Hy,Hz);
    //TestGradPhysField3D(Nx,Ny,Nz,N,Hx,Hy,Hz);
    TestGradPhysField3DZ(Nbx,Nby,Nbz,Nnbx,Nnby,Nnbz,Nx,Ny,Nz,N,Hx,Hy,Hz);
    
    std::cout << "-----Success Exit -----\n";
}