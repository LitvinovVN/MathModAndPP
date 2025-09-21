// g++ main.cpp -o app
// ./app
// clang main.cpp
// ./a

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib> // для system()
#include <cmath>
#include <utility> // swap

void SetBoundaryConditions1(unsigned Nx, unsigned Ny, double hx, double hy, 
                             double Lx, double Ly, double* phys_field_T)
{
    // Верхняя граница
    for(unsigned i=0; i<Nx; i++)
    {
            unsigned index = (Ny-1) * Nx + i;
            double value = 2; //func(x, y)
            phys_field_T[index] = value;                   
    }

    // Нижняя граница
    for(unsigned i=0; i<Nx; i++)
    {
            unsigned index = 0 * Nx + i;
            double value = 1; //func(x, y)
            phys_field_T[index] = value;                   
    }

    // Левая граница
    for(unsigned j=0; j<Ny; j++)
    {
            unsigned index = j * Nx + 0;
            double value = 3; //func(x, y)
            phys_field_T[index] = value;                   
    }

    // Правая граница
    for(unsigned j=0; j<Ny; j++)
    {
            unsigned index = j * Nx + Nx - 1;
            double value = 4; //func(x, y)
            phys_field_T[index] = value;                   
    }
}

void SavePhysField2DToFileTxt(unsigned Nx, unsigned Ny, double hx, double hy, 
                             double Lx, double Ly, double* phys_field_T, std::string fileName)
{
    // Открываем файл для записи
    std::ofstream outfile(fileName);
    
    if (!outfile.is_open()) {
        std::cerr << "Error: file not opened!" << std::endl;
        return;
    }
    
    // Записываем заголовок с информацией о сетке
    outfile << "# 2D Physical Field Data" << std::endl;
    outfile << "# Grid dimensions: " << Nx << " x " << Ny << std::endl;
    outfile << "# Grid spacing: hx = " << hx << ", hy = " << hy << std::endl;
    outfile << "# Domain size: Lx = " << Lx << ", Ly = " << Ly << std::endl;
    outfile << "# Format: x y value" << std::endl;
    outfile << std::endl;
    
    // Записываем данные поля
    for (unsigned j = 0; j < Ny; j++) {
        for (unsigned i = 0; i < Nx; i++) {
            unsigned index = j * Nx + i;
            double x = i * hx;
            double y = j * hy;
            
            outfile << std::scientific << std::setprecision(6)
                   << x << " " << y << " " << phys_field_T[index] << std::endl;
        }
        outfile << std::endl; // Пустая строка между слоями y для лучшей читаемости
    }
    
    outfile.close();
    std::cout << "Data saved in file: " << fileName << std::endl;
}

void PlotHeatmap(std::string fileName)
{
    std::string command = "python3 plot_heatmap.py " + fileName + " --save-svg";
    int result = std::system(command.c_str());
    
    if (result == 0) {
        std::cout << "Python script executed successfully" << std::endl;
    } else {
        std::cout << "Python script failed with code: " << result << std::endl;
    }
}

struct Koeff2D
{
    double B0{0};
    double B1{0};
    double B2{0};
    double B3{0};
    double B4{0};

    Koeff2D()
    {}

    Koeff2D(double B0, double B1, double B2, double B3, double B4)
        : B0(B0), B1(B1), B2(B2), B3(B3), B4(B4)
    { }

    void Print()
    {
        std::cout << "{";
        std::cout << "B0: " << B0 << "; ";
        std::cout << "B1: " << B1 << "; ";
        std::cout << "B2: " << B2 << "; ";
        std::cout << "B3: " << B3 << "; ";
        std::cout << "B4: " << B4 << "";
        std::cout << "}";
    }
};

struct Koeff2DMatrix
{
    unsigned Nx;
    unsigned Ny;
    Koeff2D* Koeff2DArray = nullptr;

    Koeff2DMatrix(unsigned Nx, unsigned Ny)
        : Nx(Nx), Ny(Ny)
    {
        Koeff2DArray = new Koeff2D[Nx*Ny];
    }

    ~Koeff2DMatrix()
    {
        delete[] Koeff2DArray;
    }

    Koeff2D* GetKoeff2DArray()
    {
        return Koeff2DArray;
    }

    unsigned GetIndex(unsigned i, unsigned j)
    {
        return j*Nx + i;
    }

    Koeff2D& GetNode(unsigned i, unsigned j)
    {
        unsigned index = GetIndex(i,j);
        return Koeff2DArray[index];
    }

    void PrintKoeff(unsigned i, unsigned j)
    {
        std::cout << "(" << i <<", " << j << ") ";
        unsigned index = GetIndex(i, j);
        std::cout << "[" << index << "]: ";
        (Koeff2DArray[index]).Print();
        std::cout << std::endl;
    }

    void AddPartialDerivative2ByX(double hx)
    {
        for(int j = 1; j < Ny-1; j++)
        {
            for(int i = 1; i < Nx-1; i++)
            {
                double B0x = -2.0/(hx*hx);
                double B1 = 1.0/(hx*hx);
                double B2 = 1.0/(hx*hx);

                unsigned index = GetIndex(i, j);
                Koeff2D* grid2DNode = &Koeff2DArray[index];
                grid2DNode->B0+=B0x;
                if(i != Nx-2)
                    grid2DNode->B1+=B1;
                if(i != 1)
                grid2DNode->B2+=B2;
            }
        }
    }

    void AddPartialDerivative2ByY(double hy)
    {
        for(int j = 1; j < Ny-1; j++)
        {
            for(int i = 1; i < Nx-1; i++)
            {
                double B0y = -2.0/(hy*hy);
                double B3 = 1.0/(hy*hy);
                double B4 = 1.0/(hy*hy);

                unsigned index = GetIndex(i, j);
                Koeff2D* grid2DNode = &Koeff2DArray[index];
                grid2DNode->B0+=B0y;
                if(j != Ny-2)
                    grid2DNode->B3+=B3;
                if(j != 1)
                    grid2DNode->B4+=B4;
            }
        }
    }
};

int main()
{
    std::cout << "Poisson 2D + Jacobi" << std::endl;

    // Размеры физической области
    double Lx = 1;
    double Ly = 1;

    // Кол-во блоков сетки
    unsigned Nbx = 5;
    unsigned Nby = 5;

    // Кол-во узлов в блоках сетки
    unsigned bx_size = 10;
    unsigned by_size = 10;

    // Кол-во узлов сетки
    unsigned Nx = Nbx * bx_size;
    unsigned Ny = Nby * by_size;

    // Размер сетки
    unsigned N = Nx*Ny;

    // Шаги сетки
    double hx = Lx/Nx;
    double hy = Ly/Ny;

    // Поле физических величин
    double* phys_field_T = new double[N];
    for(unsigned j=0; j<Ny; j++)
    {
        for(unsigned i=0; i<Nx; i++)
        {
            unsigned index = j * Nx + i;

            phys_field_T[index] = 0;
            if(i > 10 && i < 20)
            {
                if(j > 20 && j < 30)
                {
                    phys_field_T[index] = 1;
                }
            }
        }
    }

    // Граничные условия
    SetBoundaryConditions1(Nx, Ny, hx, hy, Lx, Ly, phys_field_T);
    
    std::string dataFileName = "0-init-phys_field_2D.txt";
    SavePhysField2DToFileTxt(Nx, Ny, hx, hy, Lx, Ly, phys_field_T, dataFileName);
    // Вызов Python скрипта
    PlotHeatmap(dataFileName);

    ///////////// Коэффициенты сеточных уравнений ////////////
    Koeff2DMatrix A(Nx, Ny);
    A.AddPartialDerivative2ByX(hx);
    A.AddPartialDerivative2ByY(hy);
    A.PrintKoeff(0,0);
    A.PrintKoeff(1,1);
    A.PrintKoeff(2,1);
    A.PrintKoeff(Nx-2,1);
    A.PrintKoeff(Nx-1,1);
    A.PrintKoeff(Nx-2,2);
    A.PrintKoeff(Nx-2,Ny-3);
    A.PrintKoeff(Nx-2,Ny-2);
    A.PrintKoeff(Nx-2,Ny-1);
    A.PrintKoeff(10,10);

    // Источники
    std::cout << "\n\nf...\n";
    double* f = new double[N];
    for(unsigned j=0; j<Ny; j++)
    {
        for(unsigned i=0; i<Nx; i++)
        {
            unsigned index = j * Nx + i;

            f[index] = 0;
            if(i > 10 && i < 20)
            {
                if(j > 20 && j < 30)
                {
                    f[index] = 1;
                }
            }
        }
    }

    // Правая часть СЛАУ
    std::cout << "\n\nb...\n";
    double* b = new double[N];
    for(unsigned j=1; j<Ny-1; j++)
    {
        for(unsigned i=1; i<Nx-1; i++)
        {
            unsigned index = j * Nx + i;
            //std::cout << index << ": ";
            auto& node = A.GetNode(i, j);
            //node.Print();
            b[index] = f[index];
            if(fabs(node.B1) < 1e-8)
                b[index] -= phys_field_T[index+1]/(hx*hx);
            if(fabs(node.B2) < 1e-8)
                b[index] -= phys_field_T[index-1]/(hx*hx);
            if(fabs(node.B3) < 1e-8)
                b[index] -= phys_field_T[index+Nx]/(hy*hy);
            if(fabs(node.B4) < 1e-8)
                b[index] -= phys_field_T[index-Nx]/(hy*hy);
        }
    }

    //Jacobi
    std::cout << "\n\nJacobi...\n";
    double* phys_field_T2 = new double[N];
    for(unsigned i{0}; i<N; i++)
    {
        phys_field_T2[i] = phys_field_T[i];
    }

    bool isEstablished = false;
    unsigned iter = 0;

    Koeff2D* koeff2DArray = A.GetKoeff2DArray();
    while (!isEstablished)
    {
        iter++;
        isEstablished = true;
        
        for(unsigned j=1; j<Ny-1; j++)
        {
            for(unsigned i=1; i<Nx-1; i++)
            {
                unsigned m0 = j * Nx + i;
                unsigned m1 = m0 + 1;
                unsigned m2 = m0 - 1;
                unsigned m3 = m0 + Nx;
                unsigned m4 = m0 - Nx;

                double B0 = (koeff2DArray[m0]).B0;
                double B1 = (koeff2DArray[m0]).B1;
                double B2 = (koeff2DArray[m0]).B2;
                double B3 = (koeff2DArray[m0]).B3;
                double B4 = (koeff2DArray[m0]).B4;

                phys_field_T2[m0] = (1/B0)*(b[m0]
                    - B1*phys_field_T[m1] - B2*phys_field_T[m2]
                    - B3*phys_field_T[m3] - B4*phys_field_T[m4]);

                if(isEstablished && fabs(phys_field_T[m0]-phys_field_T2[m0])>10e-8)
                    isEstablished = false;
            }
        }

        std::swap(phys_field_T, phys_field_T2);

        if(iter == 1)
        {
            std::string dataFileName = "j1-phys_field_2D.txt";
            SavePhysField2DToFileTxt(Nx, Ny, hx, hy, Lx, Ly, phys_field_T, dataFileName);
            // Вызов Python скрипта
            PlotHeatmap(dataFileName);
        }

        if(iter == 10)
        {
            std::string dataFileName = "j10-phys_field_2D.txt";
            SavePhysField2DToFileTxt(Nx, Ny, hx, hy, Lx, Ly, phys_field_T, dataFileName);
            // Вызов Python скрипта
            PlotHeatmap(dataFileName);
        }

        if(iter == 100)
        {
            std::string dataFileName = "j100-phys_field_2D.txt";
            SavePhysField2DToFileTxt(Nx, Ny, hx, hy, Lx, Ly, phys_field_T, dataFileName);
            // Вызов Python скрипта
            PlotHeatmap(dataFileName);
        }

        if(iter == 1000)
        {
            std::string dataFileName = "j1000-phys_field_2D.txt";
            SavePhysField2DToFileTxt(Nx, Ny, hx, hy, Lx, Ly, phys_field_T, dataFileName);
            // Вызов Python скрипта
            PlotHeatmap(dataFileName);
        }

        if(iter>10000)
        {
            std::cout << "iter>10000!" << std::endl;
            break;
        }
    }
    delete[] phys_field_T2;

    std::cout << "iter = " << iter << std::endl;

    std::string dataFileNameJacobi = "j" + std::to_string(iter) + "-phys_field_2D.txt";
    SavePhysField2DToFileTxt(Nx, Ny, hx, hy, Lx, Ly, phys_field_T, dataFileNameJacobi);
    // Вызов Python скрипта
    PlotHeatmap(dataFileNameJacobi);
    
    // Очистка памяти
    delete[] phys_field_T;
}