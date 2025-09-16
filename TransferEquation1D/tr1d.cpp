// g++ tr1d.cpp -o app
// ./app

#include <iostream>
#include <iomanip>

void PrintU(double* u, unsigned Nx)
{
    std::cout << std::fixed;
    std::setprecision(4);
    for(unsigned i = 0; i < Nx; i++)
    {
        std::cout << u[i] << " ";
    }
    std::cout << std::endl;
}

int main()
{
    std::cout << "du/dt + v du/dx = 0" << std::endl;

    // 1. Создаём и инициализируем одномерное поле ф-ции u
    unsigned Nx = 20;
    double* u = new double[Nx];

    // 000000001111111111111111110000
    // 0--...--Nx/4---...---2Nx/4---Nx-1

    for(unsigned i = 0; i < Nx; i++)
    {
        u[i] = 0;
        if (i > Nx/4 && i < 2*Nx/4)
             u[i] = 1;
    }

    PrintU(u, Nx);

    double v = 1;
    double tau = 0.01;
    double tmax = 1 + 0.5*tau;
    double h = 1;

    // 2. Перенос по явной схеме
    double uim1_prev = u[0];
    double ui_prev = u[1];
    double uip1_prev = u[2];

    double t = 0;
    do
    {
        t += tau;
        for(unsigned i = 1; i < Nx-1; i++)
        {
            u[i] = ui_prev - (v*tau/(2*h))*(uip1_prev-uim1_prev);
            // u>=0
            if(u[i] < 0) u[i] = 0;

            uim1_prev = ui_prev;
            ui_prev = uip1_prev;
            uip1_prev = u[i+1];
        }
        std::cout << "t=" << t << ": ";
        PrintU(u, Nx);
    } while (t <= tmax);
    
    
    
}