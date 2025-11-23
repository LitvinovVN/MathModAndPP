// g++ -O3 gradient.cpp -o gradient  -fopenmp
// ./gradient

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

int main() {
    // Размеры поля
    const int Nx = 1000, Ny = 1000, Nz = 1000;
    
    // Шаги сетки (предполагаем равномерную сетку)
    const double dx = 0.1, dy = 0.2, dz = 0.3;

    // 1. Выделение памяти и инициализация трёхмерного поля
    vector<vector<vector<double>>> field(Nx, 
                                        vector<vector<double>>(Ny, 
                                        vector<double>(Nz)));

    auto start_init = high_resolution_clock::now();
    // Инициализация функцией f(x,y,z) = x² + y² + z²
    #pragma omp parallel for
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                double x = i * dx;
                double y = j * dy;
                double z = k * dz;
                field[i][j][k] = x*x + y*y + z*z;
            }
        }
    }
    auto end_init = high_resolution_clock::now();

    // 2. Вычисление градиента
    /*vector<vector<vector<double>>> grad_x(Nx, 
                                         vector<vector<double>>(Ny, 
                                         vector<double>(Nz)));//*/
    /*vector<vector<vector<double>>> grad_y(Nx, 
                                         vector<vector<double>>(Ny, 
                                         vector<double>(Nz)));//*/
    vector<vector<vector<double>>> grad_z(Nx, 
                                         vector<vector<double>>(Ny, 
                                         vector<double>(Nz)));//*/

    // df/dx
    /*auto start_x = high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (i == 0) {
                    // Передняя разность на левой границе
                    grad_x[i][j][k] = (field[i+1][j][k] - field[i][j][k]) / dx;
                } else if (i == Nx-1) {
                    // Backward difference на правой границе
                    grad_x[i][j][k] = (field[i][j][k] - field[i-1][j][k]) / dx;
                } else {
                    // Центральная разность внутри области
                    grad_x[i][j][k] = (field[i+1][j][k] - field[i-1][j][k]) / (2*dx);
                }
            }
        }
    }
    auto end_x = high_resolution_clock::now();//*/

    // 3. df/dy
    /*auto start_y = high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (j == 0) {
                    grad_y[i][j][k] = (field[i][j+1][k] - field[i][j][k]) / dy;
                } else if (j == Ny-1) {
                    grad_y[i][j][k] = (field[i][j][k] - field[i][j-1][k]) / dy;
                } else {
                    grad_y[i][j][k] = (field[i][j+1][k] - field[i][j-1][k]) / (2*dy);
                }
            }
        }
    }
    auto end_y = high_resolution_clock::now();//*/

    // 4. df/dz
    auto start_z = high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (k == 0) {
                    grad_z[i][j][k] = (field[i][j][k+1] - field[i][j][k]) / dz;
                } else if (k == Nz-1) {
                    grad_z[i][j][k] = (field[i][j][k] - field[i][j][k-1]) / dz;
                } else {
                    grad_z[i][j][k] = (field[i][j][k+1] - field[i][j][k-1]) / (2*dz);
                }
            }
        }
    }
    auto end_z = high_resolution_clock::now();//*/

    // Вывод результатов замеров времени
    auto duration_init = duration_cast<milliseconds>(end_init - start_init);
    //auto duration_x = duration_cast<milliseconds>(end_x - start_x);
    //auto duration_y = duration_cast<milliseconds>(end_y - start_y);
    auto duration_z = duration_cast<milliseconds>(end_z - start_z);

    cout << "Field init: " << duration_init.count() << " ms" << endl;
    //cout << "Calc df/dx: " << duration_x.count() << " ms" << endl;
    //cout << "Calc df/dy: " << duration_y.count() << " ms" << endl;
    cout << "Calc df/dz: " << duration_z.count() << " ms" << endl;

    // Проверка результата в произвольной точке (например, центр)
    int i = Nx/2, j = Ny/2, k = Nz/2;
    cout << "\nProverka v tochke (" << i << "," << j << "," << k << "):" << endl;
    //cout << "df/dx: " << grad_x[i][j][k] << " (theoretical value: " << 2*i*dx << ")" << endl;
    //cout << "df/dy: " << grad_y[i][j][k] << " (theoretical value: " << 2*j*dy << ")" << endl;
    cout << "df/dz: " << grad_z[i][j][k] << " (theoretical value: " << 2*k*dz << ")" << endl;

    return 0;
}