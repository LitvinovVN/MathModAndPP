// g++ -o gradient_calc gradient_calc.cpp -O3 -fopenmp -std=c++11
// ./gradient_calc
// python3 visualize_gradient.py
#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <omp.h>

class GradientCalculator {
private:
    int nx, ny;              // Размеры сетки
    double x_min, x_max;     // Границы по x
    double y_min, y_max;     // Границы по y
    double dx, dy;           // Шаги сетки
    double h;                // Шаг для производных
    
    std::vector<std::vector<double>> f;      // Значения функции
    std::vector<std::vector<double>> grad_x; // Компонента x градиента
    std::vector<std::vector<double>> grad_y; // Компонента y градиента
    std::vector<std::vector<double>> grad_magnitude; // Модуль градиента

public:
    GradientCalculator(int n_x, int n_y, 
                      double xmin = 0.0, double xmax = 1.0,
                      double ymin = 0.0, double ymax = 1.0)
        : nx(n_x), ny(n_y), x_min(xmin), x_max(xmax), 
          y_min(ymin), y_max(ymax) {
        
        dx = (x_max - x_min) / (nx - 1);
        dy = (y_max - y_min) / (ny - 1);
        h = std::min(dx, dy) * 0.01; // Малый шаг для производных
        
        // Выделение памяти
        f.resize(nx, std::vector<double>(ny, 0.0));
        grad_x.resize(nx, std::vector<double>(ny, 0.0));
        grad_y.resize(nx, std::vector<double>(ny, 0.0));
        grad_magnitude.resize(nx, std::vector<double>(ny, 0.0));
        
        initialize_function();
    }
    
    // Инициализация тестовой функции
    void initialize_function() {
        for (int i = 0; i < nx; ++i) {
            double x = x_min + i * dx;
            for (int j = 0; j < ny; ++j) {
                double y = y_min + j * dy;
                f[i][j] = x*x + y*y + std::sin(x) * std::cos(y);
            }
        }
    }
    
    // Аналитическое вычисление градиента (для проверки)
    void analytical_gradient() {
        for (int i = 0; i < nx; ++i) {
            double x = x_min + i * dx;
            for (int j = 0; j < ny; ++j) {
                double y = y_min + j * dy;
                grad_x[i][j] = 2*x + std::cos(x) * std::cos(y);
                grad_y[i][j] = 2*y - std::sin(x) * std::sin(y);
                grad_magnitude[i][j] = std::sqrt(grad_x[i][j]*grad_x[i][j] + 
                                               grad_y[i][j]*grad_y[i][j]);
            }
        }
    }
    
    // Последовательное вычисление градиента
    double sequential_gradient() {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 1; i < nx - 1; ++i) {
            for (int j = 1; j < ny - 1; ++j) {
                // Центральные разности для ∂f/∂x
                grad_x[i][j] = (f[i+1][j] - f[i-1][j]) / (2 * dx);
                
                // Центральные разности для ∂f/∂y
                grad_y[i][j] = (f[i][j+1] - f[i][j-1]) / (2 * dy);
                
                // Модуль градиента
                grad_magnitude[i][j] = std::sqrt(grad_x[i][j]*grad_x[i][j] + 
                                               grad_y[i][j]*grad_y[i][j]);
            }
        }
        
        // Граничные условия (односторонние разности)
        process_boundaries();
        
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    // Параллельное вычисление с std::threads
    double parallel_threads_gradient(int num_threads = 0) {
        if (num_threads <= 0) {
            num_threads = std::thread::hardware_concurrency();
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::thread> threads;
        int rows_per_thread = (nx - 2) / num_threads;
        
        for (int t = 0; t < num_threads; ++t) {
            int start_row = 1 + t * rows_per_thread;
            int end_row = (t == num_threads - 1) ? nx - 1 : start_row + rows_per_thread;
            
            threads.emplace_back([this, start_row, end_row]() {
                this->compute_gradient_range(start_row, end_row);
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        process_boundaries();
        
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    // Параллельное вычисление с OpenMP
    double parallel_omp_gradient() {
        auto start = std::chrono::high_resolution_clock::now();
        
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < nx - 1; ++i) {
            for (int j = 1; j < ny - 1; ++j) {
                grad_x[i][j] = (f[i+1][j] - f[i-1][j]) / (2 * dx);
                grad_y[i][j] = (f[i][j+1] - f[i][j-1]) / (2 * dy);
                grad_magnitude[i][j] = std::sqrt(grad_x[i][j]*grad_x[i][j] + 
                                               grad_y[i][j]*grad_y[i][j]);
            }
        }
        
        process_boundaries();
        
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    
private:
    // Вычисление градиента в диапазоне строк
    void compute_gradient_range(int start_row, int end_row) {
        for (int i = start_row; i < end_row; ++i) {
            for (int j = 1; j < ny - 1; ++j) {
                grad_x[i][j] = (f[i+1][j] - f[i-1][j]) / (2 * dx);
                grad_y[i][j] = (f[i][j+1] - f[i][j-1]) / (2 * dy);
                grad_magnitude[i][j] = std::sqrt(grad_x[i][j]*grad_x[i][j] + 
                                               grad_y[i][j]*grad_y[i][j]);
            }
        }
    }
    
    // Обработка граничных точек
    void process_boundaries() {
        // Верхняя и нижняя границы
        for (int j = 0; j < ny; ++j) {
            // Верхняя граница (i = 0)
            if (nx > 1) {
                grad_x[0][j] = (f[1][j] - f[0][j]) / dx;  // Вперед
                grad_y[0][j] = (j < ny - 1) ? (f[0][j+1] - f[0][j]) / dy : 0;
            }
            
            // Нижняя граница (i = nx-1)
            if (nx > 1) {
                grad_x[nx-1][j] = (f[nx-1][j] - f[nx-2][j]) / dx;  // Назад
                grad_y[nx-1][j] = (j < ny - 1) ? (f[nx-1][j+1] - f[nx-1][j]) / dy : 0;
            }
            
            // Вычисление модуля для границ
            grad_magnitude[0][j] = std::sqrt(grad_x[0][j]*grad_x[0][j] + 
                                           grad_y[0][j]*grad_y[0][j]);
            grad_magnitude[nx-1][j] = std::sqrt(grad_x[nx-1][j]*grad_x[nx-1][j] + 
                                              grad_y[nx-1][j]*grad_y[nx-1][j]);
        }
        
        // Левая и правая границы (исключая углы)
        for (int i = 1; i < nx - 1; ++i) {
            // Левая граница (j = 0)
            grad_x[i][0] = (f[i+1][0] - f[i-1][0]) / (2 * dx);
            grad_y[i][0] = (f[i][1] - f[i][0]) / dy;  // Вперед
            
            // Правая граница (j = ny-1)
            grad_x[i][ny-1] = (f[i+1][ny-1] - f[i-1][ny-1]) / (2 * dx);
            grad_y[i][ny-1] = (f[i][ny-1] - f[i][ny-2]) / dy;  // Назад
            
            // Вычисление модуля
            grad_magnitude[i][0] = std::sqrt(grad_x[i][0]*grad_x[i][0] + 
                                           grad_y[i][0]*grad_y[i][0]);
            grad_magnitude[i][ny-1] = std::sqrt(grad_x[i][ny-1]*grad_x[i][ny-1] + 
                                              grad_y[i][ny-1]*grad_y[i][ny-1]);
        }
    }
    
public:
    // Проверка корректности вычислений
    double check_accuracy() {
        analytical_gradient();
        double max_error = 0.0;
        double avg_error = 0.0;
        int count = 0;
        
        for (int i = 1; i < nx - 1; ++i) {
            for (int j = 1; j < ny - 1; ++j) {
                double analytical_x = 2*(x_min + i*dx) + 
                                    std::cos(x_min + i*dx) * std::cos(y_min + j*dy);
                double analytical_y = 2*(y_min + j*dy) - 
                                    std::sin(x_min + i*dx) * std::sin(y_min + j*dy);
                
                double error_x = std::abs(grad_x[i][j] - analytical_x);
                double error_y = std::abs(grad_y[i][j] - analytical_y);
                double total_error = std::sqrt(error_x*error_x + error_y*error_y);
                
                max_error = std::max(max_error, total_error);
                avg_error += total_error;
                count++;
            }
        }
        
        return (count > 0) ? avg_error / count : 0.0;
    }
    
    // Сохранение результатов в файл
    void save_results(const std::string& filename) {
        std::ofstream file(filename);
        file << "x,y,f,grad_x,grad_y,grad_magnitude\n";
        
        for (int i = 0; i < nx; ++i) {
            double x = x_min + i * dx;
            for (int j = 0; j < ny; ++j) {
                double y = y_min + j * dy;
                file << std::fixed << std::setprecision(6)
                     << x << "," << y << "," 
                     << f[i][j] << "," 
                     << grad_x[i][j] << "," 
                     << grad_y[i][j] << "," 
                     << grad_magnitude[i][j] << "\n";
            }
        }
        file.close();
    }
    
    // Вывод информации о сетке
    void print_grid_info() {
        std::cout << "Сетка: " << nx << " x " << ny << " точек\n";
        std::cout << "Область: x ∈ [" << x_min << ", " << x_max 
                  << "], y ∈ [" << y_min << ", " << y_max << "]\n";
        std::cout << "Шаги: dx = " << dx << ", dy = " << dy << "\n";
        std::cout << "Шаг для производных: h = " << h << "\n";
    }
};

// Функция для тестирования производительности
void run_performance_test(int nx, int ny) {
    std::cout << "\n=== Тестирование производительности ===\n";
    std::cout << "Размер сетки: " << nx << " x " << ny << "\n";
    
    GradientCalculator calc(nx, ny);
    calc.print_grid_info();
    
    // Тестирование разных методов
    double time_seq = calc.sequential_gradient();
    double time_threads = calc.parallel_threads_gradient();
    double time_omp = calc.parallel_omp_gradient();
    
    // Проверка точности
    double accuracy = calc.check_accuracy();
    
    // Вывод результатов
    std::cout << "\n=== Результаты ===\n";
    std::cout << "Последовательная версия: " << time_seq << " мс\n";
    std::cout << "Параллельная (threads): " << time_threads << " мс\n";
    std::cout << "Параллельная (OpenMP): " << time_omp << " мс\n";
    
    std::cout << "\n=== Ускорение ===\n";
    std::cout << "Ускорение threads: " << time_seq / time_threads << "x\n";
    std::cout << "Ускорение OpenMP: " << time_seq / time_omp << "x\n";
    
    std::cout << "\n=== Точность ===\n";
    std::cout << "Средняя ошибка: " << accuracy << "\n";
    
    // Сохранение результатов
    calc.save_results("gradient_results.csv");
    std::cout << "\nРезультаты сохранены в gradient_results.csv\n";
}

// Тестирование с разным количеством потоков
void test_thread_scaling(int nx, int ny) {
    std::cout << "\n=== Масштабируемость по потокам ===\n";
    
    GradientCalculator calc(nx, ny);
    int max_threads = std::thread::hardware_concurrency();
    
    std::cout << "Доступно потоков: " << max_threads << "\n";
    
    for (int threads = 1; threads <= max_threads; ++threads) {
        double time = calc.parallel_threads_gradient(threads);
        std::cout << threads << " поток(ов): " << time << " мс\n";
    }
}

int main() {
    #ifdef _WIN32
        system("chcp 65001 > nul"); // Устанавливает UTF-8
    #endif

    std::cout << "Параллельное вычисление градиента функции\n";
    std::cout << "=========================================\n";
    
    // Тестирование на разных размерах сетки
    std::vector<std::pair<int, int>> test_sizes = {
        {100, 100},    // Малая сетка
        {500, 500},    // Средняя сетка  
        {1000, 1000},   // Большая сетка
        {10000, 1000}   // Большая сетка
    };
    
    for (const auto& size : test_sizes) {
        run_performance_test(size.first, size.second);
    }
    
    // Тестирование масштабируемости
    test_thread_scaling(1000, 1000);
    
    return 0;
}