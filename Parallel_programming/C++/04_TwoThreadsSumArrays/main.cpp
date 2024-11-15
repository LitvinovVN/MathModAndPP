// Задача 03. Выполнить замер времени исполнения функции void thread_function()
// Источник: https://en.cppreference.com/w/cpp/thread/sleep_for
// Запуск:
// g++ main.cpp  -o app
// nvcc main.cpp -o app
// ./app

#include <iostream>              // подключаем заголовочный файл iostream (содержит определение std::cout)
#include <thread>                // подключаем библиотеку для работы с потоками
#include <chrono>                // sleep_for

using namespace std::chrono_literals;// для использования единиц измерения времени (ms)

void thread_function(double* a1, double* a2, double* a, int size)                 
{
    //std::cout << "Thread function started. Pause 2000ms...\n";
    for(int i = 0; i < size; i++)
        a[i] = a1[i] + a2[i];
    //std::this_thread::sleep_for(2000ms);
    //std::cout << "Thread function ended!\n";
}

int main()
{
    // 1. Создаём массивы данных
    long N = 50000000;
    double* arr_a1 = new double[N];
    double* arr_b1 = new double[N];
    double* arr_a2 = new double[N];
    double* arr_b2 = new double[N];

    double* arr_a = new double[N];
    double* arr_b = new double[N];

    // 2. Инициализация
    for(int i = 0; i < N; i++)
    {
        arr_a1[i] = arr_b1[i] =  i * 0.1;
        arr_a2[i] = arr_b2[i] =  i * 0.01;
    }
    
    std::cout << "Main thread: Starting new threads...\n";

    auto start = std::chrono::high_resolution_clock::now();

    std::thread t1(&thread_function, arr_a1, arr_a2, arr_a, N);
    t1.join();// Закомментировано - параллельно, Раскомментировано - последовательно
    std::thread t2(&thread_function, arr_b1, arr_b2, arr_b, N);
    //t1.join();// Раскомментировано - параллельно, Закомментировано - последовательно
    t2.join(); 
    std::cout << "Main thread: Threads joined\n";

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end-start;

    std::cout << "Waited " << elapsed.count() << " ms\n";

    /*for(int i = 0; i < N; i++)
    {
        std::cout << "arr_a1[i] = " << arr_a1[i] << "; ";
        std::cout << "arr_a2[i] = " << arr_a2[i] << "; ";
        std::cout << "arr_a[i] = " << arr_a[i] << std::endl;

        std::cout << "arr_b1[i] = " << arr_b1[i] << "; ";
        std::cout << "arr_b2[i] = " << arr_b2[i] << "; ";
        std::cout << "arr_b[i] = " << arr_b[i] << std::endl;
    }*/

    return 0;
}