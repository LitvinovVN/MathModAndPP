// Задача 03. Выполнить замер времени исполнения функции void thread_function()
// Источник: https://en.cppreference.com/w/cpp/thread/sleep_for
// Запуск:
// g++ main.cpp -std=c++11 -pthread -o app
// nvcc main.cpp -o app
// ./app

#include <iostream>              // подключаем заголовочный файл iostream (содержит определение std::cout)
#include <thread>                // подключаем библиотеку для работы с потоками
#include <chrono>                // sleep_for

using namespace std::chrono_literals;// для использования единиц измерения времени (ms)

void thread_function()                 
{
    std::cout << "Thread function started. Pause 2000ms...\n";
    std::this_thread::sleep_for(2000ms);
    std::cout << "Thread function ended!\n";
}

int main()
{
    std::cout << "Main thread: Starting new thread...\n";

    auto start = std::chrono::high_resolution_clock::now();

    std::thread t(&thread_function);   // t starts running
    std::cout << "Main thread: New thread started!\n";
    t.join();   // main thread waits for the thread t to finish
    std::cout << "Main thread: Thread joined\n";

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end-start;

    std::cout << "Waited " << elapsed.count() << " ms\n";

    return 0;
}