// Задача 01. Запустить функцию void thread_function(),
// выводящую в новом потоке в консоль строку "Thread function started. Pause 2000ms...\n"",
// затем выдерживающую паузу 2000мс и выводящую строку "Thread function ended!\n".
// Источник: https://www.bogotobogo.com/cplusplus/C11/1_C11_creating_thread.php
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
    std::thread t(&thread_function);   // t starts running
    std::cout << "Main thread: New thread started!\n";
    t.join();   // main thread waits for the thread t to finish
    std::cout << "Main thread: Thread joined\n";

    return 0;
}