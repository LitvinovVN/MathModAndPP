// Задача 06. condition_variable
// Источник: https://habr.com/ru/post/182626/
// Запуск:
// g++ main.cpp -std=c++11 -pthread -o app
// nvcc main.cpp -o app
// ./app

#include <condition_variable>
#include <iostream>
#include <random>
#include <thread>
#include <mutex>
#include <queue>

std::mutex              g_lockprint;
std::mutex              g_lockqueue;
std::condition_variable g_queuecheck;
std::queue<int>         g_codes;
bool                    g_done;
bool                    g_notified;

void workerFunc(int id, std::mt19937 &generator)
{
     // стартовое сообщение
     {
          std::unique_lock<std::mutex> locker(g_lockprint);
          std::cout << "[worker " << id << "]\trunning..." << std::endl;
     }
     // симуляция работы
     std::this_thread::sleep_for(std::chrono::seconds(1 + generator() % 5));
     // симуляция ошибки
     int errorcode = id*100+1;
     {
          std::unique_lock<std::mutex> locker(g_lockprint);
          std::cout  << "[worker " << id << "]\tan error occurred: " << errorcode << std::endl;
     }
     // сообщаем об ошибке
     {
          std::unique_lock<std::mutex> locker(g_lockqueue);
          g_codes.push(errorcode);
          g_notified = true;
          g_queuecheck.notify_one();
      }
}

void loggerFunc()
{
     // стартовое сообщение
     {
          std::unique_lock<std::mutex> locker(g_lockprint);
          std::cout << "[logger]\trunning..." << std::endl;
     }
     // до тех пор, пока не будет получен сигнал
     while(!g_done)
     {
          std::unique_lock<std::mutex> locker(g_lockqueue);
          while(!g_notified) // от ложных пробуждений
               g_queuecheck.wait(locker);
          // если есть ошибки в очереди, обрабатывать их
          while(!g_codes.empty())
          {
               std::unique_lock<std::mutex> locker(g_lockprint);
               std::cout << "[logger]\tprocessing error:  " << g_codes.front()  << std::endl;
               g_codes.pop();
          }
          g_notified = false;
     }
}

int main()
{
    int numThreads;
    std::cout << "Unput Threads Number: ";    
    std::cin >> numThreads;

     // инициализация генератора псевдо-случайных чисел
     std::mt19937 generator((unsigned int)std::chrono::system_clock::now().time_since_epoch().count());
     // запуск регистратора
     std::thread loggerThread(loggerFunc);
     // запуск рабочих
     std::vector<std::thread> threads;
     
     for(int i = 0; i < numThreads; ++i)
          threads.push_back(std::thread(workerFunc, i+1, std::ref(generator)));
     for(auto &t: threads)
          t.join();
     // сообщаем регистратору о завершении и ожидаем его
     g_done = true;
     loggerThread.join();
     return 0;
}