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

enum STATUS
{
     INIT = 1,
     STARTED = 2,
     BUSY = 3,
     READY = 4
};
std::vector<STATUS> g_statuses;
double *data;             // Массив данных для обработки потоками
int command = 0;          // Команда для рабочих потоков
bool is_can_work = false; // Флаг для рабочих потоков, сигнализирующий о возможности приступать к работе
bool is_exit = false;     // Флаг для рабочих потоков, сигнализирующий о том, что все команды обработаны и нужно завершать работу

std::mutex g_lockprint;
std::mutex g_lock_statuses;

void workerFunc(int id, std::mt19937 &generator)
{
     // стартовое сообщение
     {
          std::unique_lock<std::mutex> locker(g_lockprint);
          std::cout << "[worker " << id << "]\trunning..." << std::endl;
     }

     // Устанавливаем статус INIT
     {
          std::unique_lock<std::mutex> locker(g_lock_statuses);
          g_statuses[id] = INIT;
     }

     // Симуляция инициализации
     std::this_thread::sleep_for(std::chrono::microseconds(1 + generator() % 5));

     // Устанавливаем статус STARTED
     {
          std::unique_lock<std::mutex> locker(g_lock_statuses);
          g_statuses[id] = STARTED;
     }

     // Ожидаем флаг is_can_work
     while (!is_can_work)
     {
          std::this_thread::yield();
     }

     // Устанавливаем статус BUSY
     {
          std::unique_lock<std::mutex> locker(g_lock_statuses);
          g_statuses[id] = BUSY;
     }
     // Выполняем работу
     data[id] += command * id;

     // Устанавливаем статус READY
     {
          std::unique_lock<std::mutex> locker(g_lock_statuses);
          g_statuses[id] = READY;
     }

}

void loggerFunc()
{
     int statuses_size = g_statuses.size();

     // стартовое сообщение
     {
          std::unique_lock<std::mutex> locker(g_lockprint);
          std::cout << "[logger]\trunning..." << std::endl;
     }

     // Вывод статусов потоков
     {
          std::unique_lock<std::mutex> locker(g_lockprint);
          for (int i = 0; i < statuses_size; i++)
          {
               std::cout << g_statuses[i] << " ";
          }
          std::cout << std::endl;
     }

     // Ожидаем состояния STARTED у всех рабочих потоков
     for (int i = 0; i < statuses_size; i++)
     {
          while (g_statuses[i] != STARTED)
          {
               std::this_thread::yield();
          }
     }

     // Вывод статусов потоков
     {
          std::unique_lock<std::mutex> locker(g_lockprint);
          for (int i = 0; i < statuses_size; i++)
          {
               std::cout << g_statuses[i] << " ";
          }
          std::cout << std::endl;
     }

     // Формируем команду для обработчиков
     command = 10;
     is_can_work = true;

     // Ожидаем состояния READY у всех рабочих потоков
     for (int i = 0; i < statuses_size; i++)
     {
          while (g_statuses[i] != READY)
          {
               std::this_thread::yield();
          }
     }

     // Вывод статусов потоков
     {
          std::unique_lock<std::mutex> locker(g_lockprint);
          for (int i = 0; i < statuses_size; i++)
          {
               std::cout << g_statuses[i] << " ";
          }
          std::cout << std::endl;
     }

     // Вывод результатов работы потоков
     {
          int data_size = statuses_size;
          std::unique_lock<std::mutex> locker(g_lockprint);
          for (int i = 0; i < data_size; i++)
          {
               std::cout << data[i] << " ";
          }
          std::cout << std::endl;
     }     
}

int main()
{
     int numThreads;
     std::cout << "Input Threads Number: ";
     std::cin >> numThreads;
     g_statuses.resize(numThreads); // Изменяем размер вектора статусов
     data = new double[numThreads]; // Создаём и инициализируем массив data
     for (int i = 0; i < numThreads; i++)
          data[i] = 0;

     // инициализация генератора псевдо-случайных чисел
     std::mt19937 generator((unsigned int)std::chrono::system_clock::now().time_since_epoch().count());
     // запуск регистратора
     std::thread loggerThread(loggerFunc);
     // запуск рабочих потоков
     std::vector<std::thread> threads;

     for (int i = 0; i < numThreads; ++i)
          threads.push_back(std::thread(workerFunc, i, std::ref(generator)));
     for (auto &t : threads)
          t.join();     
     loggerThread.join();
     return 0;
}