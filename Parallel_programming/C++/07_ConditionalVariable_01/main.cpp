// Задача 07_01. condition_variable
// Источник: https://betterprogramming.pub/writing-framework-for-inter-thread-message-passing-in-c-256b5308a471
// Запуск:
// g++ main.cpp -std=c++11 -pthread -o app
// nvcc main.cpp -o app
// ./app

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

int count = 0;
std::mutex count_mutex;
std::condition_variable cv;

void producer(void)
{
     for (int i = 0; i <= 10; i++)
     {
          std::unique_lock<std::mutex> lk(count_mutex);
          count = i;
          cv.notify_one();
          lk.unlock();
          std::this_thread::sleep_for(std::chrono::milliseconds(1000));
     }
}

void consumer(void)
{
     for (int i = 0; i < 10; i++)
     {
          std::unique_lock<std::mutex> lk(count_mutex);
          cv.wait(lk, []()
                  { return (count != 0); });
          if (count >= 10)
               break;
          std::cout << count << "\n";
          count = 0;
     }
}

int main()
{
     std::thread t2(consumer);
     std::thread t1(producer);

     t1.join();
     t2.join();

     return 0;
}