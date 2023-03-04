/* Задача 09. Основное приложение запускает поток-слушатель, дожидается его статуса READY,
 и передаёт ему команды пользователя, введённые в консоль
 */

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
     STOPPED,
     INIT,
     STARTED,
     BUSY,
     READY,     
     ADD,
     SUB,
     VALUE
};

bool is_can_work = false; // Флаг для рабочих потоков, сигнализирующий о возможности приступать к работе
bool is_exit = false;     // Флаг для рабочих потоков, сигнализирующий о том, что все команды обработаны и нужно завершать работу
STATUS listener_status = STOPPED;
int value = 0;

std::mutex g_lockprint;

void listenerFunc()
{
     // стартовое сообщение
     {
          std::unique_lock<std::mutex> locker(g_lockprint);
          std::cout << "[listener]\tstarting..." << std::endl;
     }     
     std::this_thread::sleep_for(std::chrono::seconds(5));
     // стартовое сообщение
     {
          std::unique_lock<std::mutex> locker(g_lockprint);
          std::cout << "[listener]\trunning..." << std::endl;
     } 

     listener_status = READY;

     while (1)
     {
          if (listener_status == STOPPED)
               break;

          if (listener_status == ADD)
          {
               value++;
               listener_status = READY;
          }
          else if (listener_status == SUB)
          {
               value--;
               listener_status = READY;
          }
          else if (listener_status == VALUE)
          {
               std::unique_lock<std::mutex> locker(g_lockprint);
               std::cout << "[listener]\tvalue = " << value << std::endl;
               listener_status = READY;
          }

          std::this_thread::yield();
     }

     // Сообщение о завершении потока-слушателя
     {
          std::unique_lock<std::mutex> locker(g_lockprint);
          std::cout << "[listener]\tstopped..." << std::endl;
     }
}

int main()
{
     std::cout << "Starting Listener..." << std::endl;
        

     // запуск регистратора
     std::thread listenerThread(listenerFunc);

     // Ожидание готовности потока-слушателя
     while (listener_status == STOPPED)
     {
          std::this_thread::yield();
     }

     std::string command = "";
     while (command != "q")
     {
          // стартовое сообщение
          {
               std::unique_lock<std::mutex> locker(g_lockprint);
               std::cout << "> ";               
          }
          std::cin >> command;

          if (listener_status == BUSY)
          {
               // стартовое сообщение
               {
                    std::unique_lock<std::mutex> locker(g_lockprint);
                    std::cout << "Listener is busy, please wait..." << std::endl;
                    continue;
               }
          }

          if (listener_status == STOPPED)
          {
               // стартовое сообщение
               {
                    std::unique_lock<std::mutex> locker(g_lockprint);
                    std::cout << "Listener is stopped..." << std::endl;
                    continue;
               }
          }

          if (command == "stop")
               listener_status = STOPPED;
          else if (command == "add")
               listener_status = ADD;
          else if (command == "sub")
               listener_status = SUB;
          else if (command == "val")
               listener_status = VALUE;
          else
               {
                    std::unique_lock<std::mutex> locker(g_lockprint);
                    std::cout << "Command is not recognized! Enter add, sub, val or stop commands" << std::endl;
               }
     }
     listener_status = STOPPED;
     listenerThread.join();

     return 0;
}