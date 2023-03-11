// Запуск:
// nvcc solver.cpp -o ../bin/solver
// ../bin/solver

#include <condition_variable>
#include <iostream>
#include <thread>
#include <mutex>
#include "helpers/config.cpp"
#include "helpers/node_config.cpp"

///// Глобальные переменные /////
Config g_config;// Конфигурация проекта
NodeConfig g_node_config;// Конфигурация узла
/////////////////////////////////



void init()
{
     // Считываем параметры конфигурации из файла
     g_config = ReadConfig("../config/main.conf");     
     // Выводим параметры конфигурации в консоль
     PrintConfig(g_config);

     // Считываем из файла параметры конфигурации узла
     g_node_config = ReadNodeConfig(g_config.node_id);
     // Выводим параметры конфигурации узла в консоль
     PrintNodeConfig(g_node_config);

     // Считываем массивы данных и размещаем их в памяти
}

int main()
{
     init();// Инициализация приложения

     //ptm_parallel();


     return 0;
}