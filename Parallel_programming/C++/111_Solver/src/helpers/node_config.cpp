#include <iostream>
#include <fstream>
#include <string>

// Структура "Конфигурация узла"
struct NodeConfig
{
     int cpu_threads_num = -1;// Количество рабочих потоков CPU
};

void PrintNodeConfig(NodeConfig config)
{
     std::cout << "----- Node Configuration ----- " << std::endl;
     std::cout << "cpu_threads_num = " << config.cpu_threads_num << std::endl;     
     std::cout << "------------------------------ " << std::endl;
}

NodeConfig ReadNodeConfig(int node_id)
{
    NodeConfig node_config;

    std::cout << "Reading node configuration... ";

     std::ifstream fs("../config/node_" + std::to_string(node_id) + "/threads.conf");

     if (!fs) 
     {
          std::cout << "[ERROR] File not opened!\n\n";
          return node_config;
     }
     
     std::string parameter;
     
     while(!fs.eof())
     {
          fs >> parameter;

          if(parameter == "cpu_threads_num")
          {
               fs >> node_config.cpu_threads_num;
          }
          
     }

     std::cout << "OK" << std::endl;

     return node_config;
}