#include <iostream>
#include <fstream>

// Структура "Конфигурация проекта"
struct Config
{
     int nodes_number = -1;
     int node_id = -1;
};

void PrintConfig(Config config)
{
     std::cout << "----- Config ----- " << std::endl;
     std::cout << "nodes_number = " << config.nodes_number << std::endl;
     std::cout << "node_id = " << config.node_id << std::endl;
     std::cout << "------------------ " << std::endl;
}

Config ReadConfig(std::string path)
{
     Config config;
     std::cout << "Reading configuration... ";

     //std::ifstream fs("../config/main.conf");
     std::ifstream fs(path);

     if (!fs) 
     {
          std::cout << "[ERROR] File not opened!\n\n";
          return config;
     }
     
     std::string parameter;
     
     while(!fs.eof())
     {
          fs >> parameter;

          if(parameter == "nodes_number")
          {
               fs >> config.nodes_number;
          }
          else if(parameter == "node_id")
          {
               fs >> config.node_id;
          }
     }

     std::cout << "OK" << std::endl;
     return config;
}