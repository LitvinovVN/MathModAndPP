#pragma once

#include <iostream>

#include "../Tasks/TaskGroup.hpp"
#include "../Tasks/Task.hpp"
#include "../Tasks/TaskDimensions.hpp"
#include "AlgorithmType.hpp"
#include "DataLocation.hpp"
#include "../PrintParams.hpp"

/// @brief Сведения об алгоритме
struct Algorithm
{
    // УИД алгоритма
    unsigned id = 0;
    // Группа задач
    TaskGroup taskGroup;
    // Задача
    Task task;
    // Размерности задачи
    TaskDimensions taskDimensions {};
    // Длина типа данных, используемая в алгоритме (float: 4; double: 8)
    unsigned dataTypeLength = 0;
    // Тип алгоритма (послед., параллельный и пр.)
    AlgorithmType algorithmType;
    // Место расположения исходных данных
    DataLocation dataLocationInput;
    // Место расположения результатов
    DataLocation dataLocationOutput;

    void Print(PrintParams pp)
    {
        std::cout << pp.startMes;

        std::cout << "id"                 << pp.splitterKeyValue << id << pp.splitter;
        std::cout << "taskGroup"          << pp.splitterKeyValue << taskGroup << pp.splitter;
        std::cout << "task"               << pp.splitterKeyValue << task << pp.splitter;
        std::cout << "taskDimensions"     << pp.splitterKeyValue;
            taskDimensions.Print(pp);
            std::cout << pp.splitter;
        std::cout << "dataTypeLength"     << pp.splitterKeyValue << dataTypeLength << pp.splitter;
        std::cout << "algorithmType"      << pp.splitterKeyValue << algorithmType << pp.splitter;
        std::cout << "dataLocationInput"  << pp.splitterKeyValue << dataLocationInput << pp.splitter;
        std::cout << "dataLocationOutput" << pp.splitterKeyValue << dataLocationOutput << pp.splitter;

        std::cout << pp.endMes;
        if(pp.isEndl)
            std::cout << std::endl;
    }
};