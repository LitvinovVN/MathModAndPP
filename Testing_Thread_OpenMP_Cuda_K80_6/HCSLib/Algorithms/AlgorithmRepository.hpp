#pragma once

#include <iostream>
#include <map>

#include "Algorithm.hpp"
#include "../CommonHelpers/PrintParams.hpp"

/// @brief Репозиторий алгоритмов
class AlgorithmRepository
{
    std::map<unsigned, Algorithm> data;

    /// @brief Инициализация репозитория алгоритмов
    void Init();
public:
    AlgorithmRepository()
    {
        Init();
    }

    void Print(PrintParams pp)
    {
        std::cout << "void AlgorithmRepository::Print();\n";

        for(auto& element : data)
        {            
            element.second.Print(pp);
        }

        if (pp.isEndl)
            std::cout << std::endl;
    }

    /// @brief Проверяет наличие алгоритма с указанным УИД
    /// @return 
    bool IsExists(unsigned id)
    {
        return data.count(id) > 0;
    }

    /// @brief Возвращает алгоритм по УИД
    /// @param id 
    /// @return 
    Algorithm Get(unsigned id)
    {
        return data[id];
    }

    /// @brief Запрашивает у пользователя id алгоритма и выводит в консоль сведения о нём
    void Get()
    {
        unsigned id = ConsoleHelper::GetUnsignedIntFromUser("Enter algorithm id: ");
        Algorithm alg = Get(id);
        if(alg.id == id && alg.id != 0)
            alg.Print(PrintParams{});
        else
            std::cout << "Algorithm not found!" << std::endl;
    }

    /// @brief Добавляет алгоритм в репозиторий
    /// @param alg 
    /// @return Результат (true - добавлен, false - не добавлен)
    bool Add(Algorithm alg)
    {
        if (alg.id == 0 || IsExists(alg.id))
            return false;

        data[alg.id] = alg;
        return true;
    }

};

///////////////////////////////////////////////////////

void AlgorithmRepository::Init()
{
    Algorithm alg1;
    alg1.id                 = 1;
    alg1.taskGroup          = TaskGroup::Vector;
    alg1.task               = Task::Sum;
    alg1.taskDimensions     = TaskDimensions{1};
    alg1.dataTypeLength     = sizeof(double);
    alg1.algorithmType      = AlgorithmType::SeqCpu;
    alg1.dataLocationInput  = DataLocation::Ram;
    alg1.dataLocationOutput = DataLocation::Ram;
    Add(alg1);

    Algorithm alg2;
    alg2.id = 2;
    alg1.taskGroup          = TaskGroup::Vector;
    alg1.task               = Task::Sum;
    alg1.taskDimensions     = TaskDimensions{1};
    alg1.dataTypeLength     = sizeof(double);
    alg2.algorithmType      = AlgorithmType::SeqGpuCuda;
    alg2.dataLocationInput  = DataLocation::Gpu;
    alg2.dataLocationOutput = DataLocation::Gpu;
    Add(alg2);
}