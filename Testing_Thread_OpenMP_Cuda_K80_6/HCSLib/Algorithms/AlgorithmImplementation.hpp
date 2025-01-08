#pragma once

#include <iostream>
#include "../CommonHelpers/PrintParams.hpp"
#include "../Functions/Function.hpp"
#include "AlgorithmImplementationExecParams.hpp"
#include "../AlgTestingResults/AlgTestingResult.hpp"

/// @brief Класс реализации алгоритма
/// (сопоставляет УИД алгоритма с функцией реализации)
class AlgorithmImplementation
{
    unsigned id{};// УИД сопоставления
    unsigned algorithmId{};// УИД алгоритма
    // Объект функции, реализующей алгоритм
    Function function{};
    // Описание
    std::string description{};

public:
    AlgorithmImplementation()
    {}

    AlgorithmImplementation(unsigned id,
        unsigned algorithmId,
        std::string description,
        Function function) :
            id(id),
            algorithmId(algorithmId),
            function(function),
            description(description)
    {}

    void Print(PrintParams pp)
    {
        std::cout << pp.startMes;
        
        std::cout << "id" << pp.splitterKeyValue << id;
        std::cout << pp.splitter;
        std::cout << "algorithmId" << pp.splitterKeyValue << algorithmId;
        std::cout << pp.splitter;
        std::cout << "description" << pp.splitterKeyValue << description;
        std::cout << pp.splitter;
        std::cout << "function" << pp.splitterKeyValue;
        function.Print(pp);              
        
        std::cout << pp.endMes;
        if(pp.isEndl)
            std::cout << std::endl;
    }

    /// @brief Возвращает УИД сопоставления алгоритма и его реализации
    /// @return УИД сопоставления (id)
    unsigned GetId() const
    {
        return id;
    }

    /// @brief Возвращает объект функции
    /// @return Объект типа Function
    Function GetFunction()
    {
        return function;
    }

    AlgTestingResult Exec(AlgorithmImplementationExecParams params)
    {
        AlgTestingResult res = function.Exec(params);
        std::cout << "\n\nAlgorithmImplementation::Exec str 69\n\n";
        return res;
    }
};