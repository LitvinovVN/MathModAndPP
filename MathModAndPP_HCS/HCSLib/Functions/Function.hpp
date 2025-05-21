#pragma once

#include <iostream>
#include <functional>

#include "../CommonHelpers/PrintParams.hpp"
#include "FunctionDataType.hpp"
#include "FunctionDataTypes.hpp"
#include "../Algorithms/AlgorithmImplementationExecParams.hpp"
#include "../AlgTestingResults/AlgTestingResult.hpp"

class Function
{
    // Указатель на функцию, реализующую алгоритм
    void* func = nullptr;
    // Тип возвращаемого значения
    FunctionDataType returnType;
    // Список типов аргументов функции
    FunctionDataTypes argumentsTypes;

public:
    Function()
    {}

    template<typename T>
    Function(T(*function)(T*, size_t, size_t))
    {
        func = (void*)function;        
        
        if(typeid(T) == typeid(float))
        {
            returnType = FunctionDataType::fdt_float;
            argumentsTypes.Add(FunctionDataType::fdt_ptr_float);
        }
        else if(typeid(T) == typeid(double))
        {
            returnType = FunctionDataType::fdt_double;
            argumentsTypes.Add(FunctionDataType::fdt_ptr_double);
        }
        else
        {
            throw std::runtime_error("Function argument type not realized");
        }

        argumentsTypes.Add(FunctionDataType::fdt_ull);
        argumentsTypes.Add(FunctionDataType::fdt_ull);
    }
    
    /// @brief Возвращает количество аргументов функции
    /// @return 
    unsigned GetArgumentsTypesCount() const
    {
        return argumentsTypes.Count();
    }

    /// @brief Проверка типов аргументов функции
    /// @return 
    bool CheckArgumentsTypes(FunctionDataTypes argsTypes) const
    {
        if(GetArgumentsTypesCount() != argsTypes.Count())
            return false;
        
        for(unsigned i{0}; i < GetArgumentsTypesCount(); i++)
        {
            if(argumentsTypes[i] != argsTypes[i])
                return false;
        }
        
        return true;
    }

    AlgTestingResult Exec(AlgorithmImplementationExecParams params)
    {
        FunctionArguments functionArguments = params.functionArguments;
        if(argumentsTypes.Count()==3)
        {
            if(argumentsTypes[0] == FunctionDataType::fdt_ptr_float
                && argumentsTypes[1] == FunctionDataType::fdt_ull
                && argumentsTypes[2] == FunctionDataType::fdt_ull)
            {
                // Преобразовываем указатель func к нужному виду
                // float* (*func_ptr)(size_t, size_t);
                auto func_ptr = (float (*)(float*, size_t, size_t))func;
                float* arg0 = functionArguments.GetArgumentValue<float*>(0);
                std::cout << "arg0 = " << arg0 << std::endl;
                size_t arg1 = functionArguments.GetArgumentValue<size_t>(1);
                std::cout << "arg1 = " << arg1 << std::endl;
                size_t arg2 = functionArguments.GetArgumentValue<size_t>(2);
                std::cout << "arg2 = " << arg2 << std::endl;
                
                std::vector<FuncResult<float>> results;
                for(unsigned i{0}; i < params.iterNumber; i++)
                {                    
                    bool calcStatus = true;
                    auto start = high_resolution_clock::now();
                    float result_f = func_ptr(arg0, arg1, arg2);                    
                    auto stop = high_resolution_clock::now();
                    std::cout << "!!! result_f = " << result_f << std::endl;

                    auto duration = duration_cast<microseconds>(stop - start);        
                    auto t = duration.count();

                    FuncResult<float> funcResF(calcStatus, result_f, t);
                    results.push_back(funcResF);
                }
                //
                CalculationStatistics stats{results};
                AlgTestingResult algTestingResult;
                algTestingResult.calculationStatistics = stats;
                algTestingResult.Print();
                return algTestingResult;
            }
            else
            {
                std::cout << "\n\nNot realized!\n";
                throw std::runtime_error("Error! Function::Exec(...) argumentsTypes.Count()==3 types combination not realized!");
            }
        }
        else
        {
            std::cout << "\n\nNot realized!\n";
            throw std::runtime_error("Error! Function::Exec(...) argumentsTypes.Count() not realized!");
        }

        //FuncResult<float> res;

        //return res;
    }

    void Print(PrintParams pp) const
    {
        std::cout << pp.startMes;
        
        std::cout << "func" << pp.splitterKeyValue << func;
        std::cout << pp.splitter;
        std::cout << "returnType" << pp.splitterKeyValue << returnType;
        std::cout << pp.splitter;
        std::cout << "argumentsTypes" << pp.splitterKeyValue << argumentsTypes;
        
        std::cout << pp.endMes;
        if(pp.isEndl)
            std::cout << std::endl;
    }
};