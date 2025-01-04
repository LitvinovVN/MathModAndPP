#pragma once

#include <iostream>
#include <functional>

#include "../CommonHelpers/PrintParams.hpp"
#include "FunctionReturnType.hpp"
#include "FunctionArgumentsType.hpp"

class Function
{
    // Указатель на функцию, реализующую алгоритм
    void* func = nullptr;
    // Тип возвращаемого значения
    FunctionReturnType returnType;
    // Тип аргументов функции
    FunctionArgumentsType argumentsType;

public:
    Function()
    {}

    template<typename T>
    Function(T(*function)(T*, size_t, size_t))
    {
        func = (void*)function;        
        
        if(typeid(T) == typeid(float))
        {
            returnType = FunctionReturnType::rt_float;
            argumentsType = FunctionArgumentsType::arg_pfloat_ull_ull;
        }
        else if(typeid(T) == typeid(double))
        {
            returnType = FunctionReturnType::rt_double;
            argumentsType = FunctionArgumentsType::arg_pdouble_ull_ull;
        }
        else
        {
            throw std::runtime_error("Function argument type not realized");
        }
    }
    /*Function(double(*function)(double*, size_t, size_t))
    {
        func = (void*)function;
    }

    Function(float(*function)(float*, size_t, size_t))
    {
        func = (void*)function;
    }*/

    void Print(PrintParams pp)
    {
        std::cout << pp.startMes;
        
        std::cout << "func" << pp.splitterKeyValue << func;
        std::cout << pp.splitter;
        std::cout << "returnType" << pp.splitterKeyValue << returnType;
        std::cout << pp.splitter;
        std::cout << "argumentsType" << pp.splitterKeyValue << argumentsType;
        
        std::cout << pp.endMes;
        if(pp.isEndl)
            std::cout << std::endl;
    }
};