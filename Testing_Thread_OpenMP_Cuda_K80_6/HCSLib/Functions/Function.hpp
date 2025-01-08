#pragma once

#include <iostream>
#include <functional>

#include "../CommonHelpers/PrintParams.hpp"
#include "FunctionDataType.hpp"
#include "FunctionDataTypes.hpp"

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
            argumentsTypes.Add(FunctionDataType::fdt_float);
            //argumentsType = FunctionArgumentsType::arg_pfloat_ull_ull;
        }
        else if(typeid(T) == typeid(double))
        {
            returnType = FunctionDataType::fdt_double;
            argumentsTypes.Add(FunctionDataType::fdt_double);
            //argumentsType = FunctionArgumentsType::arg_pdouble_ull_ull;
        }
        else
        {
            throw std::runtime_error("Function argument type not realized");
        }

        argumentsTypes.Add(FunctionDataType::fdt_ull);
        argumentsTypes.Add(FunctionDataType::fdt_ull);
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
        std::cout << "argumentsTypes" << pp.splitterKeyValue << argumentsTypes;
        
        std::cout << pp.endMes;
        if(pp.isEndl)
            std::cout << std::endl;
    }
};