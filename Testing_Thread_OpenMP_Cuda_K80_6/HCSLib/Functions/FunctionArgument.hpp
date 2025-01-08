#pragma once

#include "../Functions/FunctionDataType.hpp"
#include "../CommonHelpers/PrintParams.hpp"

/// @brief Аргументы функции
struct FunctionArgument
{
    FunctionDataType dataType;
    void* data = nullptr;

    FunctionArgument()
    {
        dataType = FunctionDataType::fdt_void;
        Print(PrintParams{});
    }

    template<typename T>
    FunctionArgument(T argument)
    {
        if(typeid(T)==typeid(int))
            dataType = FunctionDataType::fdt_int;
        else if(typeid(T)==typeid(float))
            dataType = FunctionDataType::fdt_float;
        else if(typeid(T)==typeid(float*))
            dataType = FunctionDataType::fdt_ptr_float;
        else if(typeid(T)==typeid(double))
            dataType = FunctionDataType::fdt_double;
        else if(typeid(T)==typeid(double*))
            dataType = FunctionDataType::fdt_ptr_double;
        else if(typeid(T)==typeid(size_t))
            dataType = FunctionDataType::fdt_ull;
        else
        {
            std::cout << "\nError in FunctionArgument constructor!\n";
            throw std::runtime_error("Type not recognized!");
        }

        auto ptr = new T;
        *ptr = argument;
        data = (void*)ptr;
        Print(PrintParams{});
    }

    ~FunctionArgument()
    {
        switch (dataType)
        {
        case FunctionDataType::fdt_float:
            delete (float*)data;
            break;
        
        default:
            break;
        }
    }

    template<typename T>
    T GetValue()
    {
        return *(T*)data;
    }

    void Print(PrintParams pp)
    {
        std::cout << pp.startMes;

        std::cout << dataType;
        std::cout << pp.splitter;
        std::cout << data;
        std::cout << pp.splitter;
        switch (dataType)
        {
        case FunctionDataType::fdt_void:
            std::cout << "void";
            break;
        case FunctionDataType::fdt_int:
            std::cout << GetValue<int>();
            break;
        case FunctionDataType::fdt_float:
            std::cout << GetValue<float>();
            break;
        case FunctionDataType::fdt_ptr_float:
            std::cout << GetValue<float*>();
            break;
        case FunctionDataType::fdt_double:
            std::cout << GetValue<double>();
            break;
        case FunctionDataType::fdt_ptr_double:
            std::cout << GetValue<double*>();
            break;
        case FunctionDataType::fdt_ull:
            std::cout << GetValue<size_t>();
            break;
            
        default:
            std::cout << "\nError in FunctionArgument::Print()! Type not found!\n" << std::endl;
            throw std::runtime_error("Add type in switch of FunctionArgument::Print()");
            break;
        }
        

        std::cout << pp.endMes;
        if(pp.isEndl)
            std::cout << std::endl;
    }
};

std::ostream& operator<<(std::ostream& os, FunctionArgument arg)
{
    switch (arg.dataType)
    {
    case FunctionDataType::fdt_void:
        os << "void";
        break;
    case FunctionDataType::fdt_float:
        os << arg.GetValue<float>();
        break;
    case FunctionDataType::fdt_ptr_float:
        os << arg.GetValue<float*>();
        break;
    case FunctionDataType::fdt_double:
        os << arg.GetValue<double>();
        break;
    case FunctionDataType::fdt_ptr_double:
        os << arg.GetValue<double*>();
        break;
    case FunctionDataType::fdt_ull:
        os << arg.GetValue<size_t>();
        break;
        
    default:
        break;
    }

    os << "(" << arg.dataType << ")";

    return os;
}