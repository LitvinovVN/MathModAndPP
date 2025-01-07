#pragma once

#include "../Functions/FunctionDataType.hpp"

/// @brief Аргументы функции
struct FunctionArgument
{
    FunctionDataType dataType;
    void* data = nullptr;

    FunctionArgument(float argument)
    {
        dataType = FunctionDataType::fdt_float;
        auto ptr = new float;
        *ptr = argument;
        data = (void*)ptr;
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
        os << arg.GetValue<double*>();;
        break;
    case FunctionDataType::fdt_ull:
        os << arg.GetValue<size_t>();;
        break;
        
    default:
        break;
    }

    os << "(" << arg.dataType << ")";

    return os;
}