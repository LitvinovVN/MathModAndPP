#pragma once

#include <iostream>

/// @brief Перечисление типов данных для описания прототипов функций
enum class FunctionDataType
{
    fdt_void,
    fdt_float,
    fdt_ptr_float,
    fdt_double,
    fdt_ptr_double,
    fdt_ull
};

std::ostream& operator<<(std::ostream& os, FunctionDataType fdt)
{
    switch (fdt)
    {
    case FunctionDataType::fdt_void:
        os << "void";
        break;
    case FunctionDataType::fdt_float:
        os << "float";
        break;
    case FunctionDataType::fdt_ptr_float:
        os << "float*";
        break;
    case FunctionDataType::fdt_double:
        os << "double";
        break;
    case FunctionDataType::fdt_ptr_double:
        os << "double*";
        break;
    case FunctionDataType::fdt_ull:
        os << "size_t";
        break;
        
    default:
        break;
    }

    return os;
}