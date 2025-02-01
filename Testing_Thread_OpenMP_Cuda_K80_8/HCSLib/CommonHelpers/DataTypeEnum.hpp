#pragma once

#include <iostream>

enum class DataTypeEnum
{
    dt_void,
    dt_int,
    dt_ull,
    dt_float,
    dt_ptr_float,
    dt_double,
    dt_ptr_double
};

std::ostream& operator<<(std::ostream& os, DataTypeEnum dt)
{
    switch (dt)
    {
    case DataTypeEnum::dt_void:
        os << "void";
        break;
    case DataTypeEnum::dt_int:
        os << "int";
        break;
    case DataTypeEnum::dt_ull:
        os << "ull";
        break;
    case DataTypeEnum::dt_float:
        os << "float";
        break;
    case DataTypeEnum::dt_ptr_float:
        os << "float*";
        break;
    case DataTypeEnum::dt_double:
        os << "double";
        break;
    case DataTypeEnum::dt_ptr_double:
        os << "double*";
        break;
        
    default:
        break;
    }

    return os;
}