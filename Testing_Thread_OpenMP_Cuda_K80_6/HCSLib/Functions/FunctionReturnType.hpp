#pragma once

/// @brief Перечисление типов возвращаемых функциями значений
enum class FunctionReturnType
{
    rt_void,
    rt_float,
    rt_double
};

std::ostream& operator<<(std::ostream& os, FunctionReturnType tg)
{
    switch (tg)
    {
    case FunctionReturnType::rt_void:
        os << "void";
        break;
    case FunctionReturnType::rt_float:
        os << "float";
        break;
    case FunctionReturnType::rt_double:
        os << "double";
        break;
        
    default:
        break;
    }

    return os;
}