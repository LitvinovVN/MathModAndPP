#pragma once

/// @brief Перечисление типов аргументов функций
enum class FunctionArgumentsType
{
    arg_void,// аргументы отсутствуют
    arg_pfloat_ull_ull,// (float*, size_t, size_t)
    arg_pdouble_ull_ull// (double*, size_t, size_t)
};

std::ostream& operator<<(std::ostream& os, FunctionArgumentsType argt)
{
    switch (argt)
    {
    case FunctionArgumentsType::arg_void:
        os << "()";
        break;
    case FunctionArgumentsType::arg_pfloat_ull_ull:
        os << "(float*, size_t, size_t)";
        break;
    case FunctionArgumentsType::arg_pdouble_ull_ull:
        os << "(double*, size_t, size_t)";
        break;
        
    default:
        break;
    }

    return os;
}