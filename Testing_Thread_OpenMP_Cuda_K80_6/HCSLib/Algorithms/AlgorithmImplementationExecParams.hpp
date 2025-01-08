#pragma once

#include <vector>

/// @brief Параметры запуска реализации алгоритма
struct AlgorithmImplementationExecParams
{
    // Аргументы функции, реализующей алгоритм
    FunctionArguments functionArguments;
    // Количество запусков функции
    unsigned iterNumber{100};
};