#pragma once

#include <string>

// Структура для хранения параметров запуска теста производительности
struct TestConfig
{
    std::string dir = "results";
    std::string fileCaption = "test";
    size_t lengthStart = 1000000;
    size_t lengthEnd   = 2000000;
    size_t lengthStep  = 0;
    size_t lengthMult  = 2;
    // Количество потоков CPU
    unsigned cpuThreadsMin = 1;
    unsigned cpuThreadsMax = 8;
    // Количество блоков GPU
    unsigned gpuBlocksMin = 1;
    unsigned gpuBlocksMax = 20;
    // Количество потоков GPU
    unsigned gpuThreadsMin = 1;
    unsigned gpuThreadsMax = 50;
    // Количество повторов
    unsigned iterNum = 10;    
};