#pragma once

// Параметры запуска функции обработки массива на GPU
template<typename T>
struct ArrayGpuProcessingParams
{
    unsigned deviceId;
    T* dev_arr;
    size_t indStart;
    size_t indEnd;
    unsigned blocksNum;
    unsigned threadsNum;

    void Print()
    {
        std::cout << "[";
        std::cout << deviceId << "; ";
        std::cout << dev_arr << "; ";
        std::cout << indStart << "; ";
        std::cout << indEnd << "; ";
        std::cout << blocksNum << "; ";
        std::cout << threadsNum << "; ";
        std::cout << "]";
        std::cout << std::endl;
    }
};