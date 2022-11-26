#include "FloatArray.h"

// Создаёт массив float в динамической памяти и возвращает на него указатель
float* FloatArray_RAM_Create(int numElements)
{
    float* arr = (float*)malloc(numElements * sizeof(float));
    return arr;
}