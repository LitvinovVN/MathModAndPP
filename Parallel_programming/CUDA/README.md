# Введение в программирование графических ускорителей с использованием технологии NVIDIA CUDA

## Функции для работы с динамическим массивом, содержищим элементы типа float

| Тип возвращаемого значения | Функция | Аргументы  | Описание |
| ------------- |:-------------:| -----:|---|
| float* | FloatArray_RAM_Create | int numElements | Создаёт массив float, содержащий numElements элементов, в динамической памяти и возвращает на него указатель |
| float* | FloatArray_GPU_Create | int numElements | Создаёт массив float, содержащий numElements элементов, в видеопамяти GPU и возвращает на него указатель |
| __host__ __device__ void | FloatArray_InitByIndexes | float* fArray, int numElements | Инициализирует элементы массива fArray их индексами |
| __host__ __device__ void |FloatArray_Print | float* fArray, int numElements | Выводит элементы массива fArray в консоль |


## Функции для работы со структурой Array1D (одномерный массив) 

| Тип возвращаемого значения | Функция | Аргументы  | Описание |
| ------------- |:-------------:| -----:|---|
| Array1D | Array1D_RAM_Create | int numElements | Создаёт структуру типа Array1D, содержащую numElements элементов, в ОЗУ |