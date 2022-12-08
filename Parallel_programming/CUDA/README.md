# Введение в программирование графических ускорителей с использованием технологии NVIDIA CUDA


## Вспомогательные функции для организации ввода-вывода данных (utils.c)

| Дополнительные атрибуты функции | Тип возвращаемого значения | Функция | Аргументы  | Описание |
| ------------- | ------------- |:-------------:| -----:|---|
| | int | IntNumber_Input | const char message[] | Получает строку-сообщение message и выводит её в консоль. Считывает целое число, введённое пользователем и возвращает его. |



## Функции для работы с динамическим массивом, содержищим элементы типа float (floatArray.c)

| Дополнительные атрибуты функции | Тип возвращаемого значения | Функция | Аргументы  | Описание |
| ------------- | ------------- |:-------------:| -----:|---|
| | float* | FloatArray_RAM_Create | int numElements | Создаёт массив float, содержащий numElements элементов, в динамической памяти и возвращает на него указатель |
| | float* | FloatArray_GPU_Create | int numElements | Создаёт массив float, содержащий numElements элементов, в видеопамяти GPU и возвращает на него указатель |
| \_\_host\_\_ \_\_device\_\_ | void | FloatArray_InitByIndexes | float* fArray, int numElements | Инициализирует элементы массива fArray их индексами |
| \_\_host\_\_ \_\_device\_\_ | void |FloatArray_Print | float* fArray, int numElements | Выводит элементы массива fArray в консоль |
| | void | FloatArray_CopyFromRAMtoGPU | float* fArray_RAM, float* fArray_GPU, int numElements | Копирует элементы массива fArray_RAM, расположенного в ОЗУ, в массив fArray_GPU, расположенный в видеопамяти. |
| | void | FloatArray_CopyFromGPUtoRAM | float* fArray_GPU, float* fArray_RAM, int numElements | Копирует элементы массива fArray_GPU, расположенного в видеопамяти, в массив fArray_RAM, расположенный в ОЗУ. |


***
## Структура Array1D "Одномерный массив" (Array1D.c)

```C
// Создаем структуру "Одномерный массив"
struct array_t
{
    int size;
    float* data;
};

// Определяем новый тип
typedef struct array_t Array1D;
```

***
## Функции для работы со структурой Array1D (одномерный массив) 

| Дополнительные атрибуты функции | Тип возвращаемого значения | Функция | Аргументы  | Описание |
| ------------- | ------------- |:-------------:| -----:|---|
| | Array1D | Array1D_RAM_Create | int numElements | Создаёт структуру типа Array1D, содержащую numElements элементов, в ОЗУ |
| | Array1D* | Array1D_RAM_Create_From_Array1D_GPU | Array1D* array1D_GPU | Создаёт структуру типа Array1D в ОЗУ как копию структуры Array1D, расположенной в GPU, и возвращает на неё указатель |
| | Array1D* | Array1D_GPU_Create | int numElements | Создаёт структуру типа Array1D, содержащую numElements элементов, в видеопамяти и возвращает на него указатель |
| | Array1D* | Array1DArray_GPU_Create | int numElements | Создаёт массив структур Array1D в GPU, содержащий numElements элементов, и возвращает на него указатель |
| | Array1D* | Array1D_GPU_Create_From_Array1D_RAM | Array1D array1D_RAM | Создаёт структуру типа Array1D в GPU как копию структуры array1D_RAM, расположенной в RAM, и возвращает на неё указатель |
| | Array1D* | Array1D_GPU_Array_Create_From_Array1D_RAM_Array | Array1D* array1D_RAM_Array, int numElements | Создаёт массив структур Array1D в GPU, содержащий numElements элементов, из массива структур Array1D ОЗУ |
| --- | Array1D* | Array1D_RAM_Array_Create_From_Array1D_GPU_Array | Array1D* array1D_GPU_Array, int numElements | Создаёт массив структур Array1D в RAM, содержащий numElements элементов, из массива структур Array1D GPU |
| | void | Array1D_RAM_InitByIndexes | Array1D arr | Инициализирует элементы массива data структуры Array1D их индексами |
| \_\_host\_\_ \_\_device\_\_ | void | Array1D_Print | Array1D* array1D | Выводит элементы массива array1D->data в консоль |
| | void | Array1D_RAM_Destruct | Array1D* array1D_RAM | Освобождает оперативную память, выделенную под структуру array1D_RAM |
| | void | Array1D_GPU_Destruct | Array1D* array1D_GPU | Освобождает видеопамять, выделенную под структуру array1D_GPU |




***
## Структура Arrays1D "Одномерный массив элементов типа Array1D" (Arrays1D.c)

```C
// Создаем структуру "Одномерный массив"
struct arrays_t {
    int numElements;
    Array1D* data;
};

//Определяем новый тип
typedef struct arrays_t Arrays1D;
```

***
## Функции для работы со структурой Arrays1D (одномерный массив элементов типа Array1D) 

| Дополнительные атрибуты функции | Тип возвращаемого значения | Функция | Аргументы  | Описание |
| ------------- | ------------- |:-------------:| -----:|---|
| | Arrays1D | Arrays1D_RAM_Create | int numElements | Создаёт структуру типа Arrays1D, содержащую numElements элементов типа Array1D, в ОЗУ |
| | void | Arrays1D_AddArray1D | Arrays1D* arrays1D, int index, Array1D array1D | Размещает Array1D в Arrays1D по указанному индексу index |
|------ | Arrays1D* | Arrays1D_RAM_Create_From_Arrays1D_GPU | Arrays1D* arrays1D_GPU | Создаёт структуру типа Arrays1D в ОЗУ как копию структуры Arrays1D, расположенной в GPU, и возвращает на неё указатель |
| | Arrays1D* | Arrays1D_GPU_Create | int numElements | Создаёт структуру типа Arrays1D, содержащую numElements элементов, в видеопамяти и возвращает на него указатель |
| | Arrays1D* | Arrays1D_GPU_Create_From_Arrays1D_RAM | Arrays1D arrays1D_RAM | Создаёт структуру типа Arrays1D в GPU как копию структуры arrays1D_RAM, расположенной в RAM, и возвращает на неё указатель |
|------ | void | Array1D_RAM_InitByIndexes | Array1D arr | Инициализирует элементы массива data структуры Array1D их индексами |
| \_\_host\_\_ \_\_device\_\_ | void | Arrays1D_Print | Arrays1D arr | Выводит содержимое структуры Arrays1D в консоль |
| | void | Arrays1D_RAM_Destruct | Arrays1D* arrays1D_RAM | Освобождает оперативную память, выделенную под структуру array1D_RAM |
| | void | Arrays1D_GPU_Destruct | Arrays1D* arrays1D_GPU | Освобождает видеопамять, выделенную под структуру array1D_GPU |



## CUDA-ядра 

| Дополнительные атрибуты функции | Тип возвращаемого значения | Функция | Аргументы  | Описание |
| ------------- | ------------- |:-------------:| -----:|---|
| \_\_global\_\_ | void | CudaFloatArray_Print | float* fArray_GPU, int numElements | Выводит в консоль массив fArray_GPU, расположенный в видеопамяти и содержащий numElements элементов типа float |
| \_\_global\_\_ | void | CudaArray1D_GPU_Print | Array1D* array1D_GPU | Выводит в консоль структуру Array1D, расположенную в GPU.  |
| \_\_global\_\_ | void | CudaArray1D_GPU_Array_Print | Array1D* array1D_GPU_Array, int numElements | Выводит в консоль массив структур Array1D, расположенный в GPU.  |
| \_\_global\_\_ | void | CudaArrays1D_GPU_Print | Arrays1D* arrays1D_GPU | Выводит в консоль структуру Arrays1D, расположенную в GPU.  |
| \_\_global\_\_ | void | CudaArray1D_AddNumber | Array1D* arr, float number | Прибавляет число number к каждому элементу массива arr->data, рсположенному в GPU.  |