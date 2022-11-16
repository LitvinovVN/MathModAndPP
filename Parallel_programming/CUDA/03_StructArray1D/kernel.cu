/* Задача 03. Создать структуру "Динамический одномерный массив", содержащий элементы типа float.
 Инициализировать поля структуры.
 Вывести структуру в консоль

 Запуск:
 nvcc kernel.cu -o app.exe
 ./app

*/

#include <stdio.h>
#include <stdlib.h> // Для использования malloc

///////////////////////////////////////////////////////
// Создаем структуру "Одномерный массив"
struct array_t
{
    int size;
    float* data;
};

// Определяем новый тип
typedef struct array_t Array1D;
/////////////////////////////////////////////////////// 

// Создаёт структуру типа Array1D в ОЗУ
Array1D Array1DRAM_Create(int numElements)
{
    Array1D arr = {numElements};
    arr.data = (float*)malloc(numElements * sizeof(float));

    int i = 0;

    while(i < arr.size)
    {
        arr.data[i] = 0;
        i++;
    }
    return arr;
}

///////////////////////////////////////////////////////

// Получает строку-сообщение message, выводит её в консоль.
// Считывает целое число, введённое пользователем и возвращает его.
int IntNumber_Input(const char message[])
{
    int numElements;
    printf(message);
    scanf("%d", &numElements);

    return numElements;
}

// Инициализирует элементы массива структуры Array1D их индексами
void Array1DRAM_InitByIndexes(Array1D arr)
{
    size_t i = 0;
    while(i < arr.size)
    {
        arr.data[i] = i;
        i++;
    }        
}

// Выводит элементы массива в консоль
__host__ __device__
void array1D_Print(Array1D arr)
{
    size_t i = 0;
    while(i < arr.size)
    {
        printf("%g ",arr.data[i]);
        i++;
    }
    printf("\n");      
}

//////////////////////////////////////////////////////////////////////////////
__global__ void CudaArray1D_Print(Array1D arr)
{
    printf("CudaArray1D_Print:\n");
    array1D_Print(arr);    
}

__global__ void CudaArray1D_AddNumber(Array1D arr, float number)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    arr.data[index] = arr.data[index] + number; 
}



//////////////////////////////////////////////////////////////////////////////
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Array1D Array1DGPU_Create(int numElements)
{
    Array1D arrGPU = {numElements};
    cudaMalloc((void**)&arrGPU.data, numElements * sizeof(float));
    return arrGPU;
}

void FloatArray_CopyFromRAMtoGPU(float* fArray_RAM, float* fArray_GPU, int numElements)
{
    cudaMemcpy(fArray_GPU, fArray_RAM, numElements * sizeof(float), cudaMemcpyHostToDevice);
}

void FloatArray_CopyFromGPUtoRAM(float* fArray_GPU, float* fArray_RAM, int numElements)
{
    cudaMemcpy(fArray_RAM, fArray_GPU, numElements * sizeof(float), cudaMemcpyDeviceToHost);
}

//////////////////////////////////////////////////////////////////////////////


int main()
{    
    int numElements = IntNumber_Input("Input number of array elements: ");
    printf("numElements = %d\n", numElements);

    float* fArray_RAM = FloatArrayRAM_Create(numElements);
    FloatArray_InitByIndexes(fArray_RAM, numElements);
    FloatArray_Print(fArray_RAM, numElements);

    float* fArray_GPU = FloatArrayGPU_Create(numElements);
    FloatArray_CopyFromRAMtoGPU(fArray_RAM, fArray_GPU, numElements);

    CudaFloatArray_Print<<<1,1>>>(fArray_GPU, numElements);
    cudaDeviceSynchronize();
    CudaFloatArray_AddNumber<<<1,numElements>>>(fArray_GPU, numElements, 5);
    cudaDeviceSynchronize();
    CudaFloatArray_Print<<<1,1>>>(fArray_GPU, numElements);

    FloatArray_CopyFromGPUtoRAM(fArray_GPU, fArray_RAM, numElements);
    FloatArray_Print(fArray_RAM, numElements);

    return 0;
}