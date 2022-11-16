/* Задача 02. Ввести с консоли количество элементов массива.
 Создать массив.
 Инициализировать значения элементов массива индексами.
 Вывести массив в консоль

 Запуск:
 nvcc kernel.cu -o app.exe
 ./app

*/

#include <stdio.h>
#include <stdlib.h> // Для использования malloc

// Получает строку-сообщение message, выводит её в консоль.
// Считывает целое число, введённое пользователем и возвращает его.
int IntNumber_Input(const char message[])
{
    int numElements;
    printf(message);
    scanf("%d", &numElements);

    return numElements;
}

// Создаёт массив float в динамической памяти и возвращает на него указатель
float* FloatArrayRAM_Create(int numElements)
{
    float* arr = (float*)malloc(numElements * sizeof(float));
    return arr;
}


// Инициализирует элементы массива их индексами
void FloatArray_InitByIndexes(float* fArray, int numElements)
{
    size_t i = 0;
    while(i < numElements)
    {
        fArray[i] = i;
        i++;
    }        
}

// Выводит элементы массива в консоль
__host__ __device__
void FloatArray_Print(float* fArray, int numElements)
{
    size_t i = 0;
    while(i < numElements)
    {
        printf("%g ",fArray[i]);
        i++;
    }
    printf("\n");      
}

//////////////////////////////////////////////////////////////////////////////
__global__ void CudaFloatArray_Print(float* fArray_GPU, int numElements)
{
    printf("CudaFloatArray_Print:\n");
    FloatArray_Print(fArray_GPU, numElements);    
}

__global__ void CudaFloatArray_AddNumber(float* fArray_GPU, int numElements, float number)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    fArray_GPU[index] = fArray_GPU[index] + number; 
}



//////////////////////////////////////////////////////////////////////////////

float* FloatArrayGPU_Create(int numElements)
{
    float* arrayGPU;
    cudaMalloc((void**)&arrayGPU, numElements * sizeof(float));
    return arrayGPU;
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