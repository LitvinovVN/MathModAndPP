// Создаёт массив float в динамической памяти и возвращает на него указатель
float* FloatArray_RAM_Create(int numElements)
{
    float* arr = (float*)malloc(numElements * sizeof(float));
    return arr;
}

//  Создаёт массив float в GPU и возвращает на него указатель
float* FloatArray_GPU_Create(int numElements)
{
    float* arrayGPU;
    cudaMalloc((void**)&arrayGPU, numElements * sizeof(float));
    return arrayGPU;
}

// Инициализирует элементы массива их индексами
__host__ __device__
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

void FloatArray_CopyFromRAMtoGPU(float* fArray_RAM, float* fArray_GPU, int numElements)
{
    cudaMemcpy(fArray_GPU, fArray_RAM, numElements * sizeof(float), cudaMemcpyHostToDevice);
}

void FloatArray_CopyFromGPUtoRAM(float* fArray_GPU, float* fArray_RAM, int numElements)
{
    cudaMemcpy(fArray_RAM, fArray_GPU, numElements * sizeof(float), cudaMemcpyDeviceToHost);
}
