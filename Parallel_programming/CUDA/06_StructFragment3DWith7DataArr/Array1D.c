///////////////////////////////////////////////////////
// Создаем структуру "Одномерный массив"
struct array_t
{
    int size;
    float* data;
};

// Определяем новый тип
typedef struct array_t Array1D;
////////////////////////////////////////////////// 

// Создаёт структуру типа Array1D в ОЗУ
Array1D Array1D_RAM_Create(int numElements)
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
__global__ void CudaFloatArray_Print(float* fArray_GPU, int numElements)
{
    printf("CudaFloatArray_Print:\n");
    FloatArray_Print(fArray_GPU, numElements);    
}

///////////////////////////////////////////////////////

// Создаёт структуру типа Array1D в памяти GPU и возвращает на неё указатель
Array1D* Array1D_GPU_Create(int numElements)
{
    printf("Array1D_GPU_Create started\n");
    // 1. Выделяем память в GPU под структуру Array1D
    Array1D* array1D_GPU;    
    cudaMalloc((void**)&array1D_GPU, sizeof(Array1D));
    
    // 2. Копируем numElements в поле array1D_GPU->size структуры Array1D, рассположенной в GPU
    cudaMemcpy(&(array1D_GPU->size), &numElements, sizeof(array1D_GPU->size), cudaMemcpyHostToDevice);
    
    // 3. Выделяем память в GPU для хранения массива, в котором количество элементов = numElements
    float* data_GPU = FloatArray_GPU_Create(numElements);
    
    // 4. Создаем массив data_RAM в ОЗУ и инициализируем элементы массива их индексами
    float* data_RAM = FloatArray_RAM_Create(numElements);
    FloatArray_InitByIndexes(data_RAM, numElements);       

    // 5. Копируем массив data_RAM из GPU в массив data_GPU в ОЗУ
    cudaMemcpy(data_GPU, data_RAM, numElements * sizeof(*(array1D_GPU->data)), cudaMemcpyHostToDevice);
    
    // 6. Копируем указатель на массив data_GPU в поле data массива array1D_GPU, находящегося на GPU
    cudaMemcpy(&(array1D_GPU->data), &data_GPU, sizeof(array1D_GPU->data), cudaMemcpyHostToDevice);

    printf("Array1D_GPU_Create ended\n");
    return array1D_GPU;
}

// Инициализирует элементы массива структуры Array1D их индексами
void Array1D_RAM_InitByIndexes(Array1D arr)
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
void Array1D_Print(Array1D* array1D)
{
    printf("size=%d\n",array1D->size);
    size_t i = 0;
    while(i < array1D->size)
    {
        //printf("i=%d: ",i);
        printf("%g ",array1D->data[i]);
        i++;
    }
    printf("\n");      
}

__host__ __device__
void Array1D_Array_Print(Array1D* array1D, int numElements)
{    
    for(int i = 0; i < numElements; i++)
    {
        Array1D_Print(&array1D[i]);
    }
}


//////////////////////////////////////////////////////////////////////////////

// Выводит структуру в консоль Array1D, расположенную в GPU
__global__ void CudaArray1D_GPU_Print(Array1D* array1D_GPU)
{
    printf("CudaArray1D_GPU_Print:\n");
    Array1D_Print(array1D_GPU);    
}

__global__ void CudaArray1D_GPU_Array_Print(Array1D* array1D_GPU_Array, int numElements)
{
    printf("CudaArray1D_GPU_Array_Print:\n");
    Array1D_Array_Print(array1D_GPU_Array, numElements);
    /*for(int i = 0; i < numElements; i++)
    {
        Array1D_Print(&array1D_GPU_Array[i]);
    }*/
}

// Прибавляет число number к каждому элементу массива arr.data, рсположенному в GPU
__global__ void CudaArray1D_AddNumber(Array1D* arr, float number)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    arr->data[index] = arr->data[index] + number; 
}

//////////////////////////////////////////////////////////////////////////////
/*Array1D Array1DGPU_Create(int numElements)
{
    Array1D arrGPU = {numElements};
    cudaMalloc((void**)&arrGPU.data, numElements * sizeof(float));
    return arrGPU;
}*/


//////////////////////////////////////////////////////////////////////////////

/* Создаёт структуру типа Array1D в ОЗУ как копию структуры Array1D,
 расположенной в GPU, и возвращает на неё указатель */
 Array1D* Array1D_RAM_Create_From_Array1D_GPU(Array1D* array1D_GPU)
 {
    Array1D* array1D_RAM = (Array1D*)malloc(sizeof(Array1D));
 
    Array1D* array1D_DTO = (Array1D*)malloc(sizeof(Array1D));
    // Копируем структуру из GPU в ОЗУ 
    cudaMemcpy(array1D_DTO, array1D_GPU, sizeof(Array1D), cudaMemcpyDeviceToHost);
    
    array1D_RAM->size = array1D_DTO->size;    
    array1D_RAM->data = (float*)malloc(array1D_DTO->size * sizeof(float));
    cudaMemcpy(array1D_RAM->data, array1D_DTO->data, array1D_DTO->size * sizeof(float), cudaMemcpyDeviceToHost);
    return array1D_RAM;
 }

 /* Создаёт структуру типа Array1D в GPU как копию структуры Array1D,
 расположенной в RAM, и возвращает на неё указатель */
 Array1D* Array1D_GPU_Create_From_Array1D_RAM(Array1D array1D_RAM)
 {    
    Array1D* array1D_GPU = Array1D_GPU_Create(array1D_RAM.size);
 
    Array1D* array1D_DTO = (Array1D*)malloc(sizeof(Array1D));
    cudaMemcpy(array1D_DTO, array1D_GPU, sizeof(Array1D), cudaMemcpyDeviceToHost);
    
    cudaMemcpy(array1D_DTO->data, array1D_RAM.data, array1D_DTO->size * sizeof(float), cudaMemcpyHostToDevice);
    
    free(array1D_DTO);
    
    return array1D_GPU;
 }

//  Создаёт массив структур Array1D в GPU и возвращает на него указатель
Array1D* Array1DArray_GPU_Create(int numElements)
{
    Array1D* array1D_GPU;
    cudaMalloc((void**)&array1D_GPU, numElements * sizeof(Array1D));
    return array1D_GPU;
}

// Создаёт массив структур Array1D в GPU, содержащий numElements элементов,
// из массива структур Array1D ОЗУ 
Array1D* Array1D_GPU_Array_Create_From_Array1D_RAM_Array(Array1D* array1D_RAM_Array, int numElements)
{
    Array1D* array1D_GPU_Array;
    cudaMalloc((void**)&array1D_GPU_Array, numElements * sizeof(Array1D));

    // 0. Создаём объект DTO (массив Array1D) для копирования в GPU
    Array1D* array1D_RAM_Array_DTO = (Array1D*)malloc(numElements * sizeof(Array1D));
    //Array1D* arrays1D_DTO_data = (Array1D*)malloc(arrays1D_RAM.numElements * sizeof(Array1D));
    //cudaMemcpy(arrays1D_DTO_data, arrays1D_DTO->data, arrays1D_RAM.numElements * sizeof(Array1D), cudaMemcpyDeviceToHost);
        
    for(int i = 0; i < numElements; i++)
    {
        // 1. Создаём массив float* в GPU и сохраняем указатель на него
        float* data_dev = FloatArray_GPU_Create(array1D_RAM_Array[i].size);
        // 2. Копируем содержимое массива float* из ОЗУ в GPU
        FloatArray_CopyFromRAMtoGPU(array1D_RAM_Array[i].data, data_dev, array1D_RAM_Array[i].size);
        // 3. Настраиваем i-й элемент объекта DTO (numElements и указатель на массив float* в GPU)
        array1D_RAM_Array_DTO[i].size = array1D_RAM_Array[i].size;
        array1D_RAM_Array_DTO[i].data = data_dev;
        //CudaFloatArray_Print<<<1,1>>>(array1D_RAM_Array_DTO[i].data, array1D_RAM_Array_DTO[i].size);
    }

    // 4. Копируем объект DTO в GPU
    cudaMemcpy(array1D_GPU_Array, array1D_RAM_Array_DTO, numElements * sizeof(Array1D), cudaMemcpyHostToDevice);
    //cudaDeviceSynchronize();
    //CudaArray1D_GPU_Array_Print<<<1,1>>>(array1D_GPU_Array, numElements);
    //cudaDeviceSynchronize();

    return array1D_GPU_Array;
}

//Создаёт массив структур Array1D в RAM, содержащий numElements элементов, из массива структур Array1D GPU
Array1D* Array1D_RAM_Array_Create_From_Array1D_GPU_Array (Array1D* array1D_GPU_Array, int numElements)
{
    // Создаём массив структур Array1D в RAM
    Array1D* array1D_RAM_Array = (Array1D*)malloc(numElements * sizeof(Array1D));
    
    // Создаём DTO
    Array1D* array1D_RAM_Array_DTO = (Array1D*)malloc(numElements * sizeof(Array1D));
    // Копируем в DTO данные из GPU
    cudaMemcpy(array1D_RAM_Array_DTO, array1D_GPU_Array, numElements * sizeof(Array1D), cudaMemcpyDeviceToHost);
    

    for(int i = 0; i < numElements; i++)
    {
        // 1. Создаём массив float* в RAM и сохраняем указатель на него
        float* data = FloatArray_RAM_Create(array1D_RAM_Array_DTO[i].size);        
        // 2. Копируем содержимое массива float* из GPU в ОЗУ
        FloatArray_CopyFromGPUtoRAM(array1D_RAM_Array_DTO[i].data, data, array1D_RAM_Array_DTO[i].size);
        // 3. Настраиваем i-й элемент объекта в RAM (numElements и указатель на массив float* в RAM)
        array1D_RAM_Array[i].size = array1D_RAM_Array_DTO[i].size;
        array1D_RAM_Array[i].data = data;
        //CudaFloatArray_Print<<<1,1>>>(array1D_RAM_Array_DTO[i].data, array1D_RAM_Array_DTO[i].size);
    }

    return array1D_RAM_Array;
}

//////////////////////////////////////////////////////////////////////////////

void Array1D_RAM_Destruct(Array1D* array1D_RAM)
{
    free(array1D_RAM->data);
    free(array1D_RAM);
}

void Array1D_GPU_Destruct(Array1D* array1D_GPU)
{
    Array1D* array1D_DTO = (Array1D*)malloc(sizeof(Array1D));
    cudaMemcpy(array1D_DTO, array1D_GPU, sizeof(Array1D), cudaMemcpyDeviceToHost);
    cudaFree(array1D_DTO->data);
    cudaFree(array1D_DTO);
    free(array1D_DTO);
}

/////////////////////////////////////////////////////////////////////////////