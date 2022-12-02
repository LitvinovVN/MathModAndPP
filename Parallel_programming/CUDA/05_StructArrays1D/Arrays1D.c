///////////////////////////////////////////////////////
// Создаём структуру "Одномерный массив элементов типа Array1D"
struct arrays_t {
    int numElements;
    Array1D* data;
};

//Определяем новый тип
typedef struct arrays_t Arrays1D;

//////////////////////////////////////////////////

Arrays1D Arrays1D_Create(int numElements)
{
    Arrays1D arr = {numElements};
    arr.data = (Array1D*) malloc(numElements * sizeof(Array1D));

    int i = 0;

    while(i < arr.numElements)
    {
        arr.data[i] = Array1D_RAM_Create(0);
        i++;
    }

    return arr;
}

// Размещает Array1D в Arrays1D по указанному индексу index
void Arrays1D_AddArray1D(Arrays1D* arrays1D, int index, Array1D array1D)
{    
    arrays1D->data[index] = array1D;
}

// Выводит содержимое структуры Arrays1D arr в консоль
__host__ __device__
void Arrays1D_Print(Arrays1D arr)
{
    printf("Arrays1D:\n");
    printf("\tnumElements = %d\n", arr.numElements);
    printf("\tdata:\n");

    int i = 0;

    while(i < arr.numElements)
    {
        printf("\tIndex: %d\n", i);
        Array1D_Print(&arr.data[i]);
        i++;
    }
    printf("\n");
}

// Выводит структуру в консоль Array1D, расположенную в GPU
__global__ void CudaArrays1D_GPU_Print(Arrays1D* arrays1D_GPU)
{
    printf("CudaArrays1D_GPU_Print:\n");
    Arrays1D_Print(*arrays1D_GPU);    
}

// Создаёт структуру типа Arrays1D в памяти GPU и возвращает на неё указатель
Arrays1D* Arrays1D_GPU_Create(int numElements)
{
    printf("Arrays1D_GPU_Create started\n");
    // 1. Выделяем память в GPU под структуру Arrays1D
    Arrays1D* arrays1D_GPU;    
    cudaMalloc((void**)&arrays1D_GPU, sizeof(Arrays1D));
    
    // 2. Копируем numElements в поле arrays1D_GPU->numElements структуры Arrays1D, рассположенной в GPU
    cudaMemcpy(&(arrays1D_GPU->numElements), &numElements, sizeof(arrays1D_GPU->numElements), cudaMemcpyHostToDevice);
    
    // 3. Выделяем память в GPU для хранения массива, в котором количество элементов = numElements
    //float* data_GPU = FloatArray_GPU_Create(numElements);
    
    // 4. Создаем массив data_RAM в ОЗУ и инициализируем элементы массива их индексами
    //float* data_RAM = FloatArray_RAM_Create(numElements);
    //FloatArray_InitByIndexes(data_RAM, numElements);       

    // 5. Копируем массив data_RAM из GPU в массив data_GPU в ОЗУ
    //cudaMemcpy(data_GPU, data_RAM, numElements * sizeof(*(array1D_GPU->data)), cudaMemcpyHostToDevice);
    
    // 6. Копируем указатель на массив data_GPU в поле data массива array1D_GPU, находящегося на GPU
    //cudaMemcpy(&(array1D_GPU->data), &data_GPU, sizeof(array1D_GPU->data), cudaMemcpyHostToDevice);

    printf("Array1D_GPU_Create ended\n");
    return arrays1D_GPU;
}

/* Создаёт структуру типа Arrays1D в GPU как копию структуры Arrays1D,
 расположенной в RAM, и возвращает на неё указатель */
 Arrays1D* Arrays1D_GPU_Create_From_Arrays1D_RAM(Arrays1D arrays1D_RAM)
 {    
    Arrays1D* arrays1D_GPU = Arrays1D_GPU_Create(arrays1D_RAM.numElements);
 
    Arrays1D* arrays1D_DTO = (Arrays1D*)malloc(sizeof(Arrays1D));
    cudaMemcpy(arrays1D_DTO, arrays1D_GPU, sizeof(Arrays1D), cudaMemcpyDeviceToHost);
    
    cudaMemcpy(arrays1D_DTO->data, arrays1D_RAM.data, arrays1D_DTO->numElements * sizeof(float), cudaMemcpyHostToDevice);
    
    free(arrays1D_DTO);
    
    return arrays1D_GPU;
 }

//////////////////////////////////////////////////