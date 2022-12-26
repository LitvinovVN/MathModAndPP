#include "stddef.h"

enum dataType { C0, C2, C4, C6, R };
typedef enum dataType DataType;
///////////////////////////////////////////////////////
// Создаем структуру "Трёхмерный массив"
struct array3d_t
{
    int size_x;
    int size_y;
    int size_z;   

    float* c0;
    float* c2;
    float* c4;
    float* c6;

    float* r;
};

// Определяем новый тип
typedef struct array3d_t Array3D;
////////////////////////////////////////////////// 

// Создаёт структуру типа Array3D в ОЗУ
Array3D Array3D_RAM_Create(int size_x, int size_y, int size_z)
{
    Array3D arr = {size_x, size_y, size_z};
    int numElements = size_x * size_y * size_z;

    arr.c0 = (float*)malloc(numElements * sizeof(float));
    arr.c2 = (float*)malloc(numElements * sizeof(float));
    arr.c4 = (float*)malloc(numElements * sizeof(float));
    arr.c6 = (float*)malloc(numElements * sizeof(float));
    arr.r = (float*)malloc(numElements * sizeof(float));

    int i = 0;

    while(i < numElements)
    {
        arr.c0[i] = 0;
        arr.c2[i] = 0;
        arr.c4[i] = 0;
        arr.c6[i] = 0;
        arr.r[i] = 0;
        i++;
    }


    return arr;
}

// Инициализирует элементы массива структуры Array1D их индексами
void Array3D_RAM_InitByIndexes(Array3D* array3D, DataType dt)
{
    size_t k = 0;
    while(k < array3D->size_z)
    {
        size_t j = 0;
        while(j < array3D->size_y)
        {
            size_t i = 0;
            while(i < array3D->size_x)
            {
                int ind = k * array3D->size_x * array3D->size_y + j * array3D->size_x + i;
                if(dt == C0)                
                    array3D->c0[ind] = ind;
                if(dt == C2)                
                    array3D->c2[ind] = ind;
                if(dt == C4)                
                    array3D->c4[ind] = ind;
                if(dt == C6)                
                    array3D->c6[ind] = ind;
                if(dt == R)                
                    array3D->r[ind] = ind;

                i++;
            }
            j++;
        }        
        k++;
    }    
}


// Выводит элементы массива в консоль
__host__ __device__
void Array3D_Print(Array3D* array3D, DataType dt)
{
    printf("-----Array3D_Print-----\n");
    printf("DataType=");
    if(dt == C0)                
        printf("C0");
    if(dt == C2)                
        printf("C2");
    if(dt == C4)                
        printf("C4");
    if(dt == C6)                
        printf("C6");
    if(dt == R)                
        printf("R");
    printf("\n");

    printf("size_x=%d\n", array3D->size_x);
    printf("size_y=%d\n", array3D->size_y);
    printf("size_z=%d\n", array3D->size_z);
    size_t k = 0;
    while(k < array3D->size_z)
    {
        printf("\nz=%zu: \n", k);

        size_t j = 0;
        while(j < array3D->size_y)
        {
            printf("y=%zu: | ", j);

            size_t i = 0;
            while(i < array3D->size_x)
            {
                int ind = k * array3D->size_x * array3D->size_y + j * array3D->size_x + i;
                if(dt == C0)                
                    printf("%g ", array3D->c0[ind]);
                if(dt == C2)                
                    printf("%g ", array3D->c2[ind]);
                if(dt == C4)                
                    printf("%g ", array3D->c4[ind]);
                if(dt == C6)                
                    printf("%g ", array3D->c6[ind]);
                if(dt == R)                
                    printf("%g ", array3D->r[ind]);
                
                printf(" | ");

                i++;
            }
            printf("\ty=%zu\n", j);
            j++;
        }       
        
        k++;
    }
    printf("\n");      
}


__host__ __device__
void Array3D_Array_Print(Array3D* array3D_arr, int numElements, DataType dt)
{
    printf("----------Array3D_Array_Print----------\n");    
    for(int i = 0; i < numElements; i++)
    {
        printf("\t[i]=%d\n", i);
        Array3D_Print(&array3D_arr[i], dt);
    }
}



// Создаёт структуру типа Array3D в памяти GPU и возвращает на неё указатель
Array3D* Array3D_GPU_Create(int size_x, int size_y, int size_z)
{
    printf("Array3D_GPU_Create started\n");
    // 1. Выделяем память в GPU под структуру Array3D
    Array3D* array3D_GPU;    
    cudaMalloc((void**)&array3D_GPU, sizeof(Array3D));
    
    // 2. Копируем size_x в поле array3D_GPU->size_x структуры Array3D, рассположенной в GPU
    cudaMemcpy(&(array3D_GPU->size_x), &size_x, sizeof(array3D_GPU->size_x), cudaMemcpyHostToDevice);
    cudaMemcpy(&(array3D_GPU->size_y), &size_y, sizeof(array3D_GPU->size_y), cudaMemcpyHostToDevice);
    cudaMemcpy(&(array3D_GPU->size_z), &size_z, sizeof(array3D_GPU->size_z), cudaMemcpyHostToDevice);
    
    // 3. Выделяем память в GPU для хранения массивов, в котором количество элементов = numElements
    int numElements = size_x * size_y * size_z;
    float* data_GPU_c0 = FloatArray_GPU_Create(numElements);
    float* data_GPU_c2 = FloatArray_GPU_Create(numElements);
    float* data_GPU_c4 = FloatArray_GPU_Create(numElements);
    float* data_GPU_c6 = FloatArray_GPU_Create(numElements);
    float* data_GPU_r  = FloatArray_GPU_Create(numElements);
    
    // 4. Создаем массивы в ОЗУ и инициализируем элементы массива их индексами
    float* data_RAM_c0 = FloatArray_RAM_Create(numElements);
    float* data_RAM_c2 = FloatArray_RAM_Create(numElements);
    float* data_RAM_c4 = FloatArray_RAM_Create(numElements);
    float* data_RAM_c6 = FloatArray_RAM_Create(numElements);
    float* data_RAM_r  = FloatArray_RAM_Create(numElements);
    //FloatArray_InitByIndexes(data_RAM, numElements);       

    // 5. Копируем массивы data_RAM_x из GPU в массивы data_GPU_x в ОЗУ
    cudaMemcpy(data_GPU_c0, data_RAM_c0, numElements * sizeof(*(array3D_GPU->c0)), cudaMemcpyHostToDevice);
    cudaMemcpy(data_GPU_c2, data_RAM_c2, numElements * sizeof(*(array3D_GPU->c2)), cudaMemcpyHostToDevice);
    cudaMemcpy(data_GPU_c4, data_RAM_c4, numElements * sizeof(*(array3D_GPU->c4)), cudaMemcpyHostToDevice);
    cudaMemcpy(data_GPU_c6, data_RAM_c6, numElements * sizeof(*(array3D_GPU->c6)), cudaMemcpyHostToDevice);
    cudaMemcpy(data_GPU_r,  data_RAM_r,  numElements * sizeof(*(array3D_GPU->r)),  cudaMemcpyHostToDevice);
    
    // 6. Копируем указатель на массив data_GPU в поле data массива array1D_GPU, находящегося на GPU
    cudaMemcpy(&(array3D_GPU->c0), &data_GPU_c0, sizeof(array3D_GPU->c0), cudaMemcpyHostToDevice);
    cudaMemcpy(&(array3D_GPU->c2), &data_GPU_c2, sizeof(array3D_GPU->c2), cudaMemcpyHostToDevice);
    cudaMemcpy(&(array3D_GPU->c4), &data_GPU_c4, sizeof(array3D_GPU->c4), cudaMemcpyHostToDevice);
    cudaMemcpy(&(array3D_GPU->c6), &data_GPU_c6, sizeof(array3D_GPU->c6), cudaMemcpyHostToDevice);
    cudaMemcpy(&(array3D_GPU->r),  &data_GPU_r,  sizeof(array3D_GPU->r),  cudaMemcpyHostToDevice);

    printf("Array3D_GPU_Create ended\n");
    return array3D_GPU;
}



/* Создаёт структуру типа Array3D в GPU как копию структуры Array3D,
 расположенной в RAM, и возвращает на неё указатель */
 Array3D* Array3D_GPU_Create_From_Array3D_RAM(Array3D array3D_RAM)
 {    
    Array3D* array3D_GPU = Array3D_GPU_Create(array3D_RAM.size_x, array3D_RAM.size_y, array3D_RAM.size_z);
    Array3D* array3D_DTO = (Array3D*)malloc(sizeof(Array3D));
    cudaMemcpy(array3D_DTO, array3D_GPU, sizeof(Array3D), cudaMemcpyDeviceToHost);
    
    int numElements = array3D_DTO->size_x * array3D_DTO->size_y * array3D_DTO->size_z;
    cudaMemcpy(array3D_DTO->c0, array3D_RAM.c0, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(array3D_DTO->c2, array3D_RAM.c2, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(array3D_DTO->c4, array3D_RAM.c4, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(array3D_DTO->c6, array3D_RAM.c6, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(array3D_DTO->r,  array3D_RAM.r,  numElements * sizeof(float), cudaMemcpyHostToDevice);
    
    free(array3D_DTO);
    
    return array3D_GPU;
 }


 //////////////////////////////////////////////////////////////////////////////

// Выводит структуру Array3D, расположенную в GPU, в консоль 
__global__ void CudaArray3D_GPU_Print(Array3D* array3D_GPU, DataType dt)
{
    printf("CudaArray3D_GPU_Print:\n");
    Array3D_Print(array3D_GPU, dt);    
}