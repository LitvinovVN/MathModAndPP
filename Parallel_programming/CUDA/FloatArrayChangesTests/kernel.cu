// Задача 01. Вывести с помощью GPU текстовую строку "Hello World from GPU!"
// Запуск:
// nvcc kernel.cu -o app
// ./app

#include <iostream>
#include <time.h>

__host__ __device__
void printArray(float* arr_GPU, int numElements)
{
    for(int i = 0; i < numElements; i++)
    {
        printf("%g ", arr_GPU[i]);
    }
    printf("\n"); 
}

__host__ __device__
void testCalculation(float* arr_GPU, int numElements)
{
    for(int i = 0; i < numElements; i++)
    {
        arr_GPU[i] += 10;
    }  
}


__global__ void cuda_print_array(float* arr_GPU, int numElements){
    printf("Hello World from GPU!\n");

    printArray(arr_GPU, numElements);
    printf("\n");

    testCalculation(arr_GPU, numElements);
    printArray(arr_GPU, numElements);
}

__global__ void cuda_calculation(float* arr_GPU_01, float* arr_GPU_02, int numElements){
    //printf("---cuda_calculation---\n");
    testCalculation(arr_GPU_01, numElements);
    //testCalculation(arr_GPU_02, numElements);
}

int main() {
    int numElements = 1000000;
    size_t dataSize = numElements * sizeof(float);
    float* arr_RAM = (float*)malloc(dataSize);
    for(int i = 0; i < numElements; i++)
    {
        arr_RAM[i] = i;
    }
    
    /////// CPU ///////
    float* arr_RAM_res = (float*)malloc(dataSize);
    for(int i = 0; i < numElements; i++)
    {
        arr_RAM_res[i] = arr_RAM[i];
    }

    clock_t t;
    t = clock();
    testCalculation(arr_RAM_res, numElements);
    t = clock() - t;
    double time_taken = ((double)t)*1000/CLOCKS_PER_SEC; // in milliseconds
 
    printf("CPU testCalculation() took %f milliseconds to execute \n", time_taken);
    ///////////////////


    float* arr_GPU_01;
    cudaMalloc((void**)&arr_GPU_01, dataSize);
    cudaMemcpy(arr_GPU_01, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_02;
    cudaMalloc((void**)&arr_GPU_02, dataSize);
    cudaMemcpy(arr_GPU_02, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    // Prepare
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start record
    cudaEventRecord(start, 0);
    // Do something on GPU
    cuda_calculation<<<1,1>>>(arr_GPU_01, arr_GPU_02, numElements);    
    // Stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
    // Clean up:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\nelapsedTime GPU = %g", elapsedTime);


    //////////// Сравнение результатов CPU и GPU ///////////
    float* arr_GPU_res1 = (float*)malloc(dataSize);
    cudaMemcpy(arr_GPU_res1, arr_GPU_01, dataSize, cudaMemcpyDeviceToHost);
    float* arr_GPU_res2 = (float*)malloc(dataSize);
    cudaMemcpy(arr_GPU_res2, arr_GPU_02, dataSize, cudaMemcpyDeviceToHost);

    for(int i = 0; i < numElements; i++)
    {
        if(arr_GPU_res1[i] - arr_RAM_res[i] > 0.000001)
            printf("ERROR! i=%d %g %g\n", i, arr_GPU_res1[i], arr_RAM_res[i]);
    }

    ////////////////////////////////////////////////////////


    return 0;
}