// Задача 01. Вывести с помощью GPU текстовую строку "Hello World from GPU!"
// Запуск:
// nvcc kernel2.cu -o app --run
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
void testCalculation(float* arr, int numElements)
{
    for(int i = 1; i < numElements-1; i++)
    {
        arr[i] = - 1.5f * arr[i-1] + 2.0f * arr[i] - arr[i+1] + 10;
    }  
}


__global__ void cuda_print_array(float* arr_GPU, int numElements){
    printf("Hello World from GPU!\n");

    printArray(arr_GPU, numElements);
    printf("\n");

    testCalculation(arr_GPU, numElements);
    printArray(arr_GPU, numElements);
}

__global__ void cuda_calculation(float* arr_GPU_01, float* arr_GPU_02, float* arr_GPU_03, float* arr_GPU_04,
    float* arr_GPU_05, float* arr_GPU_06, float* arr_GPU_07, float* arr_GPU_08,
    float* arr_GPU_09, float* arr_GPU_10, float* arr_GPU_11, float* arr_GPU_12,
    float* arr_GPU_13, float* arr_GPU_14, float* arr_GPU_15, float* arr_GPU_16,
    float* arr_GPU_17, float* arr_GPU_18, float* arr_GPU_19, float* arr_GPU_20,
    float* arr_GPU_21, float* arr_GPU_22, float* arr_GPU_23, float* arr_GPU_24,
    float* arr_GPU_25, float* arr_GPU_26, float* arr_GPU_27, float* arr_GPU_28,
    int numElements){
    //printf("---cuda_calculation---\n");
    float* calculatingArr = NULL;
    int blockIndex = blockIdx.x;
    if(blockIndex == 0) calculatingArr = arr_GPU_01;
    if(blockIndex == 1) calculatingArr = arr_GPU_02;
    if(blockIndex == 2) calculatingArr = arr_GPU_03;
    if(blockIndex == 3) calculatingArr = arr_GPU_04;
    if(blockIndex == 4) calculatingArr = arr_GPU_05;
    if(blockIndex == 5) calculatingArr = arr_GPU_06;
    if(blockIndex == 6) calculatingArr = arr_GPU_07;
    if(blockIndex == 7) calculatingArr = arr_GPU_08;
    if(blockIndex == 8) calculatingArr = arr_GPU_09;
    if(blockIndex == 9) calculatingArr = arr_GPU_10;
    if(blockIndex == 10) calculatingArr = arr_GPU_11;
    if(blockIndex == 11) calculatingArr = arr_GPU_12;
    if(blockIndex == 12) calculatingArr = arr_GPU_13;
    if(blockIndex == 13) calculatingArr = arr_GPU_14;
    if(blockIndex == 14) calculatingArr = arr_GPU_15;
    if(blockIndex == 15) calculatingArr = arr_GPU_16;
    if(blockIndex == 16) calculatingArr = arr_GPU_17;
    if(blockIndex == 17) calculatingArr = arr_GPU_18;
    if(blockIndex == 18) calculatingArr = arr_GPU_19;
    if(blockIndex == 19) calculatingArr = arr_GPU_20;
    if(blockIndex == 20) calculatingArr = arr_GPU_21;
    if(blockIndex == 21) calculatingArr = arr_GPU_22;
    if(blockIndex == 22) calculatingArr = arr_GPU_23;
    if(blockIndex == 23) calculatingArr = arr_GPU_24;
    if(blockIndex == 24) calculatingArr = arr_GPU_25;
    if(blockIndex == 25) calculatingArr = arr_GPU_26;
    if(blockIndex == 26) calculatingArr = arr_GPU_27;
    if(blockIndex == 27) calculatingArr = arr_GPU_28;
    
    if (calculatingArr != NULL) testCalculation(calculatingArr, numElements);
    //testCalculation(arr_GPU_01, numElements);
    //testCalculation(arr_GPU_02, numElements);
}

int main() {
    int numElements = 10000000;
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
    //testCalculation(arr_RAM_res, numElements);
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

    float* arr_GPU_03;
    cudaMalloc((void**)&arr_GPU_03, dataSize);
    cudaMemcpy(arr_GPU_03, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_04;
    cudaMalloc((void**)&arr_GPU_04, dataSize);
    cudaMemcpy(arr_GPU_04, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_05;
    cudaMalloc((void**)&arr_GPU_05, dataSize);
    cudaMemcpy(arr_GPU_05, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_06;
    cudaMalloc((void**)&arr_GPU_06, dataSize);
    cudaMemcpy(arr_GPU_06, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_07;
    cudaMalloc((void**)&arr_GPU_07, dataSize);
    cudaMemcpy(arr_GPU_07, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_08;
    cudaMalloc((void**)&arr_GPU_08, dataSize);
    cudaMemcpy(arr_GPU_08, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_09;
    cudaMalloc((void**)&arr_GPU_09, dataSize);
    cudaMemcpy(arr_GPU_09, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_10;
    cudaMalloc((void**)&arr_GPU_10, dataSize);
    cudaMemcpy(arr_GPU_10, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_11;
    cudaMalloc((void**)&arr_GPU_11, dataSize);
    cudaMemcpy(arr_GPU_11, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_12;
    cudaMalloc((void**)&arr_GPU_12, dataSize);
    cudaMemcpy(arr_GPU_12, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_13;
    cudaMalloc((void**)&arr_GPU_13, dataSize);
    cudaMemcpy(arr_GPU_13, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_14;
    cudaMalloc((void**)&arr_GPU_14, dataSize);
    cudaMemcpy(arr_GPU_14, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_15;
    cudaMalloc((void**)&arr_GPU_15, dataSize);
    cudaMemcpy(arr_GPU_15, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_16;
    cudaMalloc((void**)&arr_GPU_16, dataSize);
    cudaMemcpy(arr_GPU_16, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_17;
    cudaMalloc((void**)&arr_GPU_17, dataSize);
    cudaMemcpy(arr_GPU_17, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_18;
    cudaMalloc((void**)&arr_GPU_18, dataSize);
    cudaMemcpy(arr_GPU_18, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_19;
    cudaMalloc((void**)&arr_GPU_19, dataSize);
    cudaMemcpy(arr_GPU_19, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_20;
    cudaMalloc((void**)&arr_GPU_20, dataSize);
    cudaMemcpy(arr_GPU_20, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_21;
    cudaMalloc((void**)&arr_GPU_21, dataSize);
    cudaMemcpy(arr_GPU_21, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_22;
    cudaMalloc((void**)&arr_GPU_22, dataSize);
    cudaMemcpy(arr_GPU_22, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_23;
    cudaMalloc((void**)&arr_GPU_23, dataSize);
    cudaMemcpy(arr_GPU_23, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_24;
    cudaMalloc((void**)&arr_GPU_24, dataSize);
    cudaMemcpy(arr_GPU_24, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_25;
    cudaMalloc((void**)&arr_GPU_25, dataSize);
    cudaMemcpy(arr_GPU_25, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_26;
    cudaMalloc((void**)&arr_GPU_26, dataSize);
    cudaMemcpy(arr_GPU_26, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_27;
    cudaMalloc((void**)&arr_GPU_27, dataSize);
    cudaMemcpy(arr_GPU_27, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    float* arr_GPU_28;
    cudaMalloc((void**)&arr_GPU_28, dataSize);
    cudaMemcpy(arr_GPU_28, arr_RAM, dataSize, cudaMemcpyHostToDevice);

    // Prepare
    cudaStream_t stream1, stream2, stream3, stream4;
    cudaStream_t stream5, stream6, stream7, stream8;
    cudaStream_t stream9, stream10, stream11, stream12;
    cudaStream_t stream13, stream14, stream15, stream16;
    cudaStreamCreate (&stream1);
    cudaStreamCreate (&stream2);
    cudaStreamCreate (&stream3);
    cudaStreamCreate (&stream4);
    cudaStreamCreate (&stream5);
    cudaStreamCreate (&stream6);
    cudaStreamCreate (&stream7);
    cudaStreamCreate (&stream8);
    cudaStreamCreate (&stream9);
    cudaStreamCreate (&stream10);
    cudaStreamCreate (&stream11);
    cudaStreamCreate (&stream12);
    cudaStreamCreate (&stream13);
    cudaStreamCreate (&stream14);
    cudaStreamCreate (&stream15);
    cudaStreamCreate (&stream16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start record
    cudaEventRecord(start, 0);
    // Do something on GPU
    cuda_calculation<<<28,1,0,stream1>>>(arr_GPU_01, arr_GPU_02, arr_GPU_03, arr_GPU_04,
        arr_GPU_05, arr_GPU_06, arr_GPU_07, arr_GPU_08,
        arr_GPU_09, arr_GPU_10, arr_GPU_11, arr_GPU_12,
        arr_GPU_13, arr_GPU_14, arr_GPU_15, arr_GPU_16,
        arr_GPU_17, arr_GPU_18, arr_GPU_19, arr_GPU_20,
        arr_GPU_21, arr_GPU_22, arr_GPU_23, arr_GPU_24,
        arr_GPU_25, arr_GPU_26, arr_GPU_27, arr_GPU_28,
        numElements);
    //cuda_calculation<<<1,1,0,stream2>>>(arr_GPU_02, arr_GPU_02, numElements);
    //cuda_calculation<<<1,1,0,stream3>>>(arr_GPU_03, arr_GPU_03, numElements);
    //cuda_calculation<<<1,1,0,stream4>>>(arr_GPU_04, arr_GPU_04, numElements);
    //cuda_calculation<<<1,1,0,stream5>>>(arr_GPU_05, arr_GPU_05, numElements);
    //cuda_calculation<<<1,1,0,stream6>>>(arr_GPU_06, arr_GPU_06, numElements);
    //cuda_calculation<<<1,1,0,stream7>>>(arr_GPU_07, arr_GPU_07, numElements);
    //cuda_calculation<<<1,1,0,stream8>>>(arr_GPU_08, arr_GPU_08, numElements);
    //cuda_calculation<<<1,1,0,stream9>>>(arr_GPU_09, arr_GPU_09, numElements);
    //cuda_calculation<<<1,1,0,stream10>>>(arr_GPU_10, arr_GPU_10, numElements);
    //cuda_calculation<<<1,1,0,stream11>>>(arr_GPU_11, arr_GPU_11, numElements);
    //cuda_calculation<<<1,1,0,stream12>>>(arr_GPU_12, arr_GPU_12, numElements);
    //cuda_calculation<<<1,1,0,stream13>>>(arr_GPU_13, arr_GPU_13, numElements);
    //cuda_calculation<<<1,1,0,stream14>>>(arr_GPU_14, arr_GPU_14, numElements);
    //cuda_calculation<<<1,1,0,stream15>>>(arr_GPU_15, arr_GPU_15, numElements);
    //cuda_calculation<<<1,1,0,stream16>>>(arr_GPU_16, arr_GPU_16, numElements);    
    cudaDeviceSynchronize ();    
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
    float* arr_GPU_res3 = (float*)malloc(dataSize);
    cudaMemcpy(arr_GPU_res3, arr_GPU_03, dataSize, cudaMemcpyDeviceToHost);
    float* arr_GPU_res4 = (float*)malloc(dataSize);
    cudaMemcpy(arr_GPU_res4, arr_GPU_04, dataSize, cudaMemcpyDeviceToHost);
    float* arr_GPU_res5 = (float*)malloc(dataSize);
    cudaMemcpy(arr_GPU_res5, arr_GPU_05, dataSize, cudaMemcpyDeviceToHost);
    float* arr_GPU_res6 = (float*)malloc(dataSize);
    cudaMemcpy(arr_GPU_res6, arr_GPU_06, dataSize, cudaMemcpyDeviceToHost);
    float* arr_GPU_res7 = (float*)malloc(dataSize);
    cudaMemcpy(arr_GPU_res7, arr_GPU_07, dataSize, cudaMemcpyDeviceToHost);
    float* arr_GPU_res8 = (float*)malloc(dataSize);
    cudaMemcpy(arr_GPU_res8, arr_GPU_08, dataSize, cudaMemcpyDeviceToHost);
    float* arr_GPU_res9 = (float*)malloc(dataSize);
    cudaMemcpy(arr_GPU_res9, arr_GPU_09, dataSize, cudaMemcpyDeviceToHost);
    float* arr_GPU_res10 = (float*)malloc(dataSize);
    cudaMemcpy(arr_GPU_res10, arr_GPU_10, dataSize, cudaMemcpyDeviceToHost);
    float* arr_GPU_res11 = (float*)malloc(dataSize);
    cudaMemcpy(arr_GPU_res11, arr_GPU_11, dataSize, cudaMemcpyDeviceToHost);
    float* arr_GPU_res12 = (float*)malloc(dataSize);
    cudaMemcpy(arr_GPU_res12, arr_GPU_12, dataSize, cudaMemcpyDeviceToHost);
    float* arr_GPU_res13 = (float*)malloc(dataSize);
    cudaMemcpy(arr_GPU_res13, arr_GPU_13, dataSize, cudaMemcpyDeviceToHost);
    float* arr_GPU_res14 = (float*)malloc(dataSize);
    cudaMemcpy(arr_GPU_res14, arr_GPU_14, dataSize, cudaMemcpyDeviceToHost);
    float* arr_GPU_res15 = (float*)malloc(dataSize);
    cudaMemcpy(arr_GPU_res15, arr_GPU_15, dataSize, cudaMemcpyDeviceToHost);
    float* arr_GPU_res16 = (float*)malloc(dataSize);
    cudaMemcpy(arr_GPU_res16, arr_GPU_16, dataSize, cudaMemcpyDeviceToHost);
/*
    for(int i = 0; i < numElements; i++)
    {
        if(abs(arr_GPU_res1[i] - arr_RAM_res[i]) > 0.000001)
            printf("ERROR! i=%d %g %g\n", i, arr_GPU_res1[i], arr_RAM_res[i]);
    }
    //printf("\n----TEST 1 OK------");

    for(int i = 0; i < numElements; i++)
    {
        if(arr_GPU_res2[i] - arr_RAM_res[i] > 0.000001)
            printf("ERROR! i=%d %g %g\n", i, arr_GPU_res2[i], arr_RAM_res[i]);
    }
    //printf("\n----TEST 2 OK------");

    for(int i = 0; i < numElements; i++)
    {
        if(arr_GPU_res3[i] - arr_RAM_res[i] > 0.000001)
            printf("ERROR! i=%d %g %g\n", i, arr_GPU_res3[i], arr_RAM_res[i]);
    }
    //printf("\n----TEST 3 OK------");

    for(int i = 0; i < numElements; i++)
    {
        if(arr_GPU_res4[i] - arr_RAM_res[i] > 0.000001)
            printf("ERROR! i=%d %g %g\n", i, arr_GPU_res4[i], arr_RAM_res[i]);
    }
    //printf("\n----TEST 4 OK------");  

    for(int i = 0; i < numElements; i++)
    {
        if(arr_GPU_res5[i] - arr_RAM_res[i] > 0.000001)
            printf("ERROR! i=%d %g %g\n", i, arr_GPU_res5[i], arr_RAM_res[i]);
    }
    //printf("\n----TEST 5 OK------");

    for(int i = 0; i < numElements; i++)
    {
        if(arr_GPU_res6[i] - arr_RAM_res[i] > 0.000001)
            printf("ERROR! i=%d %g %g\n", i, arr_GPU_res6[i], arr_RAM_res[i]);
    }
    //printf("\n----TEST 6 OK------");

    for(int i = 0; i < numElements; i++)
    {
        if(arr_GPU_res7[i] - arr_RAM_res[i] > 0.000001)
            printf("ERROR! i=%d %g %g\n", i, arr_GPU_res7[i], arr_RAM_res[i]);
    }
    //printf("\n----TEST 7 OK------");

    for(int i = 0; i < numElements; i++)
    {
        if(arr_GPU_res8[i] - arr_RAM_res[i] > 0.000001)
            printf("ERROR! i=%d %g %g\n", i, arr_GPU_res8[i], arr_RAM_res[i]);
    }
    //printf("\n----TEST 8 OK------");

    for(int i = 0; i < numElements; i++)
    {
        if(arr_GPU_res9[i] - arr_RAM_res[i] > 0.000001)
            printf("ERROR! i=%d %g %g\n", i, arr_GPU_res9[i], arr_RAM_res[i]);
    }
*/


    printf("\n----TESTS OK------\n");

    //printArray(arr_RAM_res, numElements);
    //printf("\n----TESTS OK------\n");
    //printArray(arr_GPU_res1, numElements);
    //printf("\n----TESTS OK------\n");
    //printArray(arr_GPU_res2, numElements);
    ////////////////////////////////////////////////////////


    return 0;
}