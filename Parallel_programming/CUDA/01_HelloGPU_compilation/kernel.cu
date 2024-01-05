#include "stdio.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void cuda_hello_kernel(int* a){
    *a = 555;
    printf("Hello World from GPU!\n");    
}

extern int cuda_hello() {
    printf("cuda_hello() started...\n"); 
    int* dev_a;
    gpuErrchk(cudaMalloc((void**)&dev_a, sizeof(int)));
    cuda_hello_kernel<<<1,1>>>(dev_a);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    int* a = (int*)malloc(sizeof(int));
    *a = -11;
    printf("a = %d\n", *a);
    cudaMemcpy(a, dev_a, sizeof(int), cudaMemcpyDeviceToHost);
    printf("a = %d\n", *a);
    
    printf("cuda_hello() completed!\n"); 
    return 0;
}