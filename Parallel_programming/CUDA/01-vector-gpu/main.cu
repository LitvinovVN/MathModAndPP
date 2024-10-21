// nvcc main.cu -o app -allow-unsupported-compiler
// ./app

#include <iostream>
#include "GpuLib/_include.cu"

////////////////////////////////
int main()
{
    VectorGpu<> v1(10);
    v1.initVectorByRange(0.1,0.5);
    VectorGpuHelper<double>::print(v1);

    kernel_vector<double><<<2,2>>>(v1);
    cudaDeviceSynchronize();
}