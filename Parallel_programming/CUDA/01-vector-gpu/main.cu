// nvcc main.cu -o app -allow-unsupported-compiler
// ./app

#include <iostream>
#include "GpuLib/_include.cu"

////////////////////////////////
int main()
{
    try
    {
        VectorGpu<> v1{20};
        v1.initVectorByRange(1, 20);
        VectorGpuHelper<double>::Print(v1);
        VectorGpuHelper<double>::Print(v1, 2);
        VectorGpuHelper<double>::Print(v1, 2, 3);
    
        auto sum = v1.Sum(2,3);
        std::cout << "sum = " << sum << std::endl;

        //VectorGpuHelper<double>::Print(v1, 9, 2);

        kernel_vector<double><<<2,2>>>(v1);
        cudaDeviceSynchronize();

        v1.Clear_dev_data();
    }
    catch(const std::exception& ex)
    {
        std::cout << "Exception! " << ex.what()<< std::endl;
    }

    
}