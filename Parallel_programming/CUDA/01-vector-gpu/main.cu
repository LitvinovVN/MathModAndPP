// nvcc main.cu -o app -allow-unsupported-compiler -DDEBUG -O3
// ./app
// -allow-unsupported-compiler - в случае конфликта версий nvcc и cl
// -DDEBUG                     - режим вывода отладочной информации

#include <iostream>
#include "GpuLib/_include.cu"

////////////////////////////////
int main()
{
    try
    {
        VectorGpu<> v1{200000000};
        //v1.InitVectorByRange(0.1, 5.9);
        v1.InitVectorByScalar(0.001);
        //VectorGpuHelper<double>::Print(v1);
        //VectorGpuHelper<double>::Print(v1, 2);
        VectorGpuHelper<double>::Print(v1, 0, 10);
    
        for(int i = 1; i <= 10; i++)
        {
            auto res = v1.Sum(2,10);
            std::cout << i << ", " << 10 << ": ";
            res.Print();
        }
        

        
        v1.Clear_dev_data();
    }
    catch(const std::exception& ex)
    {
        std::cout << "Exception! " << ex.what()<< std::endl;
    }

    
}