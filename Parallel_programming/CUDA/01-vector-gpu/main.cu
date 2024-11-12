// nvcc main.cu -o app -O3 -allow-unsupported-compiler -DDEBUG
// ./app
// -allow-unsupported-compiler - в случае конфликта версий nvcc и cl
// -DDEBUG                     - режим вывода отладочной информации
// --- Profiling ---
// nvprof ./app
// nvprof ./app --print-gpu-trace
// nvprof --analysis-metrics -o app.nvprof ./app --benchmark -numdevices=1 -i=1

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include "GpuLib/_include.cu"

// Выполняет набор тестов производительности для VectorCpu::Sum
void StartTestVectorCpuSum(TestConfig conf)
{
    std::string fileName = conf.dir + "/" + conf.fileCaption + "-CPU.txt";
    std::cout << "Opening " << fileName << "...";
    std::ofstream out(fileName);
    if(!out.is_open())
    {
        throw std::runtime_error("File " + fileName + " not created!");
    }    
    std::cout << "OK" << std::endl;

    out << "--- StartTestVectorCpuSum Report ---" << std::endl;
    
    std::cout << "CPU specification:" << std::endl;
    //std::string str = "";
    //GetCpu(str); //Intel(R) Core(TM) i5-7400 CPU @ 3.00GHz -----
    //std::cout << str << std::endl;

    std::cout << "Starting test Sum..." << std::endl;
    for(auto vectorLength = conf.lengthStart;
            vectorLength <=conf.lengthEnd;
            (conf.lengthStep == 0) ? vectorLength *= conf.lengthMult : vectorLength += conf.lengthStep )
    {        

        out << "VectorCpu._size = " << vectorLength << std::endl;

        std::cout << "Init VectorCPU: " << vectorLength << "...";
        VectorCpu<> v1{vectorLength};        
        v1.InitVectorByScalar(0.001);
        //getchar();
        //v1.Print(); getchar();
        std::cout << "OK" << std::endl;

        out << "threadsNum; avg, mks; minValue; median; percentile_95; maxValue; stdDev" << std::endl;
       
        for(int threadsNum = conf.cpuThreadsMin; threadsNum <= conf.cpuThreadsMax; threadsNum++)
        {
            std::cout << "threadsNum = " << threadsNum << "; ";
            std::vector<FuncResultScalar<double>> results{conf.iterNum};
            for(int iter = 0; iter < conf.iterNum; iter++)
            {
                auto res = v1.Sum(threadsNum);
                results[iter] = res;
                //std::cout << results[iter].Status << " | " << results[iter].Result << " | " << results[iter].Time_mks << std::endl;
            }
            CalculationStatistics stat(results);

            //out << blocksNum << "; " << threadsNum << "; " << results[0].Status << "; " << results[0].Result << "; "<< results[0].Time_mks << std::endl;
            out     << threadsNum << "; "
                    << stat.avg << "; " 
                    << stat.minValue << "; "
                    << stat.median << "; "
                    << stat.percentile_95 << "; "
                    << stat.maxValue << "; "
                    << stat.stdDev << std::endl;
        }
        std::cout << std::endl;
        
        v1.Clear_data();
    }
    out.close();
}

// Выполняет набор тестов производительности для VectorGpu::Sum
void StartTestVectorGpuSum(TestConfig conf)
{
    std::string fileName = conf.dir + "/" + conf.fileCaption + "-GPU.txt";
    std::cout << "Opening " << fileName << "...";
    std::ofstream out(fileName);
    if(!out.is_open())
    {
        throw std::runtime_error("File " + fileName + " not created!");
    }    
    std::cout << "OK" << std::endl;

    out << "--- StartTestVectorGpuSum Report ---" << std::endl;
    
    std::cout << "GPU specification:" << std::endl;
    CudaHelper<double>::WriteGpuSpecs(out);

    std::cout << "Starting test Sum..." << std::endl;
    for(auto vectorLength = conf.lengthStart;
            vectorLength <=conf.lengthEnd;
            (conf.lengthStep == 0) ? vectorLength *= conf.lengthMult : vectorLength += conf.lengthStep )
    {        

        out << "VectorGpu._size = " << vectorLength << std::endl;

        std::cout << "Init VectorGPU: " << vectorLength << "...";
        VectorGpu<> v1{vectorLength};        
        v1.InitVectorByScalar(0.001);
        std::cout << "OK" << std::endl;

        out << "blocksNum; threadsNum; avg, mks; minValue; median; percentile_95; maxValue; stdDev" << std::endl;
        for(int blocksNum = conf.gpuBlocksMin; blocksNum <= conf.gpuBlocksMax; blocksNum++)
        {
            std::cout << "blocksNum = " << blocksNum << ": ";
            for(int threadsNum = conf.gpuThreadsMin; threadsNum <= conf.gpuThreadsMax; threadsNum++)
            {
                std::cout << threadsNum << "; ";
                std::vector<FuncResultScalar<double>> results{conf.iterNum};
                for(int iter = 0; iter < conf.iterNum; iter++)
                {
                    auto res = v1.Sum(blocksNum, threadsNum);
                    results[iter] = res;
                    //std::cout << results[iter].Status << " | " << results[iter].Result << " | " << results[iter].Time_mks << std::endl;
                }
                CalculationStatistics stat(results);

                //out << blocksNum << "; " << threadsNum << "; " << results[0].Status << "; " << results[0].Result << "; "<< results[0].Time_mks << std::endl;
                out << blocksNum << "; " << threadsNum << "; "
                    << stat.avg << "; " 
                    << stat.minValue << "; "
                    << stat.median << "; "
                    << stat.percentile_95 << "; "
                    << stat.maxValue << "; "
                    << stat.stdDev << std::endl;
            }
            std::cout << std::endl;
        }
        
        out.close();
        v1.Clear_dev_data();
    }
}

template<typename T>
void func(T var)
{    
    std::cout << var << std::endl;
}

template<typename T>
void func2(T* var)
{    
    std::cout << *var << std::endl;
    *var += *var;
    std::cout << *var << std::endl;
}

template<typename T>
void func3(T& var)
{    
    std::cout << var << std::endl;
    var += var;
    std::cout << var << std::endl;
}

////////////////////////////////
int main()
{   

    try
    {
        double val = 1.234;        
        std::thread t1(
            [] (auto val)
            {
                func(val);
            },
            val // Передаём параметры для лямбда-функции в потоке
        );
        t1.join();        
        getchar();

        double val2 = 12.34;
        std::cout << "before: val2 = " << val2 << std::endl;       
        std::thread t2(
            [] (auto val2)
            {
                func2(&val2);
            },
            std::ref(val2) // Передаём параметры для лямбда-функции в потоке
        );
        t2.join();
        std::cout << "after: val2 = " << val2 << std::endl;
        getchar();

        double val3 = 123.4;
        std::cout << "before: val3 = " << val3 << std::endl;       
        std::thread t3(
            [] (auto& val3)
            {
                func3(val3);
            },
            std::ref(val3) // Передаём параметры для лямбда-функции в потоке
        );
        t3.join();
        std::cout << "after: val3 = " << val3 << std::endl;
        getchar();

        VectorCpu<float> vcpu(15);
        vcpu.InitVectorByRange(-6.7, 5.9);
        vcpu.Print();
        VectorGpu<float> vgpu(vcpu);
        vgpu.Print();
        
        getchar();

        // Запускаем тесты функции суммирования на CPU
        StartTestVectorCpuSum(TestConfig{});

        // Запускаем тесты функции суммирования на GPU
        StartTestVectorGpuSum(TestConfig{});


        VectorGpu<> v1{64000000};
        //v1.InitVectorByRange(0.1, 5.9);
        v1.InitVectorByScalar(0.001);
        //VectorGpuHelper<double>::Print(v1);
        //VectorGpuHelper<double>::Print(v1, 2);
        VectorGpuHelper<double>::Print(v1, 0, 10);
    
        for(int i = 1; i <= 5; i++)
        {
            auto res = v1.Sum(20,20);
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