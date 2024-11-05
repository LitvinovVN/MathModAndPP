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
#include "GpuLib/_include.cu"

// Структура для хранения параметров запуска теста производительности
struct TestConfig
{
    std::string dir = "results";
    std::string fileCaption = "test";
    size_t lengthStart = 1000000;
    size_t lengthEnd   = 2000000;
    size_t lengthStep  = 0;
    size_t lengthMult  = 2;
    // Количество блоков GPU
    unsigned gpuBlocksMin = 1;
    unsigned gpuBlocksMax = 20;
    // Количество потоков GPU
    unsigned gpuThreadsMin = 1;
    unsigned gpuThreadsMax = 50;
    // Количество повторов
    unsigned iterNum = 10;    
};

// Статистические параметры результатов эксперимента
struct CalculationStatistics
{
    // Количество запусков численного эксперимента
    unsigned numIter;
    // Минимальное значение
    double minValue;
    // Максимальное значение
    double maxValue;
    // Среднее арифметическое
    double avg;
    // Медиана
    double median;
    // 
};

CalculationStatistics CalculateStatistics(std::vector<FuncResultScalar<double>> results)
{
    CalculationStatistics stat;
    return stat;
}


// Выполняет набор тестов производительности для VectorGpu::Sum
void StartTestVectorGpuSum(TestConfig conf)
{
    std::string fileName = conf.dir + "/" + conf.fileCaption + ".txt";
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

        out << "blocksNum; threadsNum; Status; Result; Time_mks" << std::endl;
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
                CalculationStatistics stat = CalculateStatistics(results);

                out << blocksNum << "; " << threadsNum << "; " << results[0].Status << "; " << results[0].Result << "; "<< results[0].Time_mks << std::endl;
            }
            std::cout << std::endl;
        }
        
        out.close();
        v1.Clear_dev_data();
    }
}

////////////////////////////////
int main()
{   

    try
    {
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