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

/*void __cpuid(int CPUInfo[4], int InfoType)
{
		try
		{
			__asm
			{
				mov    esi, CPUInfo
				mov    eax, InfoType
				xor    ecx, ecx
				cpuid
				mov    dword ptr[esi + 0], eax
				mov    dword ptr[esi + 4], ebx
				mov    dword ptr[esi + 8], ecx
				mov    dword ptr[esi + 12], edx
			}
		}
		catch (...) { return; }
}

void GetCpu(std::string& output)
{
		try
		{
			int CPUInfo[4] = { -1 };
			__cpuid(CPUInfo, 0x80000000);
			unsigned int nExIds = CPUInfo[0];

			char CPUBrandString[0x40] = { 0 };
			for (unsigned int i = 0x80000000; i <= nExIds; ++i)
			{
				__cpuid(CPUInfo, i);
				if (i == 0x80000002)
				{
					memcpy(CPUBrandString,
						CPUInfo,
						sizeof(CPUInfo));
				}
				else if (i == 0x80000003)
				{
					memcpy(CPUBrandString + 16,
						CPUInfo,
						sizeof(CPUInfo));
				}
				else if (i == 0x80000004)
				{
					memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
				}
			}

			output = CPUBrandString;
		}
		catch (...) { return; }
}*/

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

////////////////////////////////
int main()
{   

    try
    {
        /*VectorCpu<float> vcpu(15);
        vcpu.InitVectorByRange(-6.7, 5.9);
        vcpu.Print();
        VectorGpu<float> vgpu(vcpu);
        vgpu.Print();*/
        
        //getchar();

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