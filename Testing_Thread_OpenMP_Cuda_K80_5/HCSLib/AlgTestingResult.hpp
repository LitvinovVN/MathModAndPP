#pragma once

#include <sstream>

/// @brief Результаты тестового запуска алгоритма
struct AlgTestingResult
{
    // УИД тестового запуска
    size_t id = 0;
    // УИД вычислительной системы
    unsigned compSystemId = 0;    
    // УИД алгоритма
    unsigned algorithmId = 0;    
    // Количество потоков CPU
    unsigned threadsNumCpu = 0;
    // Количество блоков GPU
    unsigned threadBlocksNumGpu = 0;
    // Количество нитей GPU в блоке
    unsigned threadsNumGpu = 0;
    // Статистики вычислительного эксперимента
    CalculationStatistics calculationStatistics;

    AlgTestingResult()
    {
    }

    AlgTestingResult(std::string strToParse)
    {
        std::stringstream obj_ss(strToParse);        
        obj_ss >> id;
        obj_ss >> compSystemId;        
        obj_ss >> algorithmId;
        obj_ss >> threadsNumCpu;
        obj_ss >> threadBlocksNumGpu;
        obj_ss >> threadsNumGpu;
        obj_ss >> calculationStatistics.numIter;
        obj_ss >> calculationStatistics.minValue;
        obj_ss >> calculationStatistics.median;
        obj_ss >> calculationStatistics.avg;
        obj_ss >> calculationStatistics.percentile_95;
        obj_ss >> calculationStatistics.maxValue;
        obj_ss >> calculationStatistics.stdDev;
    }

    void Print()
    {
        std::cout << "id: " << id << "; ";
        std::cout << "compSystemId: " << compSystemId << "; ";
        std::cout << "algorithmId: "             << algorithmId << "; ";
        std::cout << "threadsNumCpu: "      << threadsNumCpu << "; ";
        std::cout << "threadBlocksNumGpu: " << threadBlocksNumGpu << "; ";
        std::cout << "threadsNumGpu: "      << threadsNumGpu << "; ";
        calculationStatistics.Print();
        std::cout << std::endl;
    }

    friend std::ofstream& operator<<(std::ofstream& fout, const AlgTestingResult& data)
    {
        fout << data.id << " "
             << data.compSystemId << " "
             << data.algorithmId << " "
             << data.threadsNumCpu << " "
             << data.threadBlocksNumGpu << " "
             << data.threadsNumGpu << " ";
        fout << data.calculationStatistics;
        fout << "\n";

        return fout;
    }
};

