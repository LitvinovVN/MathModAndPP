#pragma once

#include <sstream>

/// @brief Результаты тестового запуска алгоритма
struct AlgTestingResult
{
    // УИД тестового запуска
    size_t id = 0;
    // УИД вычислительной системы
    unsigned compSystemId = 0;
    // УИД группы задач (вектор, вектор-матрица и пр) | TaskGroup
    unsigned taskGroupId = 0;
    // УИД задачи (сумма элементов вектора, скалярное произведение векторов и пр) | Task
    unsigned taskId = 0;
    // Размерность задачи (кол-во )
    TaskDimensions taskDimensions {};
    // УИД алгоритма
    unsigned algorithmId = 0;
    // Длина типа данных, используемая в алгоритме (float: 4; double: 8)
    unsigned algorithmDataTypeLength = 0;
    // Тип алгоритма:
    // 1 - последовательный CPU
    // 2 - последовательный GPU
    // 3 - параллельный CPU std::thread
    // 4 - параллельный CPU OpenMP
    // 5 - параллельный CUDA
    unsigned algorithmType = 0;
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
        //std::cout <<"Source String to Split: " << strToParse << "\n\n";
        obj_ss >> id;
        obj_ss >> compSystemId;
        obj_ss >> taskGroupId;
        obj_ss >> taskId;
        obj_ss >> taskDimensions.dim;
        obj_ss >> taskDimensions.x;
        obj_ss >> taskDimensions.y;
        obj_ss >> taskDimensions.z;
        obj_ss >> taskDimensions.t;
        obj_ss >> algorithmId;
        obj_ss >> algorithmDataTypeLength;
        obj_ss >> algorithmType;
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

        //std::cout << "Parsed: ";
        //Print();
    }

    void Print()
    {
        std::cout << "id: " << id << "; ";
        std::cout << "compSystemId: " << compSystemId << "; ";
        std::cout << "taskGroupId: "     << taskGroupId << "; ";
        std::cout << "taskId: "          << taskId << "; ";
        std::cout << "taskDimensions.dim: " << taskDimensions.dim << "; ";
        std::cout << "taskDimensions.x: "   << taskDimensions.x << "; ";
        std::cout << "taskDimensions.y: "   << taskDimensions.y << "; ";
        std::cout << "taskDimensions.z: "   << taskDimensions.z << "; ";
        std::cout << "taskDimensions.t: "   << taskDimensions.t << "; ";
        std::cout << "algorithmId: "             << algorithmId << "; ";
        std::cout << "algorithmDataTypeLength: " << algorithmDataTypeLength << "; ";
        std::cout << "algorithmType: "           << algorithmType << "; ";
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
             << data.taskGroupId << " "
             << data.taskId << " ";
        fout << data.taskDimensions << " ";
        fout << data.algorithmId << " "
             << data.algorithmDataTypeLength << " "
             << data.algorithmType << " "
             << data.threadsNumCpu << " "
             << data.threadBlocksNumGpu << " "
             << data.threadsNumGpu << " ";
        fout << data.calculationStatistics;
        fout << "\n";

        return fout;
    }
};

