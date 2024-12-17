#pragma once


/// @brief Рузультаты тестового запуска алгоритма
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

