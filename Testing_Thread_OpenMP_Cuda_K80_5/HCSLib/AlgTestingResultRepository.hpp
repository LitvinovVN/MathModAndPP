#pragma once


/// @brief Репозиторий результатов тестовых запусков алгоритмов
class AlgTestingResultRepository
{
    std::string dir_name = "AlgTestingResultRepository";// Каталог с данными
    std::string file_name = "data.txt";  // Файл с данными
    std::vector<AlgTestingResult> cache; // Кэш данных в памяти
    // Ключ - compSystemId;
    // значение - вектор индексов УИД тестовых запусков
    // вычислительной системы compSystemId
    std::map<unsigned, std::vector<size_t>> compSystemIndex;

    /// @brief Проверка существования каталогов
    void CheckDirectories()
    {        
        if(!FileSystemHelper::IsDirExists(dir_name))
            FileSystemHelper::CreateDir(dir_name);
    }

    public:
    AlgTestingResultRepository()
    {
        CheckDirectories();
    }

    bool Write(AlgTestingResult& data)
    {
        std::string filePath = FileSystemHelper::CombinePath(dir_name, "1.txt");
        std::ofstream fout(filePath, std::ios::app);
        fout << data.id << " "
             << data.compSystemId << " "
             << data.taskGroupId << " "
             << data.taskId << " "
             << data.algorithmId << " "
             << data.algorithmDataTypeLength << " "
             << data.algorithmType << " "
             << data.threadsNumCpu << " "
             << data.threadBlocksNumGpu << " "
             << data.threadsNumGpu << " "
             << data.calculationStatistics.minValue << " "
             << data.calculationStatistics.median << " "
             << data.calculationStatistics.avg << " "
             << data.calculationStatistics.percentile_95 << " "
             << data.calculationStatistics.maxValue << " "
             << data.calculationStatistics.stdDev << " "
             << data.calculationStatistics.numIter << " "
             << "\n";
        fout.close();

        return true;
    }

    /// @brief Запуск по команде меню
    void Write()
    {
        AlgTestingResult res;
        res.id = 111;
        res.compSystemId = 222;

        Write(res);
    }
};

