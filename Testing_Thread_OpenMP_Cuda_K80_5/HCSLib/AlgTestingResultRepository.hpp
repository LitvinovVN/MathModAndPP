#pragma once


/// @brief Репозиторий результатов тестовых запусков алгоритмов
class AlgTestingResultRepository
{
    bool isInitialized = false;
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
        if (!isInitialized) return;

        if(!FileSystemHelper::IsDirExists(dir_name))
            FileSystemHelper::CreateDir(dir_name);
    }

    public:
    AlgTestingResultRepository(bool isInitialized = true)
        : isInitialized(isInitialized)
    {
        CheckDirectories();
    }

    AlgTestingResultRepository(std::string dir_name)
        : dir_name(dir_name)
    {
        isInitialized = true;
        CheckDirectories();
    }

    /// @brief Считывает значение пути к каталогу с данными
    /// @param dir 
    std::string Get_dir_name()
    {
        return dir_name;
    }

    /// @brief Устанавливает значение пути к каталогу с данными
    /// @param dir 
    void Set_dir_name(std::string dir)
    {
        dir_name = dir;
    }

    /// @brief Записывает результаты тестового запуска в файл
    /// @param data 
    /// @return 
    bool Write(AlgTestingResult& data)
    {
        std::string filePath = FileSystemHelper::CombinePath(dir_name, "1.txt");
        std::ofstream fout(filePath, std::ios::app);
        fout << data;
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

