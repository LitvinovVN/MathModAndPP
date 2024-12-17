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

    void PrintConfig()
    {
        std::cout   << "isInitialized: " << isInitialized << "; "
                    << "dir_name: " << dir_name << "; "
                    << "file_name: " << file_name << std::endl;
    }

    /// @brief Считывает значение пути к каталогу с данными
    /// @param dir 
    std::string Get_dir_name()
    {
        return dir_name;
    }

    /// @brief Возвращает полный путь к файлу с данными 
    std::string GetFullPath()
    {
        return FileSystemHelper::CombinePath(dir_name, file_name);
    }

    /// @brief Возвращает наибольший использованный УИД тестового запуска
    /// @return 
    size_t GetLastId()
    {
        std::ifstream fin(GetFullPath());

        if(!fin.is_open())
            throw std::runtime_error("File not opened!");

        size_t id_max = 0;

        while(!fin.eof())
        {
            std::string line;
            std::getline(fin,line);
            //std::cout << line << std::endl;
            if(line.size() < 2)
                continue;
            AlgTestingResult algTestingResult(line);
            if(algTestingResult.id > id_max)
                id_max = algTestingResult.id;
        }

        return id_max;
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
        std::string filePath = GetFullPath();
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

    /// @brief Поиск записи в файле по команде меню
    void Find()
    {
        size_t id = ConsoleHelper::GetUnsignedLongLongFromUser("Enter id: ");
        std::cout << "ull: " << id << std::endl;
        AlgTestingResult entry = TryFind(id);
    }
};

