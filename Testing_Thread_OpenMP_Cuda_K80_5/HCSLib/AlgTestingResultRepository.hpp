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
    bool Add(AlgTestingResult& data)
    {
        try
        {
            std::string filePath = GetFullPath();
            std::ofstream fout(filePath, std::ios::app);
            fout << data;
            fout.close();
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            return false;
        }               

        return true;
    }

    /// @brief Запуск по команде меню
    void Add()
    {
        AlgTestingResult res;
        res.id = GetLastId() + 1;
        res.compSystemId = 222;
        res.algorithmId = 333;

        bool result = Add(res);

        if(result)
            std::cout << "Item with id=" + std::to_string(res.id) + " added." << std::endl;
        else
            std::cout << "Error in adding item with id=" + std::to_string(res.id) << std::endl;
    }


    AlgTestingResult Find(size_t id)
    {
        std::ifstream fin(GetFullPath());

        if(!fin.is_open())
            throw std::runtime_error("File not opened!");
        
        while(!fin.eof())
        {
            std::string line;
            std::getline(fin,line);
            //std::cout << line << std::endl;
            if(line.size() < 2)
                continue;
            AlgTestingResult algTestingResult(line);
            if(algTestingResult.id == id)
                return algTestingResult;
        }

        throw std::runtime_error("AlgTestingResult entry with id=" + std::to_string(id) + " not found!");
    }

    /// @brief Поиск записи в файле по команде меню
    void Find()
    {
        size_t id = ConsoleHelper::GetUnsignedLongLongFromUser("Enter id: ");
        //std::cout << "ull: " << id << std::endl;

        try
        {
            AlgTestingResult entry = Find(id);
            entry.Print();
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }        
    }

    /// @brief Проверка существования записи с указанным id
    bool IsExists(size_t id)
    {
        std::ifstream fin(GetFullPath());
        if(!fin.is_open())
            throw std::runtime_error("File not opened!");        

        while(!fin.eof())
        {
            std::string line;
            std::getline(fin,line);
            if(line.size() < 2)
                continue;
            std::stringstream obj_ss(line);
            size_t cur_id;
            obj_ss >> cur_id;
            if(cur_id == id)
                return true;
        }
        return false;
    }

    /// @brief Проверка существования записи с указанным id по еоманде меню
    void IsExists()
    {
        size_t id = ConsoleHelper::GetUnsignedLongLongFromUser("Enter id: ");

        bool isExists = IsExists(id);
        if(isExists)            
            std::cout << "Item with id=" + std::to_string(id) + " exists." << std::endl;
        else
            std::cout << "Item with id=" + std::to_string(id) + " not exists."  << std::endl;
    }
};

