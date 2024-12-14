#pragma once


/// @brief Репозиторий сведений о вычислительных системах
class ComputingSystemRepository
{
    std::string dir_name = "ComputingSystemRepository";// Имя каталога со сведениями о вычислительных системах
    std::string file_name = "List.txt";// Имя файла со сведениями о вычислительных системах

    std::vector<int> computerSystemIds;// Вектор идентификаторов вычислительных систем

    /// @brief Проверка существования каталогов
    void CheckDirectories()
    {        
        if(!FileSystemHelper::IsDirExists(dir_name))
            FileSystemHelper::CreateDir(dir_name);
    }

    void CheckFiles()
    {
        if(!FileSystemHelper::IsFileExists(dir_name, file_name))
        {
            bool result = FileSystemHelper::CreateFile(dir_name, file_name, "ComputingSystemRepository");
            if (!result)
            {
                std::cerr << "File " + file_name + " in directory " + dir_name + " is not created!";
                exit(-1);
            }            
        }
    }

    /// @brief Считывает содержимое файла со сведениями о вычислительных системах
    /// @return 
    bool ReadFile()
    {
        std::string filePath = FileSystemHelper::CombinePath(dir_name, file_name);

        std::ifstream f(filePath);
        
        if(!f.is_open())
        {            
            std::string message = "File \"" + filePath + "\" is not opened!";
            std::cerr << message << std::endl;
            return false;
        }

        // Проверка формата файла
        std::string str;
        f >> str;
        if (str != "ComputingSystemRepository")
        {            
            std::string message = "File \"" + filePath + "\" format is not AppConfig!";
            std::cerr << message << std::endl;
            return false;
        }

        // Считываем пары "Параметр Значение"
        int value;
        while(f >> value)
        {
            //std::cout << value << std::endl;
            computerSystemIds.push_back(value);
        }

        return true;
    }

    /// @brief Записывает new_id в конец файла 
    /// @param new_id 
    /// @return 
    bool AddIdToFile(const int new_id)
    {
        std::string filePath = FileSystemHelper::CombinePath(dir_name, file_name);

        std::ofstream fout(filePath,std::ios::app);
        if(!fout.is_open())
        {
            return false;
        }

        fout << '\n' << new_id ;

        fout.close();
        return true;
    }

public:
    ComputingSystemRepository()
    {
        CheckDirectories();
        CheckFiles();
        ReadFile();
    }

    ComputingSystemRepository(std::string dir_name)
        : dir_name(dir_name)
    {
        CheckDirectories();
        CheckFiles();
        ReadFile();
    }

    bool IsExists(int computingSystemId) const
    {
        for(auto& id : computerSystemIds)
        {
            if(id == computingSystemId)
                return true;
        }

        return false;
    }

    bool TryAddComputingSystem(ComputingSystem& computingSystem)
    {
        int new_id = computingSystem.GetId();
        // Если уже есть информация о вычислительной системе
        // с таким идентификатором, информацию не добавляем
        // и возвращаем false
        if (IsExists(new_id))
            return false;

        // Записать данные о выч. системе в каталог dir_name
        computingSystem.Serialize(dir_name);

        AddIdToFile(new_id);

        computerSystemIds.push_back(new_id);

        return true;
    }

    ComputingSystem GetComputingSystem(int id)
    {
        if(!IsExists(id))
            throw std::logic_error("Computing system not found!");

        return ComputingSystem::Deserialize(dir_name, id);
    }

    /// @brief 2 Print config
    void PrintConfig()
    {
        std::cout << "dir_name: "  << dir_name  << "; ";
        std::cout << "file_name: " << file_name << std::endl;
    }

    /// @brief 3 Print computing system list
    void PrintList()
    {
        std::cout << "Computing system ids: [";
        for(auto& id : computerSystemIds)
            std::cout << id << " ";
        std::cout << "]" << std::endl;
    }
    
    /// @brief 4 Print computing system details
    void PrintDetails()
    {
        std::cout << "PrintDetails()" << std::endl;
        int id = ConsoleHelper::GetIntFromUser("Enter computing system id: ");

        if(!IsExists(id))
        {
            std::cout << "Not found!" << std::endl;
            return;
        }

        ComputingSystem computingSystem = GetComputingSystem(id);
        computingSystem.Print();
    }

    /// @brief 5 Add computing system
    void Add()
    {
        std::cout << "Add()" << std::endl;
        ComputingSystem computingSystem = ComputingSystem::GetDataFromUser();        

        if(TryAddComputingSystem(computingSystem))
        {
            std::cout << "Computing system " << computingSystem.GetId() << " added." << std::endl;
        }
        else
        {
            std::cout << "Error in adding computing system " << computingSystem.GetId() << "!" << std::endl;
        }
    }

    /// @brief 6 Change computing system
    void Change()
    {
        std::cout << "Change()" << std::endl;
    }

    /// @brief 7 Remove computing system
    void Remove()
    {
        std::cout << "Remove()" << std::endl;
    }

    /// @brief 8 Is computing system exists
    void IsExists()
    {
        int compSystemId = ConsoleHelper::GetIntFromUser("Enter computing system id: ", "Error! Enter integer number!");                
        bool isExists = IsExists(compSystemId);

        std::cout << "id: "       << compSystemId << "; ";
        std::cout << "isExists: " << isExists     << std::endl;
    }

};
