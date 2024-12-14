#pragma once


/// @brief Конфигурация приложения
class AppConfig
{
    bool isInitialized{true};// Статус инициализации объекта конфигурации приложения
    std::string message{"AppConfig status: OK"};// Строка с описанием статуса инициализации объекта конфигурации приложения

    std::string fileConfig{"config.txt"};// Имя файла конфигурации

    int compSystemId {1};// Идентификатор вычислительной системы
    std::string dir_calcTestResults{"CalcTestResults"}; // Каталог с данными о результатах вычислительных экспериментов
    std::string dir_computingSystemRepository{"ComputingSystemRepository"}; // Каталог с данными о вычислительных системах

    /// @brief Проверка существования каталогов
    void CheckDirectories()
    {        
        if(!FileSystemHelper::IsDirExists(dir_calcTestResults))
            FileSystemHelper::CreateDir(dir_calcTestResults);

        if(!FileSystemHelper::IsDirExists(dir_computingSystemRepository))
            FileSystemHelper::CreateDir(dir_computingSystemRepository);
    }

    /// @brief Считывает конфигурацию из файла
    /// @return true - успех; false - наличие ошибок считывания
    bool ReadConfigFile()
    {
        std::ifstream f(fileConfig);
        
        if(!f.is_open())
        {            
            message = "Config file \"" + fileConfig + "\" is not opened!";
            return false;
        }

        // Проверка формата файла
        std::string str;
        f >> str;
        if (str != "AppConfig")
        {            
            message = "Config file \"" + fileConfig + "\" format is not AppConfig!";
            return false;
        }

        // Считываем пары "Параметр Значение"
        std::string param, value;
        while(f >> param >> value)
        {
            //std::cout << param << ": " << value << std::endl;
            if(param == "compSystemId")
            {
                try
                {
                    compSystemId = std::stoi(value);
                    //std::cout << "!!! " << compSystemId << std::endl;
                }
                catch(const std::exception& e)
                {                    
                    message = "Config file \"" + fileConfig + "\": compSystemId parameter is not recognized!";
                    return false;
                }                
            }
            else if (param == "dir_calcTestResults")
                dir_calcTestResults = value;
            else if (param == "dir_computingSystemRepository")
                dir_computingSystemRepository = value;
            else
            {
                message = "Config file \"" + fileConfig + "\": parameter \"" + param + "\" with value \"" + value + "\" is not recognized!";
                return false;
            }
        }

        return true;
    }

public:
    AppConfig()
    {
        // Проверка существования каталогов
        CheckDirectories();
    }

    AppConfig(std::string fileName)
    {
        if(!FileSystemHelper::IsFileExists(fileName))
        {
            isInitialized = false;
            message = "Error! Config file \"" + fileName + "\" not found!";
            return;
        }
        fileConfig = fileName;

        bool result = ReadConfigFile();
        if(!result)
        {
            isInitialized = false;            
            return;
        }

        // Проверка существования каталогов
        CheckDirectories();
    }

    bool IsInitialized() const
    {
        return isInitialized;
    }

    std::string GetMessage() const
    {
        return message;
    }

    std::string GetDirComputingSystemRepository() const
    {
        return dir_computingSystemRepository;
    }

    void Print()
    {
        if(!isInitialized)
        {
            std::cout   << "AppConfig: ["
                        << "NOT INITIALIZED; "
                        << message
                        << "]" << std::endl;
            return;
        }

        std::cout   << "AppConfig: ["
                    << "compSystemId: " << compSystemId << "; "
                    << "dir_calcTestResults: " << dir_calcTestResults << "; "
                    << "dir_computingSystemRepository: " << dir_computingSystemRepository
                    << "]" << std::endl;
    }

};
