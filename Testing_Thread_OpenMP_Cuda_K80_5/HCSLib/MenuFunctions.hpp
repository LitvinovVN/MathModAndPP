#pragma once


/// @brief Функции меню
struct MenuFunctions
{    
    /// @brief Выводит параметры GPU
    static void PrintGpuParameters()
    {
        CudaHelper::PrintCudaDeviceProperties();
    }

    /// @brief Выводит в консоль список поддерживаемых библиотек
    static void PrintLibSupport()
    {
        // Определяем перечень поддерживаемых библиотек
        LibSupport support;
        support.Print();// Выводим список поддерживаемых библиотек
    }

    /// @brief Тестирование функций класса ArrayHelper
    static void Testing_TestArrayHelper()
    {
        if(TestArrayHelper())
            std::cout << "TestArrayHelper correct!" << std::endl;
        else
            std::cout << "TestArrayHelper not correct!" << std::endl;
    }

    /// @brief Записывает параметры видеокарт в текстовый файл gpu-specs.txt
    static void WriteGpuSpecsToTxtFile()
    {
        int cudaDeviceNumber = CudaHelper::GetCudaDeviceNumber();
        std::cout << "Cuda devices number: " << cudaDeviceNumber << std::endl;
        //CudaHelper::PrintCudaDeviceProperties();

        if(cudaDeviceNumber > 0)
        {
            for(int i = 0; i < cudaDeviceNumber; i++)
            {
                auto devProps = CudaHelper::GetCudaDeviceProperties();
                devProps.Print();
            }
            
            std::ofstream f("gpu-specs.txt");
            CudaHelper::WriteGpuSpecs(f);
            f.close();
        }
    }

    /// @brief Запускает тест работоспособности VectorGpu
    static void Testing_TestVectorGpu()
    {
        // Запускаем тест работоспособности VectorGpu
        if(TestVectorGpu())
            std::cout << "VectorGpu correct!" << std::endl;
        else
            std::cout << "VectorGpu not correct!" << std::endl;
    }

    /// @brief Запускает функцию тестирования суммирования массивов
    static void Testing_TestSum()
    {
        // Запускаем функцию тестирования суммирования массивов
        if(TestSum())
            std::cout << "TestSum correct!" << std::endl;
        else
            std::cout << "TestSum not correct!" << std::endl;
    }
    
    /// @brief Конфигурирование приложения
    static void Application_Config(AppConfig& config)
    {
        std::cout   << "----- Application configuration -----\n"
                    << "1 Back to main menu\n"
                    << "2 Print config" << std::endl;

        
        int command = 0;
        while(command != 1)
        {
            std::cout << ">> ";
            std::string commandString;
            std::cin >> commandString;
            
            try
            {
                command = std::stoi(commandString);
            }
            catch(const std::exception& e)
            {
                command = 0;
            }
                        
            switch (command)
            {
            case 1:
                std::cout << "Back to main menu" << std::endl;
                break;
            case 2:
                config.Print();
                break;
            
            default:
                std::cout << "Command not recognized!" << std::endl;
                break;
            }
        }
    
    }

    /// @brief Конфигурирование приложения
    static void ComputingSystemRepository_Config(ComputingSystemRepository& repo)
    {
        std::cout   << "----- Computing system repository configuration -----\n"
                    << "1 Back to main menu\n"
                    << "2 Print config\n"
                    << "3 Print computing system list\n"
                    << "4 Print computing system details\n"
                    << "5 Add computing system\n"
                    << "6 Change computing system\n"
                    << "7 Remove computing system\n"
                    << "8 Is computing system exists\n";

        int command = 0;
        while(command != 1)
        {
            std::cout << ">> ";
            std::string commandString;
            std::cin >> commandString;
            
            try
            {
                command = std::stoi(commandString);
            }
            catch(const std::exception& e)
            {
                command = 0;
            }
                        
            switch (command)
            {
            case 1:
                std::cout << "Back to main menu" << std::endl;
                break;
            case 2:
                std::cout   << "Command: 2 Print config\n";
                repo.PrintConfig();
                break;
            case 3:
                std::cout   << "Command: 3 Print computing system list\n";
                repo.PrintList();
                break;
            case 4:
                std::cout   << "Command: 4 Print computing system details\n";
                repo.PrintDetails();
                break;
            case 5:
                std::cout   << "Command: 5 Add computing system\n";
                repo.Add();
                break;
            case 6:
                std::cout   << "Command: 6 Change computing system\n";
                repo.Change();
                break;
            case 7:
                std::cout   << "Command: 7 Remove computing system\n";
                repo.Remove();
                break;
            case 8:
                std::cout   << "Command: 8 Is computing system exists\n";
                repo.IsExists();
                break;
            default:
                std::cout << "Command not recognized!" << std::endl;
                break;
            }
        }
    
    }

    static void AlgTestingResultRepository_Config(AlgTestingResultRepository& repo)
    {
        std::cout   << "----- AlgTestingResultRepository configuration -----\n"
                    << "1 Back to main menu\n"
                    << "2 Print config\n"
                    << "3 Print AlgTestingResultRepository list\n"
                    << "4 Print AlgTestingResultRepository details\n"
                    << "5 Add test alg result data\n"
                    << "6 Change AlgTestingResultRepository\n"
                    << "7 Remove AlgTestingResultRepository\n"
                    << "8 Is AlgTestingResultRepository exists\n";

        int command = 0;
        while(command != 1)
        {
            std::cout << ">> ";
            std::string commandString;
            std::cin >> commandString;
            
            try
            {
                command = std::stoi(commandString);
            }
            catch(const std::exception& e)
            {
                command = 0;
            }
                        
            switch (command)
            {
            case 1:
                std::cout << "Back to main menu" << std::endl;
                break;
            case 2:
                std::cout   << "Command: 2 Print config\n";
                //repo.PrintConfig();
                break;
            case 3:
                std::cout   << "Command: 3 Print computing system list\n";
                //repo.PrintList();
                break;
            case 4:
                std::cout   << "Command: 4 Print computing system details\n";
                //repo.PrintDetails();
                break;
            case 5:
                std::cout   << "Command: 5 Add test alg result data\n";
                repo.Write();
                break;
            case 6:
                std::cout   << "Command: 6 Change computing system\n";
                //repo.Change();
                break;
            case 7:
                std::cout   << "Command: 7 Remove computing system\n";
                //repo.Remove();
                break;
            case 8:
                std::cout   << "Command: 8 Is computing system exists\n";
                //repo.IsExists();
                break;
            default:
                std::cout << "Command not recognized!" << std::endl;
                break;
            }
        }
    }

    // Тестирование функциональности класса FileSystemHelper
    static void Testing_FileSystemHelper()
    {
        std::cout   << "----- FileSystemHelper -----\n"
                    << "1 Back to main menu\n"
                    << "2 CombinePath\n"
                    << "3 CreateFile\n"
                    << "4 IsFileExists\n"
                    << "5 CreateDir\n"
                    << "6 IsDirExists\n"
                    << "7 RemoveFile\n"
                    << "8 RemoveDir\n";

        int command = 0;
        while(command != 1)
        {
            std::cout << ">> ";
            std::string commandString;
            std::cin >> commandString;
            
            try
            {
                command = std::stoi(commandString);
            }
            catch(const std::exception& e)
            {
                command = 0;
            }
                        
            switch (command)
            {
            case 1:
                std::cout << "Back to main menu" << std::endl;
                break;
            case 2:
                std::cout   << "Command: 2 CombinePath\n";
                FileSystemHelper::CombinePath();
                break;
            case 3:
                std::cout   << "Command: 3 CreateFile\n";
                FileSystemHelper::CreateFile();
                break;
            case 4:
                std::cout   << "Command: 4 IsFileExists\n";
                FileSystemHelper::IsFileExists();
                break;
            case 5:
                std::cout   << "Command: 5 CreateDir\n";
                FileSystemHelper::CreateDir();                
                break;
            case 6:
                std::cout   << "Command: 6 IsDirExists\n";
                FileSystemHelper::IsDirExists();
                break;
            case 7:
                std::cout   << "Command: 7 RemoveFile\n";
                FileSystemHelper::RemoveFile();
                break;
            case 8:
                std::cout   << "Command: 8 RemoveDir\n";
                FileSystemHelper::RemoveDir();
                break;            
            default:
                std::cout << "Command not recognized!" << std::endl;
                break;
            }
        }
    }
};
