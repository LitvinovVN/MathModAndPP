#pragma once

#include "../GlobalTestFunctions.hpp"
#include "../Algorithms/AlgorithmImplementationExecutor.hpp"
#include "../Algorithms/AlgorithmImplementationExecutorHelper.hpp"
#include "../Matrices/MatricesHelper.hpp"

/// @brief Функции меню
struct MenuFunctions
{    
    /// @brief Выводит параметры GPU
    /*static void PrintGpuParameters()
    {
        CudaHelper::PrintCudaDeviceProperties();
    }*/

    /// @brief Выводит в консоль список поддерживаемых библиотек
    static void PrintLibSupport()
    {
        // Определяем перечень поддерживаемых библиотек
        LibSupport support;
        support.Print();// Выводим список поддерживаемых библиотек
    }

    /// @brief Работа с классом CudaHelper
    static void CudaHelper()
    {
        std::cout   << "----- CudaHelper -----\n"
                    << "1 Back to main menu\n"
                    << "2 IsCudaSupported()\n"
                    << "3 GetCudaDeviceNumber()\n"
                    << "4 GetCudaDeviceProperties(int deviceId = 0)\n"
                    << "5 WriteGpuSpecsToTxtFile_ConsoleUI()\n"
                    << "6 -\n"
                    << "7 -\n"
                    << "8 -\n";

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
                std::cout   << "Command: 2 IsCudaSupported()\n";                
                std::cout << std::boolalpha
                          << CudaHelper::IsCudaSupported()
                          << std::endl;
                break;
            case 3:
                std::cout   << "Command: 3 GetCudaDeviceNumber()\n";
                std::cout << std::boolalpha
                          << CudaHelper::GetCudaDeviceNumber()
                          << std::endl;
                break;
            case 4:
                std::cout << "Command: 4 GetCudaDeviceProperties(int deviceId = 0)\n";
                CudaHelper::PrintCudaDeviceProperties_ConsoleUI();
                break;
            case 5:
                std::cout   << "Command: 5 WriteGpuSpecsToTxtFile_ConsoleUI()\n";
                CudaHelper::WriteGpuSpecsToTxtFile_ConsoleUI();               
                break;
            case 6:
                std::cout   << "Command: 6 -\n";
                //FileSystemHelper::IsDirExists();
                break;
            case 7:
                std::cout   << "Command: 7 -\n";
                //FileSystemHelper::RemoveFile();
                break;
            case 8:
                std::cout   << "Command: 8 -\n";
                //FileSystemHelper::RemoveDir();
                break;            
            default:
                std::cout << "Command not recognized!" << std::endl;
                break;
            }
        }

    }

    /// @brief Тестирование функций класса ArrayHelper
    static void ArrayHelper()
    {        
        std::cout   << "----- ArrayHelper -----\n"
                    << "1 Back to main menu\n"
                    << "2 ArrayHelper::SumOpenMP\n"
                    << "3 ArrayHelper::SumCudaMultiGpu\n"
                    << "4 ArrayHelper::CopyRamToGpu\n"
                    << "5 ArrayHelper::CopyGpuToRam\n"
                    << "6 ArrayHelper::ScalarProductRamSeq\n"
                    << "7 ArrayHelper::ScalarProductRamParThread\n"
                    << "8 ArrayHelper::ScalarProductGpuParCuda\n"
                    << "9 ArrayHelper::ScalarProductMultiGpuParCuda\n";

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
                std::cout   << "Command: 2 ArrayHelper::SumOpenMP()\n";                
                ArrayHelper_ConsoleUI::SumOpenMP_ConsoleUI();
                break;
            case 3:
                std::cout   << "Command: 3 ArrayHelper::SumCudaMultiGpu()\n";
                ArrayHelper_ConsoleUI::SumCudaMultiGpu_ConsoleUI();
                break;
            case 4:
                std::cout << "Command: 4 ArrayHelper::CopyRamToGpu()\n";
                ArrayHelper_ConsoleUI::CopyRamToGpu_ConsoleUI();
                break;
            case 5:
                std::cout   << "Command: 5 ArrayHelper::CopyGpuToRam()\n";
                ArrayHelper_ConsoleUI::CopyGpuToRam_ConsoleUI();               
                break;
            case 6:
                std::cout   << "Command: 6 ArrayHelper::ScalarProductRamSeq\n";
                ArrayHelper_ConsoleUI::ScalarProductRamSeq_ConsoleUI();
                break;
            case 7:
                std::cout   << "Command: 7 ArrayHelper::ScalarProductRamParThread\n";
                ArrayHelper_ConsoleUI::ScalarProductRamParThread_ConsoleUI();
                break;
            case 8:
                std::cout   << "Command: 8 ArrayHelper::ScalarProductGpuParCuda\n";
                ArrayHelper_ConsoleUI::ScalarProductGpuParCuda_ConsoleUI();
                break;
            case 9:
                std::cout   << "Command: 8 ArrayHelper::ScalarProductMultiGpuParCuda\n";
                ArrayHelper_ConsoleUI::ScalarProductMultiGpuParCuda_ConsoleUI();
                break;
            default:
                std::cout << "Command not recognized!" << std::endl;
                break;
            }
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

    /// @brief Запускает тесты работоспособности классов матриц
    static void Testing_Matrices()
    {
        std::cout << "Testing_Matrices()" << std::endl;
        std::cout << "-- MatrixRamZeroTesting()" << std::endl;
        MatricesHelper::MatrixRamZeroTesting();
        std::cout << "-- MatrixRamETesting()" << std::endl;
        MatricesHelper::MatrixRamETesting();
    }

    /// @brief Запускает функцию тестирования суммирования элементов массивов
    static void Testing_TestSum()
    {
        // Запускаем функцию тестирования суммирования элементов массивов
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
                    << "8 Is computing system exists\n"
                    << "9 Clear computing system repository\n"
                    << "10 Init computing system repository\n";

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
            case 9:
                std::cout   << "Command: 9 Clear computing system repository \n";
                repo.Clear();
                break;
            case 10:
                std::cout   << "Command: 10 Init computing system repository \n";
                repo.Init();
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
                    << "3 Get last Id\n"
                    << "4 Find alg testing result\n"
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
                repo.PrintConfig();
                break;
            case 3:
                std::cout   << "Command: 3 Get last Id\n";
                try
                {
                    std::cout << repo.GetLastId() << std::endl;
                }
                catch(const std::exception& e)
                {
                    std::cerr << e.what() << '\n';
                }
                                
                break;
            case 4:
                std::cout   << "Command: 4 Find alg testing result\n";
                repo.Find();
                break;
            case 5:
                std::cout   << "Command: 5 Add test alg result data\n";
                repo.Add();
                break;
            case 6:
                std::cout   << "Command: 6 Change alg testing result\n";
                //repo.Change();
                break;
            case 7:
                std::cout   << "Command: 7 Remove alg testing result\n";
                //repo.Remove();
                break;
            case 8:
                std::cout   << "Command: 8 Is alg testing result exists\n";
                repo.IsExists();
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

    static void AlgorithmRepository(AlgorithmRepository& repo)
    {
        std::cout   << "----- AlgorithmRepository configuration -----\n"
                    << "1 Back to main menu\n"
                    << "2 Print algorithms\n"
                    << "3 Get algorithm\n";
                    

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
                repo.Print(PrintParams{});
                break;
            case 3:
                std::cout   << "Command: 3 Get algorithm\n";
                try
                {
                    repo.Get();
                }
                catch(const std::exception& e)
                {
                    std::cerr << e.what() << '\n';
                }
                break;
            
            default:
                std::cout << "Command not recognized!" << std::endl;
                break;
            }
        }
    }

    /// @brief Работа с репозиторием реализаций алгоритмов
    /// @param repo Объект типа AlgorithmImplementationRepository
    static void Menu_AlgorithmImplementationRepository(AlgorithmImplementationRepository& repo)
    {
        std::cout   << "----- AlgorithmImplementationRepository configuration -----\n"
                    << "1 Back to main menu\n"
                    << "2 Print algorithm implementations\n"
                    << "3 Get algorithm implementation\n";
                    

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
                std::cout   << "2 Print algorithm implementations\n";
                repo.Print(PrintParams{"[\n"});
                break;
            case 3:
                std::cout   << "3 Get algorithm implementation\n";
                try
                {
                    //repo.Get();
                }
                catch(const std::exception& e)
                {
                    std::cerr << e.what() << '\n';
                }
                break;
            
            default:
                std::cout << "Command not recognized!" << std::endl;
                break;
            }
        }
    }

    /// @brief Запуск различных реализаций алгоритмов
    /// @param repo Объект типа AlgorithmImplementationRepository
    static void Menu_AlgorithmImplementationExecutor(AlgorithmImplementationExecutor& algorithmImplementationExecutor)
    {
        std::cout   << "----- AlgorithmImplementationExecutor -----\n"
                    << "1 Back to main menu\n"
                    << "2 Exec T* Sum\n"
                    << "3 Get T* Sum results\n";
                    

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
                std::cout   << "2 Exec T* Sum\n";                
                AlgorithmImplementationExecutorHelper::Exec(algorithmImplementationExecutor);                
                break;
            case 3:
                std::cout   << "3 Get T* Sum results\n";
                try
                {
                    //repo.Get();
                }
                catch(const std::exception& e)
                {
                    std::cerr << e.what() << '\n';
                }
                break;
            
            default:
                std::cout << "Command not recognized!" << std::endl;
                break;
            }
        }
    }

};

