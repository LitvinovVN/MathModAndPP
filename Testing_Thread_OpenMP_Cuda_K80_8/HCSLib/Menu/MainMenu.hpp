#pragma once

/// @brief Главное меню приложения
class MainMenu
{
    // Список команд меню
    std::vector<MenuCommandItem> menuCommands;

    //MenuCommand command = MenuCommand::None;// Выбранная команда меню
    MenuCommandItem command;// Выбранная команда меню
    
    /// @brief Распознаёт команду
    /// @param commandString 
    /// @return 
    bool RecognizeCommand(std::string commandString)
    {        
        command.Reset();
        for(auto& menuItem : menuCommands)
        {
            if(menuItem.CheckKey(commandString))
            {
                command = menuItem;
                return true;
            }
        }

        return false;
    }

    /// @brief Исполняет команду
    void RunCommand()
    {
        if(command.func == nullptr)        
            return;
        
        std::cout << "----- Starting: " << command.desc << "-----------" << std::endl;
        command.func();
        std::cout << "-------------------------------------" << std::endl;
    }

    /// @brief Выводит в консоль справочную информацию
    void PrintHelp()
    {
        std::cout << "----- Command list -----" << std::endl;
        for(auto& menuItem : menuCommands)
        {            
            for(auto& key : menuItem.keys)
            {
                std::cout << key << " ";
            }
            std::cout << "\t" << menuItem.desc << std::endl;
        }
    }

public:

    MainMenu()
    {
        // Инициализация меню
        MenuCommandItem item1;
        item1.comm = MenuCommand::Help;
        item1.keys = {std::to_string((int)MenuCommand::Help),"?","h","help"};
        item1.func = nullptr;
        item1.desc = "Print help";
        menuCommands.push_back(item1);

        MenuCommandItem item2;
        item2.comm = MenuCommand::Exit;
        item2.keys = {std::to_string((int)MenuCommand::Exit),"q","exit"};
        item2.func = nullptr;
        item2.desc = "Exit from menu";
        menuCommands.push_back(item2);
        
        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::PrintLibSupport,
                {std::to_string((int)MenuCommand::PrintLibSupport),"libs"},
                MenuFunctions::PrintLibSupport,
                "Print supported libs (OpenMP, Cuda etc.)"
            }
        );

        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::CudaHelper,
                {std::to_string((int)MenuCommand::CudaHelper),"CudaHelper"},
                MenuFunctions::CudaHelper,
                "Class CudaHelper"
            }
        );

        /*menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::PrintGpuParameters,
                {std::to_string((int)MenuCommand::PrintGpuParameters),"gpu"},
                MenuFunctions::PrintGpuParameters,
                "Print default (0) Cuda-device properties"
            }
        );*/

        /*menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::WriteGpuSpecsToTxtFile,
                {std::to_string((int)MenuCommand::WriteGpuSpecsToTxtFile),"gpu"},
                MenuFunctions::WriteGpuSpecsToTxtFile,
                "Write GPU specification to txt file gpu-specs.txt"
            }
        );*/

        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::ArrayHelper,
                {std::to_string((int)MenuCommand::ArrayHelper),"ArrayHelper"},
                MenuFunctions::ArrayHelper,
                "Testing TestArrayHelper class"
            }
        );

        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::Testing_TestVectorGpu,
                {std::to_string((int)MenuCommand::Testing_TestVectorGpu),"test-vec-gpu"},
                MenuFunctions::Testing_TestVectorGpu,
                "Testing VectorGpu class"
            }
        );

        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::Testing_Matrices,
                {std::to_string((int)MenuCommand::Testing_Matrices),"test-mat"},
                MenuFunctions::Testing_Matrices,
                "Testing Matrices classes"
            }
        );

        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::Testing_TestSum,
                {"9","test-sum"},
                MenuFunctions::Testing_TestSum,
                "Testing sum functions"
            }
        );

        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::Application_Config,
                {"10","app-conf"},
                nullptr,
                "Application configuration"
            }
        );

        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::ComputingSystemRepository_Config,
                {"11","cs-repo-conf"},
                nullptr,
                "Computing system repository configuration"
            }
        );

        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::AlgTestingResultRepository_Config,
                {"12","algtr-repo-conf"},
                nullptr,
                "AlgTestingResultRepository configuration"
            }
        );
        
        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::Testing_FileSystemHelper,
                {"13","fs-hlp"},
                MenuFunctions::Testing_FileSystemHelper,
                "Testing FileSystemHelper"
            }
        );

        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::AlgorithmRepository,
                {"14","alg-repo"},
                nullptr,
                "Testing AlgorithmRepository"
            }
        );

        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::AlgorithmImplementationRepository,
                {"15","alg-impl-repo"},
                nullptr,
                "AlgorithmImplementationRepository"
            }
        );

        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::AlgorithmImplementationExecutor,
                {"16","alg-impl-exec"},
                nullptr,
                "AlgorithmImplementationExecutor"
            }
        );
    }

    /// @brief Запуск главного меню
    void Start(AppConfig& appConfig,
        ComputingSystemRepository& compSysRepo,
        AlgorithmRepository& algorithmRepository,
        AlgorithmImplementationRepository& algorithmImplementationRepo,
        AlgTestingResultRepository& algTestingResultRepo,
        AlgorithmImplementationExecutor& algorithmImplementationExecutor)
    {
        std::cout << "--- Main Menu ('1', '?', 'h' or 'help' for print help)---" << std::endl;
        std::string commandString;// Введённая пользователем команда
        
        while(command.comm != MenuCommand::Exit)
        {
            std::cout << "> ";
            std::cin >> commandString;
            if ( !RecognizeCommand(commandString))// Распознаём команду
            {
                std::cout << "Error! Command not recognized! Please enter command again. '?' or 'help' for print help." << std::endl;
                continue;
            }

            switch (command.comm)
            {
            case MenuCommand::Help:
                PrintHelp();
                break;
            case MenuCommand::Application_Config:
                MenuFunctions::Application_Config(appConfig);
                break;
            case MenuCommand::ComputingSystemRepository_Config:
                MenuFunctions::ComputingSystemRepository_Config(compSysRepo);
                break;
            case MenuCommand::AlgTestingResultRepository_Config:
                MenuFunctions::AlgTestingResultRepository_Config(algTestingResultRepo);
            case MenuCommand::AlgorithmRepository:
                MenuFunctions::AlgorithmRepository(algorithmRepository);
            case MenuCommand::AlgorithmImplementationRepository:
                MenuFunctions::Menu_AlgorithmImplementationRepository(algorithmImplementationRepo);
            case MenuCommand::AlgorithmImplementationExecutor:
                MenuFunctions::Menu_AlgorithmImplementationExecutor(algorithmImplementationExecutor);
            default:
                RunCommand();
                break;
            }            
        }
        std::cout << "--- Good bye! ---" << std::endl;
    }

};

