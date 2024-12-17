#pragma once


/// @brief Приложение
class Application
{
    MainMenu menu; // Главное меню
    AppConfig appConfig;// Конфигурация приложения
    ComputingSystemRepository computingSystemRepository{false};// Репозиторий сведений о вычислительных сстемах
    AlgTestingResultRepository algTestingResultRepository{false};// Репозиторий сведений о тестовых запусках различных алгоритмов

public:

    AppConfig& GetAppConfig()
    {
        return appConfig;
    }

    void Start()
    {
        // 1. Считываем конфигурацию из файла
        std::string appConfigFileName {"config.txt"};
        appConfig = AppConfig(appConfigFileName);
        if(!appConfig.IsInitialized())
        {
            std::cerr << appConfig.GetMessage() << std::endl;
            exit(-1);
        }        
        std::cout << "Application initialization: OK" << std::endl;

        // 2. Считываем сведения о вычислительной системе
        computingSystemRepository = ComputingSystemRepository {appConfig.GetDirComputingSystemRepository()};
        std::cout << "Computing system repository initialization: OK" << std::endl;

        // 3. Считываем сведения о результатах тестовых запусков алгоритмов
        algTestingResultRepository = AlgTestingResultRepository {appConfig.GetDirAlgTestingResultRepository()};
        std::cout << "Computing system repository initialization: OK" << std::endl;

        // 4. Запускаем главное меню
        menu.Start(appConfig,
            computingSystemRepository,
            algTestingResultRepository
        );//*/
    }
};

