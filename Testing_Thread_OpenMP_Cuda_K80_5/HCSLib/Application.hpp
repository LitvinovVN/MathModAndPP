#pragma once


/// @brief Приложение
class Application
{
    MainMenu menu; // Главное меню
    AppConfig appConfig;// Конфигурация приложения
    ComputingSystemRepository computingSystemRepository;// Репозиторий сведений о вычислительных сстемах
    AlgTestingResultRepository algTestingResultRepository;// Репозиторий сведений о тестовых запусках различных алгоритмов

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

        // 3. Запускаем главное меню
        menu.Start(appConfig,
            computingSystemRepository,
            algTestingResultRepository
        );
    }
};

