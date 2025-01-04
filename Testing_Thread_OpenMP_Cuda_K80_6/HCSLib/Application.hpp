#pragma once


/// @brief Приложение
class Application
{
    // Главное меню
    MainMenu menu;
    // Конфигурация приложения
    AppConfig appConfig;
    // Репозиторий сведений о вычислительных системах
    ComputingSystemRepository computingSystemRepository{false};
    // Репозиторий алгоритмов
    AlgorithmRepository algorithmRepository;
    // Репозиторий реализаций алгоритмов     
    AlgorithmImplementationRepository algorithmImplementationRepository;
    // Репозиторий сведений о тестовых запусках различных алгоритмов
    AlgTestingResultRepository algTestingResultRepository{false};

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

        // 3. Инициализируем репозиторий алгоритмов
        algorithmRepository = AlgorithmRepository{};

        // 4. Инициализируем репозиторий реализаций алгоритмов
        algorithmImplementationRepository = AlgorithmImplementationRepository{};

        // 5. Считываем сведения о результатах тестовых запусков алгоритмов
        algTestingResultRepository = AlgTestingResultRepository {appConfig.GetDirAlgTestingResultRepository()};
        std::cout << "Computing system repository initialization: OK" << std::endl;

        // 5. Запускаем главное меню
        menu.Start(appConfig,
            computingSystemRepository,
            algorithmRepository,
            algorithmImplementationRepository,
            algTestingResultRepository
        );//*/
    }
};

