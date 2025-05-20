#pragma once


/// @brief Перечисление команд меню
enum class MenuCommand
{
    None,                     // Не выбрано    
    Help,                     // Вывод в консоль справки
    Exit,                     // Выход из меню
    PrintLibSupport,          // Вывод в консоль списка поддерживаемых библиотек
    CudaHelper,               // Работа с классом CudaHelper
    //PrintGpuParameters,     // Вывод в консоль параметров GPU
    //WriteGpuSpecsToTxtFile, // Записывает параметры видеокарт в текстовый файл gpu-specs.txt
    ArrayHelper,              // Работа с классом ArrayHelper
    ArrayPerfTestHelper,      // Работа с классом ArrayPerfTestHelper
    Testing_TestVectorGpu,    // Тестирование класса VectorGpu
    VectorsHelper_ConsoleUI,  // Работа с модулем Vectors
    Testing_Matrices,         // Тестирование классов матриц
    Math,                     // Работа с модулем Math
    Testing_TestSum,          // Тестирование функций суммирования
    Application_Config,       // Конфигурация приложения
    ComputingSystemRepository_Config, // Конфигурирование хранилища сведений о вычислительных системах
    AlgTestingResultRepository_Config, // Работа с хранилищем результатов тестовых запусков
    Testing_FileSystemHelper, // Тестирование вспомогательного класса для работы с файловой системой
    AlgorithmRepository,       // Тестирование репозитория алгоритмов
    AlgorithmImplementationRepository, // Работа с репозиторием реализаций алгоритмов
    AlgorithmImplementationExecutor,// Запуск различных реализаций алгоритмов
    GeometryHelper_ConsoleUI,  // Работа с модулем Geometry
    DifferentialEquations_ConsoleUI // Работа с модулем DifferentialEquations
};

