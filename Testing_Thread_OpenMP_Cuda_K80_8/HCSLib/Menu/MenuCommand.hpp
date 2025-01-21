#pragma once


/// @brief Перечисление команд меню
enum class MenuCommand
{
    None,                     // Не выбрано    
    Help,                     // Вывод в консоль справки
    Exit,                     // Выход из меню
    PrintLibSupport,          // Вывод в консоль списка поддерживаемых библиотек
    CudaHelper,               // Работа с классом CudaHelper
    //PrintGpuParameters,       // Вывод в консоль параметров GPU
    //WriteGpuSpecsToTxtFile,   // Записывает параметры видеокарт в текстовый файл gpu-specs.txt
    ArrayHelper,              // Работа с классом ArrayHelper
    Testing_TestVectorGpu,    // Тестирование класса VectorGpu
    Testing_Matrices,         // Тестирование классов матриц
    Testing_TestSum,          // Тестирование функций суммирования
    Application_Config,       // Конфигурация приложения
    ComputingSystemRepository_Config, // Конфигурирование хранилища сведений о вычислительных системах
    AlgTestingResultRepository_Config, // Работа с хранилищем результатов тестовых запусков
    Testing_FileSystemHelper, // Тестирование вспомогательного класса для работы с файловой системой
    AlgorithmRepository,       // Тестирование репозитория алгоритмов
    AlgorithmImplementationRepository, // Работа с репозиторием реализаций алгоритмов
    AlgorithmImplementationExecutor// Запуск различных реализаций алгоритмов
};

