#pragma once


/// @brief Перечисление команд меню
enum class MenuCommand
{
    None,                     // Не выбрано
    Exit,                     // Выход из меню
    Help,                     // Вывод в консоль справки
    PrintLibSupport,          // Вывод в консоль списка поддерживаемых библиотек
    PrintGpuParameters,       // Вывод в консоль параметров GPU
    WriteGpuSpecsToTxtFile,   // Записывает параметры видеокарт в текстовый файл gpu-specs.txt
    Testing_TestArrayHelper,  // Тестирование класса TestArrayHelper
    Testing_TestVectorGpu,    // Тестирование класса VectorGpu
    Testing_TestSum,          // Тестирование функций суммирования
    Application_Config,       // Конфигурация приложения
    ComputingSystemRepository_Config, // Конфигурирование хранилища сведений о вычислительных системах
    AlgTestingResultRepository_Config, // Работа с хранилищем результатов тестовых запусков
    Testing_FileSystemHelper  // Тестирование вспомогательного класса для работы с файловой системой
};

