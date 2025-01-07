#pragma once

#include <iostream>
//#include "../_IncludeLib.hpp"
#include "../AlgTestingResults/AlgTestingResult.hpp"
#include "../AlgTestingResults/AlgTestingResultRepository.hpp"

/// @brief Класс для запуска алгоритма на заданной вычислительной системе
class AlgorithmImplementationExecutor
{
private:
    // Ссылка на репозиторий вычислительных систем
    ComputingSystemRepository& computingSystemRepository;
    // УИД текущей вычислительной системы
    unsigned computingSystemId{};
    // Ссылка на репозиторий реализаций алгоритмов
    AlgorithmImplementationRepository& algorithmImplementationRepository;
    // Ссылка на репозиторий результатов тестовых запусков алгоритмов
    AlgTestingResultRepository& algTestingResultRepository;
public:    
    AlgorithmImplementationExecutor(ComputingSystemRepository& computingSystemRepository,
        AlgorithmImplementationRepository& algorithmImplementationRepository,
        AlgTestingResultRepository& algTestingResultRepository)
          : computingSystemRepository(computingSystemRepository),
            algorithmImplementationRepository(algorithmImplementationRepository),
            algTestingResultRepository(algTestingResultRepository)
    {
    }

    /// @brief Проверяет готовность
    /// @return 
    bool IsConfigured()
    {
        if(!(bool)computingSystemId)
            return false;

        return true;
    }
    
    /// @brief Устанавливает УИД текущей вычислительной системы
    /// @param id УИД вычислительной системы
    void SetComputingSystemId(unsigned id)
    {
        if (!computingSystemRepository.IsExists(id))
            throw std::runtime_error("ComputingSystem not found!");
        
        computingSystemId = id;
    }

    /// @brief Устанавливает УИД текущей вычислительной системы
    /// @param id УИД вычислительной системы
    /// @return true - успех, false - неудача
    bool TrySetComputingSystemId(unsigned id)
    {                
        if (!computingSystemRepository.IsExists(id))
            return false;

        computingSystemId = id;       
        return true;
    }

    /// @brief Запускает реализацию алгоритма с заданными параметрами
    ///        на текущей вычислительной системе
    /// @param AlgorithmImplementationId УИД реализации алгоритма
    /// @return 
    AlgTestingResult Exec(unsigned AlgorithmImplementationId,
        AlgorithmImplementationExecParams params)
    {
        return AlgTestingResult{};
    }
    
};
