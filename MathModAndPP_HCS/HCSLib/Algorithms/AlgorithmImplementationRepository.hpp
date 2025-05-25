#pragma once

#include <iostream>
#include <map>

#include "AlgorithmImplementation.hpp"
#include "../CommonHelpers/PrintParams.hpp"

// Репозиторий реализаций алгоритмов.
// Сопоставляет УИД алгоритма с функцией, реализующей алгоритм
class AlgorithmImplementationRepository
{
    std::map<unsigned, AlgorithmImplementation> data;

    /// @brief Инициализация репозитория реализаций алгоритмов
    void Init();
public:

    AlgorithmImplementationRepository()
    {
        Init();
    }

    void Print(PrintParams pp)
    {
        std::cout << pp.startMes;
        for(auto& el : data)
        {
            std::cout << el.first << pp.splitterKeyValue;
            el.second.Print(PrintParams{}.SetIsEndl(false));
            std::cout << pp.splitter;
            if(pp.isEndl)
                std::cout << std::endl;
        }
        std::cout << pp.endMes;
        if(pp.isEndl)
            std::cout << std::endl;
    }

    /// @brief Проверяет наличие алгоритма с указанным УИД
    /// @return
    bool IsExists(unsigned id)
    {
        return data.count(id) > 0;
    }

    /// @brief Возвращает алгоритм по УИД
    /// @param id
    /// @return
    AlgorithmImplementation Get(unsigned id)
    {
        return data[id];
    }

    /// @brief Запрашивает у пользователя id алгоритма и выводит в консоль сведения о нём
    void Get()
    {
        unsigned id = ConsoleHelper::GetUnsignedIntFromUser("Enter algorithm implementation id: ");
        AlgorithmImplementation algImpl = Get(id);
        unsigned algImpl_id = algImpl.GetId();
        if(algImpl_id == id && algImpl_id != 0)
            algImpl.Print(PrintParams{});
        else
            std::cout << "Algorithm implementation not found!" << std::endl;
    }

    /// @brief Добавляет реализацию алгоритма в репозиторий
    /// @param algImpl
    /// @return Результат (true - добавлен, false - не добавлен)
    bool Add(AlgorithmImplementation algImpl)
    {
        unsigned algImpl_id = algImpl.GetId();
        if (algImpl_id == 0 || IsExists(algImpl_id))
            return false;

        data[algImpl_id] = algImpl;
        return true;
    }

};

///////////////////////////////////////////////////////

void AlgorithmImplementationRepository::Init()
{
    Function f1{ArrayHelper::Sum<float>};
    AlgorithmImplementation algImpl_01{1, 1, "ArrayHelper::Sum<float>", f1};
    Add(algImpl_01);
    ///////////////////////////////
    Function f2{ArrayHelper::Sum<double>};
    AlgorithmImplementation algImpl_02{2, 2, "ArrayHelper::Sum<double>", f2};
    Add(algImpl_02);

}