#pragma once

#include <iostream>
#include <vector>
#include "../CommonHelpers/FuncResult.hpp"

/// @brief Статистические параметры результатов численного эксперимента
struct CalculationStatistics
{
    // Количество запусков численного эксперимента
    unsigned numIter = 0;
    // Минимальное значение
    double minValue = 0;
    // Среднее арифметическое
    double avg = 0;
    // Медиана
    double median = 0;
    // 95 процентиль
    double percentile_95 = 0;
    // Максимальное значение
    double maxValue = 0;
    // Среднеквадратическое отклонение
    double stdDev = 0;

    CalculationStatistics()
    {}

    template<typename T>
    CalculationStatistics(std::vector<FuncResult<T>> results)
    {
        auto resultsSize = results.size();
        if (resultsSize == 0)
            throw std::logic_error("results size is 0");

        // Проверяем корректность результатов        
        for(unsigned i = 1; i < resultsSize; i++)
        {
            if(results[i].status == false)
                throw std::logic_error("results[i].Status = 0");
            
            if( fabs((results[i].result - results[0].result) / (double)results[0].result) > 0.0001 )
                throw std::logic_error("fabs((results[i].result - results[0].result) / results[0].Result) > 0.0001");
        }

        //print(std::string("---Before sort---"), results);
        // Сортируем results
        std::sort(results.begin(), results.end(), FuncResult<T>::compare);
        //print(std::string("---After sort---"), results);        
        //std::cout << "----------" << std::endl;

        minValue = results[0].time;
        maxValue = results[resultsSize - 1].time;

        if(resultsSize % 2 == 0)
        {
            median = (results[resultsSize / 2 - 1].time + results[resultsSize / 2].time)/2;
        }
        else
        {
            median = results[resultsSize / 2].time;
        }

        // Вычисляем среднее арифметическое
        double sum = 0;
        for(auto& item : results)
            sum += item.time;
        
        avg = sum / resultsSize;

        // Вычисляем стандартное отклонение
        double sumSq = 0;
        for(auto& item : results)
            sumSq += pow(item.time - avg, 2);
        
        stdDev = sqrt(sumSq / resultsSize);

        // Вычисляем 95 перцентиль
        double rang95 = 0.95*(resultsSize-1) + 1;
        unsigned rang95okrVniz = (unsigned)floor(rang95);
        percentile_95 = results[rang95okrVniz-1].time + (rang95-rang95okrVniz)*(results[rang95okrVniz].time - results[rang95okrVniz-1].time);// Доделать

        //Print();
    }

    void Print()
    {
        std::cout   << "numIter: " << numIter  << "; "
                    << "minValue: " << minValue << "; "
                    << "median: " << median   << "; "
                    << "avg: " << avg      << "; "
                    << "percentile_95: " << percentile_95   << "; "
                    << "maxValue: " << maxValue << "; "                                                            
                    << "stdDev: " << stdDev   << "; ";
    }

    friend std::ofstream& operator<<(std::ofstream& fout, const CalculationStatistics& data)
    {
        fout << data.numIter << " "
             << data.minValue << " "
             << data.median << " "
             << data.avg << " "
             << data.percentile_95 << " "
             << data.maxValue << " "
             << data.stdDev << " ";

        return fout;
    }
};

