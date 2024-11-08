#pragma once

#include <iostream>
#include <vector>

// Статистические параметры результатов эксперимента
struct CalculationStatistics
{
    // Количество запусков численного эксперимента
    unsigned numIter;
    // Минимальное значение
    double minValue;
    // Максимальное значение
    double maxValue;
    // Среднее арифметическое
    double avg;
    // Медиана
    double median;
    // 95 процентиль
    double percentile_95;
    // Среднеквадратическое отклонение
    double stdDev;

    CalculationStatistics(std::vector<FuncResultScalar<double>> results)
    {
        auto resultsSize = results.size();
        if (resultsSize == 0)
            throw std::logic_error("results size is 0");

        // Проверяем корректность результатов        
        for(unsigned i = 1; i < resultsSize; i++)
        {
            if(results[i].Status == false)
                throw std::logic_error("results[i].Status = 0");
            
            if( fabs((results[i].Result - results[0].Result) / results[0].Result) > 0.0001 )
                throw std::logic_error("fabs((results[i].Result - results[0].Result) / results[0].Result) > 0.0001");
        }

        //print(std::string("---Before sort---"), results);
        // Сортируем results
        std::sort(results.begin(), results.end(), compare);
        //print(std::string("---After sort---"), results);        
        //std::cout << "----------" << std::endl;

        minValue = results[0].Time_mks;
        maxValue = results[resultsSize - 1].Time_mks;

        if(resultsSize % 2 == 0)
        {
            median = (results[resultsSize / 2 - 1].Time_mks + results[resultsSize / 2].Time_mks)/2;
        }
        else
        {
            median = results[resultsSize / 2].Time_mks;
        }

        // Вычисляем среднее арифметическое
        double sum = 0;
        for(auto& item : results)
            sum += item.Time_mks;
        
        avg = sum / resultsSize;

        // Вычисляем стандартное отклонение
        double sumSq = 0;
        for(auto& item : results)
            sumSq += pow(item.Time_mks - avg, 2);
        
        stdDev = sqrt(sumSq / resultsSize);

        // Вычисляем 95 перцентиль
        double rang95 = 0.95*(resultsSize-1) + 1;
        unsigned rang95okrVniz = (unsigned)floor(rang95);
        percentile_95 = results[rang95okrVniz-1].Time_mks + (rang95-rang95okrVniz)*(results[rang95okrVniz].Time_mks - results[rang95okrVniz-1].Time_mks);// Доделать

        //Print();
    }

    void Print()
    {
        std::cout   << "minValue: "      << minValue << "; "
                    << "median: "        << median   << "; "
                    << "avg: "           << avg      << "; "
                    << "percentile_95: " << percentile_95   << "; "
                    << "maxValue: "      << maxValue << "; "                                                            
                    << "stdDev: "        << stdDev   << "; "
                    << std::endl;
    }
};
