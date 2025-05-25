#pragma once

/// @brief Результаты вычислительного эксперимента
struct PerfTestResults
{
    CalculationStatistics calculationStatistics;
    ParallelCalcIndicators parallelCalcIndicators;

    void Print(PrintParams pp = PrintParams{})
    {
        std::cout << pp.startMes;

        std::cout << pp.endMes;

        if(pp.isEndl)
            std::cout << std::endl;
    }
};