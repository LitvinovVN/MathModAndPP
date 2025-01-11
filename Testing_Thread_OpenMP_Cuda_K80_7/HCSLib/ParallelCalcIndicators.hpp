#pragma once


/// @brief Показатели параллельного вычислительного процесса (ускорение, эффективность)
struct ParallelCalcIndicators
{
    unsigned Nthreads;

    double Smin;
    double Smax;
    double Savg;
    double Smedian;
    double Sperc95;

    double Emin;
    double Emax;
    double Eavg;
    double Emedian;
    double Eperc95;

    ParallelCalcIndicators(CalculationStatistics& stat_seq,
                           CalculationStatistics& stat_par,
                           unsigned Nthreads) : Nthreads(Nthreads)
    {
        Smin = stat_seq.minValue / stat_par.minValue;
        Smax = stat_seq.maxValue / stat_par.maxValue;
        Savg = stat_seq.avg / stat_par.avg;
        Smedian = stat_seq.median / stat_par.median;
        Sperc95 = stat_seq.percentile_95 / stat_par.percentile_95;

        Emin = Smin / Nthreads;
        Emax = Smax / Nthreads;
        Eavg = Savg / Nthreads;
        Emedian = Smedian / Nthreads;
        Eperc95 = Sperc95 / Nthreads;
    }

    void Print()
    {
        std::cout << "N threads: " << Nthreads << std::endl;

        std::cout << "Smin: " << Smin << std::endl;
        std::cout << "Smax: " << Smax << std::endl;
        std::cout << "Savg: " << Savg << std::endl;
        std::cout << "Smedian: " << Smedian << std::endl;
        std::cout << "Sperc95: " << Sperc95 << std::endl;

        std::cout << "Emin: " << Emin << std::endl;
        std::cout << "Emax: " << Emax << std::endl;
        std::cout << "Eavg: " << Eavg << std::endl;
        std::cout << "Emedian: " << Emedian << std::endl;
        std::cout << "Eperc95: " << Eperc95 << std::endl;
    }
};

