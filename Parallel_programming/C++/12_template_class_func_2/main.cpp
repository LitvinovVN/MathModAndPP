#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>
#include <functional>
#include <algorithm>
#include <cmath>

using namespace std::chrono;

template<typename T>
class ArrayRamHelper
{
public:
    static void SumThread(T* data, size_t indStart, size_t indEnd, T& sum, std::mutex& m)
    {
        T local_sum = 0;

        for (size_t i = indStart; i <= indEnd; i++)
        {
            local_sum += data[i];
        }
        
        {
            std::lock_guard<std::mutex> lock(m);
            //m.lock();
            /*std::cout << "thread " << std::this_thread::get_id()
                << "| local_sum = " << local_sum
                << std::endl;*/
            sum += local_sum;
            //m.unlock();
        }
    }
};

template<typename T>
class VectorRam
{
public:
    T* data;
    size_t size;

    VectorRam(size_t size) : size(size)
    {
        data = new T[size];
    }

    ~VectorRam()
    {
        delete[] data;
    }

    void InitByVal(T val)
    {
        for (size_t i = 0; i < size; i++)
        {
            data[i] = val;
        }        
    }

    void PrintToConsole()
    {
        for (size_t i = 0; i < size; i++)
        {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;     
    }

    T Sum(size_t indStart, size_t indEnd)
    {
        T result = 0;
        for (size_t i = indStart; i <= indEnd; i++)
        {
            result += data[i];
        }
        return result;
    }

    T Sum()
    {
        T result = Sum(0, size-1);
        return result;
    }

    T Sum(size_t indStart, size_t indEnd, unsigned threadsNum)
    {
        std::mutex m;
        T sum = 0;
        size_t blockSize = indEnd - indStart + 1;
        std::vector<std::thread> threads;
        size_t thBlockSize = blockSize / threadsNum;
        for (size_t i = 0; i < threadsNum; i++)
        {            
            size_t thIndStart = i * thBlockSize;
            size_t thIndEnd = thIndStart + thBlockSize - 1;
            if(i == threadsNum - 1)
                thIndEnd = indEnd;

            threads.push_back(std::thread(ArrayRamHelper<T>::SumThread, data, thIndStart, thIndEnd, std::ref(sum), std::ref(m)));
        }
        
        for(auto& th : threads)
        {
            th.join();
        }

        return sum;
    }

    T Sum(unsigned threadsNum)
    {
        return Sum(0, size - 1, threadsNum);
    }

};


template<typename T>
struct FuncResult
{
    bool _status;
    T _result;
    long long _time;

    FuncResult(bool status, T result, double time) : 
        _status(status), _result(result), _time(time)
    { }

    void Print()
    {
        std::cout << "[val: " << _result
                  << "; time: " << _time << "]" << std::endl;
    }
};

class VectorRamHelper
{
public:
    template<typename T>
    static
    FuncResult<T> Sum(VectorRam<T>& v, size_t indStart, size_t indEnd)
    {
        bool calcStatus = true;
        auto start = high_resolution_clock::now();
        T result = v.Sum(indStart, indEnd);
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);        
        auto t = duration.count();

        return FuncResult<T>(calcStatus, result, t);
    }

    template<typename T>    
    static
    FuncResult<T> Sum(VectorRam<T>& v)
    {
        return Sum(v, 0, v.size - 1);
    }

    template<typename T>
    static
    FuncResult<T> Sum(VectorRam<T>& v, size_t indStart, size_t indEnd, unsigned threadsNum)
    {
        bool calcStatus = true;
        auto start = high_resolution_clock::now();
        T result = v.Sum(indStart, indEnd, threadsNum);
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);        
        auto t = duration.count();

        return FuncResult<T>(calcStatus, result, t);
    }

    template<typename T>
    static
    FuncResult<T> Sum(VectorRam<T>& v, unsigned threadsNum)
    {
        return Sum(v, 0, v.size, threadsNum);
    }
};

class TestHelper
{
public:
    template<typename T>
    static auto LaunchSum(VectorRam<T>& v)
    {
        std::cout << "-------LaunchSum(VectorRam<T>& v) Start ------" << std::endl;
        auto iterNum = 10;
        std::vector<FuncResult<T>> results;
        for(unsigned i{0}; i < iterNum; i++)
        {
            FuncResult<T> res = VectorRamHelper::Sum(v);
            results.push_back(res);
        }
        
        std::cout << "-------LaunchSum(VectorRam<T>& v) End --------" << std::endl;
        return results;
    }

    template<typename T>
    static auto LaunchSum(VectorRam<T>& v, unsigned Nthreads)
    {
        std::cout << "-------LaunchSum(VectorRam<T>& v, unsigned Nthreads) Start ------" << std::endl;
        auto iterNum = 10;
        std::vector<FuncResult<T>> results;
        for(unsigned i{0}; i < iterNum; i++)
        {
            FuncResult<T> res = VectorRamHelper::Sum(v, Nthreads);
            results.push_back(res);
        }
        
        std::cout << "-------LaunchSum(VectorRam<T>& v, unsigned Nthreads) End --------" << std::endl;
        return results;
    }
};

template<typename T>
bool compare(const FuncResult<T>& left, const FuncResult<T>& right) 
{ 
    return left._time < right._time; 
}


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

    CalculationStatistics(std::vector<FuncResult<double>> results)
    {
        auto resultsSize = results.size();
        if (resultsSize == 0)
            throw std::logic_error("results size is 0");

        // Проверяем корректность результатов        
        for(unsigned i = 1; i < resultsSize; i++)
        {
            if(results[i]._status == false)
                throw std::logic_error("results[i].Status = 0");
            
            if( fabs((results[i]._result - results[0]._result) / results[0]._result) > 0.0001 )
                throw std::logic_error("fabs((results[i]._result - results[0]._result) / results[0].Result) > 0.0001");
        }

        //print(std::string("---Before sort---"), results);
        // Сортируем results
        std::sort(results.begin(), results.end(), compare<double>);
        //print(std::string("---After sort---"), results);        
        //std::cout << "----------" << std::endl;

        minValue = results[0]._time;
        maxValue = results[resultsSize - 1]._time;

        if(resultsSize % 2 == 0)
        {
            median = (results[resultsSize / 2 - 1]._time + results[resultsSize / 2]._time)/2;
        }
        else
        {
            median = results[resultsSize / 2]._time;
        }

        // Вычисляем среднее арифметическое
        double sum = 0;
        for(auto& item : results)
            sum += item._time;
        
        avg = sum / resultsSize;

        // Вычисляем стандартное отклонение
        double sumSq = 0;
        for(auto& item : results)
            sumSq += pow(item._time - avg, 2);
        
        stdDev = sqrt(sumSq / resultsSize);

        // Вычисляем 95 перцентиль
        double rang95 = 0.95*(resultsSize-1) + 1;
        unsigned rang95okrVniz = (unsigned)floor(rang95);
        percentile_95 = results[rang95okrVniz-1]._time + (rang95-rang95okrVniz)*(results[rang95okrVniz]._time - results[rang95okrVniz-1]._time);// Доделать

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

/*
   Показатели параллельного
   вычислительного процесса
   (ускорение, эффективность)
*/
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

int main()
{
    // 1. Подготовка данных
    unsigned Nthreads = 4;
    size_t size = 300000000;
    double elVal = 0.001;
    VectorRam<double> v(size);
    v.InitByVal(elVal);
    //v.PrintToConsole();
    
    // 2. Запуск тестов и получение массива результатов
    // 2.1 Последовательный алгоритм
    auto testResults_seq = TestHelper::LaunchSum(v);
    std::cout << "Seq: testResults_seq size = " << testResults_seq.size() << std::endl;
    for(auto& res : testResults_seq)
        res.Print();
    // 2.2 Параллельный алгоритм
    auto testResults_par = TestHelper::LaunchSum(v, Nthreads);
    std::cout << "Parallel: testResults size = " << testResults_par.size() << std::endl;
    for(auto& res : testResults_par)
        res.Print();

    // 3. Статистическая обработка результатов
    CalculationStatistics stat_seq{testResults_seq};
    stat_seq.Print();

    CalculationStatistics stat_par{testResults_par};
    stat_par.Print();

    // 4. Вычисляем ускорение и эффективность
    ParallelCalcIndicators parallelCalcIndicators(stat_seq, stat_par, Nthreads);
    parallelCalcIndicators.Print();
}