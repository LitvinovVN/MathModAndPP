// g++  main.cpp -o app -fopenmp
// nvcc main.cpp -o app -Xcompiler="/openmp" -x cu -allow-unsupported-compiler
#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>
#include <functional>
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std::chrono;

/////////////// CUDA ////////////////

/// @brief Структура для хранения параметров видеокарты
struct CudaDeviceProperties
{
    bool IsInitialized = false;

    int Major;
    int Minor;
    std::string Name;
    size_t TotalGlobalMem;
    size_t SharedMemoryPerBlock;
    size_t RegsPerBlock;
    int WarpSize;
    size_t MemPitch;
    size_t MaxThreadsPerBlock;
    int MultiProcessorCount;
    bool DeviceOverlap;

    void Print()
    {
        if(!IsInitialized)
        {
            std::cout << "CudaDeviceProperties object is not initialized!" << std::endl;
            return;
        }            

        std::cout << "Major revision number:         " <<  Major                << std::endl;
        std::cout << "Minor revision number:         " <<  Minor                << std::endl;
        std::cout << "Name:                          " <<  Name                 << std::endl;
        std::cout << "Total global memory:           " <<  TotalGlobalMem       << std::endl;
        std::cout << "Total shared memory per block: " <<  SharedMemoryPerBlock << std::endl;
        std::cout << "Total registers per block:     " <<  RegsPerBlock         << std::endl;
        std::cout << "Warp size:                     " <<  WarpSize             << std::endl;
        std::cout << "Maximum memory pitch:          " <<  MemPitch             << std::endl;
        std::cout << "Maximum threads per block:     " <<  MaxThreadsPerBlock   << std::endl;
        /*for (int i = 0; i < 3; ++i)
            printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
        for (int i = 0; i < 3; ++i)
            printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
        printf("Clock rate:                    %d\n",  devProp.clockRate);
        printf("Total constant memory:         %u\n",  devProp.totalConstMem);
        printf("Texture alignment:             %u\n",  devProp.textureAlignment);
        printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));*/
        std::cout << "Number of multiprocessors:     " <<  MultiProcessorCount  << std::endl;
        //printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    }
};

/// @brief Класс для хранения вспомогательных функций Cuda
struct CudaHelper
{
    static bool IsCudaSupported()
    {
        #ifdef __NVCC__
        return true;        
        #endif
        return false;
    }

    /// @brief Возвращает количество Cuda-совместимых устройств
    /// @return Количество Cuda-совместимых устройств
    static int GetCudaDeviceNumber()
    {
        int devCount = 0;
        #ifdef __NVCC__
        cudaGetDeviceCount(&devCount);        
        #endif

        return devCount;
    }
    
    /// @brief Возвращает структуру с параметрами видеокарты
    /// @param deviceId Идентификатор Cuda-устройства
    /// @return Объект CudaDeviceProperties с параметрами видеокарты
    static CudaDeviceProperties GetCudaDeviceProperties(int deviceId = 0)
    {
        CudaDeviceProperties prop;
        #ifdef __NVCC__        
        // Get device properties
        printf("\nCUDA Device #%d\n", deviceId);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, deviceId);
        
        prop.IsInitialized = true;
        prop.Major = devProp.major;
        prop.Minor = devProp.minor;        
        prop.Name = std::string(devProp.name);        
        prop.TotalGlobalMem = devProp.totalGlobalMem;
        prop.SharedMemoryPerBlock = devProp.sharedMemPerBlock;
        prop.RegsPerBlock = devProp.regsPerBlock;
        prop.WarpSize = devProp.warpSize;
        prop.MemPitch = devProp.memPitch;
        prop.MaxThreadsPerBlock = devProp.maxThreadsPerBlock;
        //for (int i = 0; i < 3; ++i)
        //    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
        //for (int i = 0; i < 3; ++i)
        //    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
        //printf("Clock rate:                    %d\n",  devProp.clockRate);
        //printf("Total constant memory:         %u\n",  devProp.totalConstMem);
        //printf("Texture alignment:             %u\n",  devProp.textureAlignment);
        prop.DeviceOverlap = devProp.deviceOverlap;
        prop.MultiProcessorCount = devProp.multiProcessorCount;
        //printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));//*/
        #endif
        return prop;
    }

    // Print device properties
    static void PrintCudaDeviceProperties(int deviceId = 0)
    {
        #ifdef __NVCC__        
        // Get device properties
        std::cout << "\nCUDA Device #"                 << deviceId                      << std::endl;
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, deviceId);
        
        std::cout << "Major revision number:         " << devProp.major                << std::endl;
        std::cout << "Minor revision number:         " << devProp.minor                << std::endl;
        std::cout << "Name:                          " << devProp.name                 << std::endl;
        std::cout << "Total global memory:           " << devProp.totalGlobalMem       << std::endl;
        std::cout << "Total shared memory per block: " << devProp.sharedMemPerBlock    << std::endl;
        std::cout << "Total registers per block:     " << devProp.regsPerBlock         << std::endl;
        std::cout << "Warp size:                     " << devProp.warpSize             << std::endl;
        std::cout << "Maximum memory pitch:          " << devProp.memPitch             << std::endl;
        std::cout << "Maximum threads per block:     " << devProp.maxThreadsPerBlock   << std::endl;
        /*for (int i = 0; i < 3; ++i)
            printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
        for (int i = 0; i < 3; ++i)
            printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
        printf("Clock rate:                    %d\n",  devProp.clockRate);
        printf("Total constant memory:         %u\n",  devProp.totalConstMem);
        printf("Texture alignment:             %u\n",  devProp.textureAlignment);
        printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));*/
        std::cout << "Number of multiprocessors:     " << devProp.multiProcessorCount  << std::endl;
        //printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
        #else
        std::cout << "printDevProp(): CUDA is not supported!" << std::endl;
        #endif
    }


    

    

};
/////////////////// CUDA (END) /////////////////////////

/// @brief Структура для хранения поддерживаемых библиотек
struct LibSupport
{
    bool IsOpenMP = false;// Поддержка OpenMP
    bool IsCuda   = false;// Поддержка CUDA

    LibSupport()
    {
        // OpenMP
        #ifdef _OPENMP
        IsOpenMP = true;
        #endif

        // CUDA        
        #ifdef __NVCC__
        IsCuda = true;        
        #endif
    }

    void Print()
    {
        std::cout << "Supported libs: ";
        if (IsOpenMP) 
            std::cout << "OpenMP ";
        if (IsCuda) 
            std::cout << "CUDA ";
        std::cout << std::endl;
    }
};

/// @brief Структура для хранения методов обработки массивов
struct ArrayRamHelper
{
    ////////////////////////// Суммирование элементов массива (начало) /////////////////////////////

    ///// Последовательное суммирование /////
    template<typename T>
    static T Sum(T* data, size_t indStart, size_t indEnd)
    {
        T result = 0;
        for (size_t i = indStart; i <= indEnd; i++)
        {
            result += data[i];
        }
        return result;
    }

    template<typename T>
    static T Sum(T* data, size_t size)
    {
        T result = Sum(data, 0, size-1);
        return result;
    }
    ///////////////////////////////////////////////

    ///// Суммирование с помощью std::thread //////
    // Функция для исполнения потоком std::thread
    template<typename T>
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

    template<typename T>
    static T Sum(T* data, size_t indStart, size_t indEnd, unsigned threadsNum)
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

            threads.push_back(std::thread(SumThread<T>, data, thIndStart, thIndEnd, std::ref(sum), std::ref(m)));
        }
        
        for(auto& th : threads)
        {
            th.join();
        }

        return sum;
    }

    template<typename T>
    static T Sum(T* data, size_t size, unsigned threadsNum)
    {
        return Sum(data, 0, size - 1, threadsNum);
    }
    ///////////////////////////////////////////////

    ///// Суммирование с помошью OpenMP /////
    template<typename T>
    static T SumOpenMP(T* data, size_t indStart, size_t indEnd, unsigned threadsNum)
    {
        #ifdef _OPENMP
        omp_set_num_threads(threadsNum);
        T sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (long long i = indStart; i <= indEnd; i++)
        {
            sum += data[i];
        }
        return sum;
        #else
            throw std::runtime_error("OpenMP not supported!");
        #endif
    }

    template<typename T>
    static T SumOpenMP(T* data, size_t size, unsigned threadsNum)
    {
        return SumOpenMP(data, 0, size - 1, threadsNum);
    }
    ////////////////////////// Суммирование элементов массива (конец) /////////////////////////////

    /*  ---   Другие алгоритмы   ---  */

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

};


template<typename T>
struct FuncResult
{
    bool        _status;
    T           _result;
    long long   _time;

    FuncResult(bool status, T result, double time) : 
        _status(status), _result(result), _time(time)
    { }

    void Print()
    {
        std::cout << "[val: " << _result
                  << "; time: " << _time << "]" << std::endl;
    }
    
    static bool compare(const FuncResult<T>& left, const FuncResult<T>& right) 
    { 
        return left._time < right._time; 
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
        T result = ArrayRamHelper::Sum(v.data, indStart, indEnd);
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
        T result = ArrayRamHelper::Sum(v.data, indStart, indEnd, threadsNum);
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);        
        auto t = duration.count();

        return FuncResult<T>(calcStatus, result, t);
    }

    template<typename T>
    static
    FuncResult<T> Sum(VectorRam<T>& v, unsigned threadsNum)
    {
        return Sum(v, 0, v.size - 1, threadsNum);
    }

    /////////////////// OpenMP ////////////////////
    template<typename T>
    static
    FuncResult<T> SumOpenMP(VectorRam<T>& v, size_t indStart, size_t indEnd, unsigned threadsNum)
    {
        bool calcStatus = true;
        auto start = high_resolution_clock::now();
        T result = ArrayRamHelper::SumOpenMP(v.data, indStart, indEnd, threadsNum);
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);        
        auto t = duration.count();

        return FuncResult<T>(calcStatus, result, t);
    }

    template<typename T>
    static
    FuncResult<T> SumOpenMP(VectorRam<T>& v, unsigned threadsNum)
    {
        return SumOpenMP(v, 0, v.size - 1, threadsNum);
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

    template<typename T>
    static auto LaunchSumOpenMP(VectorRam<T>& v, unsigned Nthreads)
    {
        std::cout << "-------LaunchSumOpenMP(VectorRam<T>& v, unsigned Nthreads) Start ------" << std::endl;
        auto iterNum = 10;
        std::vector<FuncResult<T>> results;

        #ifdef _OPENMP

        for(unsigned i{0}; i < iterNum; i++)
        {
            FuncResult<T> res = VectorRamHelper::SumOpenMP(v, Nthreads);
            results.push_back(res);
        }

        #endif
        
        std::cout << "-------LaunchSumOpenMP(VectorRam<T>& v, unsigned Nthreads) End --------" << std::endl;
        return results;
    }
};



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
            if(results[i]._status == false)
                throw std::logic_error("results[i].Status = 0");
            
            if( fabs((results[i]._result - results[0]._result) / (double)results[0]._result) > 0.0001 )
                throw std::logic_error("fabs((results[i]._result - results[0]._result) / results[0].Result) > 0.0001");
        }

        //print(std::string("---Before sort---"), results);
        // Сортируем results
        std::sort(results.begin(), results.end(), FuncResult<T>::compare);
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
    // 2.2 Параллельный алгоритм std::thread
    auto testResults_par = TestHelper::LaunchSum(v, Nthreads);
    std::cout << "Parallel: testResults size = " << testResults_par.size() << std::endl;
    for(auto& res : testResults_par)
        res.Print();
    // 2.3 Параллельный алгоритм OpenMP
    auto testResults_par_OpenMP = TestHelper::LaunchSumOpenMP(v, Nthreads);
    std::cout << "Parallel OpenMP: testResults size = " << testResults_par_OpenMP.size() << std::endl;
    for(auto& res : testResults_par_OpenMP)
        res.Print();

    // 3. Статистическая обработка результатов
    CalculationStatistics stat_seq{testResults_seq};
    stat_seq.Print();

    CalculationStatistics stat_par{testResults_par};
    stat_par.Print();

    CalculationStatistics stat_par_OpenMP;
    try
    {
        stat_par_OpenMP = CalculationStatistics{testResults_par_OpenMP};
        stat_par_OpenMP.Print();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    

    // 4. Вычисляем ускорение и эффективность
    std::cout << "--- std::thread ---" << std::endl;
    ParallelCalcIndicators parallelCalcIndicators(stat_seq, stat_par, Nthreads);
    parallelCalcIndicators.Print();

    try
    {
        std::cout << "--- OpenMP ---" << std::endl;
        ParallelCalcIndicators parallelCalcIndicators_OpenMP(stat_seq, stat_par_OpenMP, Nthreads);
        parallelCalcIndicators_OpenMP.Print();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
        
    // Вызов функции суммирования с помощью OpenMP
    try
    {
        double sum = ArrayRamHelper::SumOpenMP(v.data, 0, v.size, Nthreads);
        std::cout << "ArrayRamHelper::SumOpenMP(v.data, 0, v.size): " << sum << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
    
    LibSupport support;
    support.Print();

    std::cout << "Cuda devices number: " << CudaHelper::GetCudaDeviceNumber() << std::endl;
    CudaHelper::PrintCudaDeviceProperties();

    auto devProps = CudaHelper::GetCudaDeviceProperties();
    devProps.Print();
}