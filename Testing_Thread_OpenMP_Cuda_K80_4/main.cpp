// set PATH=%PATH%;C:\mingw64\bin
// g++  main.cpp -o app -fopenmp -O3 -Wall
// g++  main.cpp -o app -lpthread -O3 -Wall
// nvcc main.cpp -o app -Xcompiler="/openmp -Wall"  -x cu -allow-unsupported-compiler -std=c++17
// nvcc main.cpp -o app -Xcompiler="-fopenmp -Wall" -x cu
#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>
#include <functional>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <map>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __NVCC__
#include "kernels.cu"
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
    int AsyncEngineCount;// Number of asynchronous engines
    size_t MemoryClockRate;//Memory Clock Rate (KHz)
    int MemoryBusWidth;//Memory Bus Width (bits)
    
    double GetPeakMemoryBandwidthGBs()
    {
        return 2.0*MemoryClockRate*(MemoryBusWidth/8)/1.0e6;
    }
                 

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
        std::cout << "Number of asynchronous engines: " <<  AsyncEngineCount           << std::endl;
        std::cout << "Memory Clock Rate (KHz):        " << MemoryClockRate             << std::endl;
        std::cout << "Memory Bus Width (bits):        " << MemoryBusWidth              << std::endl;
        std::cout << "Peak Memory Bandwidth (GB/s):   " << GetPeakMemoryBandwidthGBs() << std::endl;
    }
};

/// @brief Класс для хранения вспомогательных функций Cuda
struct CudaHelper
{
    static bool IsCudaSupported()
    {
        #ifdef __NVCC__
        return true;        
        #else
        return false;
        #endif
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
    /// @return Объект CudaDeviceProperties с параметрами видеокарты (поле IsInitialized = true в случае успеха, иначе IsInitialized = false)
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
        prop.AsyncEngineCount = devProp.asyncEngineCount;
        prop.MemoryClockRate = devProp.memoryClockRate;
        prop.MemoryBusWidth = devProp.memoryBusWidth;        
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
        
        std::cout << "Major revision number:          " << devProp.major                << std::endl;
        std::cout << "Minor revision number:          " << devProp.minor                << std::endl;
        std::cout << "Name:                           " << devProp.name                 << std::endl;
        std::cout << "Total global memory:            " << devProp.totalGlobalMem       << std::endl;
        std::cout << "Total shared memory per block:  " << devProp.sharedMemPerBlock    << std::endl;
        std::cout << "Total registers per block:      " << devProp.regsPerBlock         << std::endl;
        std::cout << "Warp size:                      " << devProp.warpSize             << std::endl;
        std::cout << "Maximum memory pitch:           " << devProp.memPitch             << std::endl;
        std::cout << "Maximum threads per block:      " << devProp.maxThreadsPerBlock   << std::endl;
        /*for (int i = 0; i < 3; ++i)
            printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
        for (int i = 0; i < 3; ++i)
            printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
        printf("Clock rate:                    %d\n",  devProp.clockRate);
        printf("Total constant memory:         %u\n",  devProp.totalConstMem);
        printf("Texture alignment:             %u\n",  devProp.textureAlignment);
        printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));*/
        std::cout << "Number of multiprocessors:      " << devProp.multiProcessorCount  << std::endl;
        //printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
        std::cout << "Number of asynchronous engines: " <<  devProp.asyncEngineCount         << std::endl;
        std::cout << "Memory Clock Rate (KHz):        " << devProp.memoryClockRate           << std::endl;
        std::cout << "Memory Bus Width (bits):        " << devProp.memoryBusWidth            << std::endl;
        std::cout << "Peak Memory Bandwidth (GB/s):   " << 2.0*devProp.memoryClockRate*(devProp.memoryBusWidth/8)/1.0e6 << std::endl;
        #else
        std::cout << "printDevProp(): CUDA is not supported!" << std::endl;
        #endif
    }

    static void WriteGpuSpecs(std::ofstream& out)
    {
        #ifdef __NVCC__
        out << "WriteGpuSpecs()" << std::endl;

        int nDevices;
        cudaGetDeviceCount(&nDevices);
        for (int i = 0; i < nDevices; i++)
        {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            out << "Device Number: "             << i << std::endl;
            out << "  Device name: "             << prop.name << std::endl;
            out << "  Compute capability: "      << prop.major << "." << prop.minor << std::endl;
            out << "  MultiProcessorCount: "     << prop.multiProcessorCount << std::endl;
            out << "  asyncEngineCount: "        <<  prop.asyncEngineCount<< " (Number of asynchronous engines)" << std::endl;
            out << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << std::endl;
            out << "  Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
            out << "  Peak Memory Bandwidth (GB/s): "
                << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << std::endl;
        }
        #else
        out << "printDevProp(): CUDA is not supported!" << std::endl;
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


/// @brief Структура для хранения методов обработки массивов T*
struct ArrayHelper
{
    ////////////////////////// Суммирование элементов массива (начало) /////////////////////////////

    ///// Последовательное суммирование на CPU /////
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

    ///// Суммирование с помощью std::thread на CPU //////
    // Структура для передачи аргументов в потоковую функцию
    template<typename T>
    struct SumThreadArgs
    {
        T* data;
        size_t indStart;
        size_t indEnd;
        T& sum;
        std::mutex& m;

        SumThreadArgs(T* data,
            size_t indStart,
            size_t indEnd,
            T& sum,
            std::mutex& m) : 
                data(data),
                indStart(indStart),
                indEnd(indEnd),
                sum(sum),
                m(m)
        {}
    };

    // Функция для исполнения потоком std::thread
    template<typename T>
    static void SumThread(SumThreadArgs<T> args)
    {
        T* data = args.data;
        auto indStart = args.indStart;
        auto indEnd = args.indEnd;
        T local_sum = 0;
        
        for (size_t i = indStart; i <= indEnd; i++)
        {
            local_sum += data[i];
        }
        
        {
            std::lock_guard<std::mutex> lock(args.m);
            args.sum += local_sum;
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
                        
            SumThreadArgs<T> args(data, thIndStart, thIndEnd, sum, m);
            threads.push_back(std::thread(SumThread<T>, args));
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

    ///// Суммирование с помошью OpenMP на CPU /////
    template<typename T>
    static T SumOpenMP(T* data, size_t indStart, size_t indEnd, unsigned threadsNum)
    {
        #ifdef _OPENMP
        omp_set_num_threads(threadsNum);
        T sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (long long i = (long long)indStart; i <= (long long)indEnd; i++)
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


    ///// Суммирование с помощью Cuda /////      

    template<typename T>
    static T SumCuda(T* dev_arr, size_t indStart, size_t indEnd, unsigned blocksNum, unsigned threadsNum)
    {
        #ifdef __NVCC__

        size_t length = indEnd - indStart + 1;
                        
        #ifdef DEBUG
        std::cout << "T Sum(" << dev_arr << ", "
                  << length << ", "<< blocksNum << ", "
                  << threadsNum << ")" <<std::endl;
        #endif
        
        T sum{0};
        T* dev_sum;
        cudaMalloc(&dev_sum, sizeof(T));
        //cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

        // Выделяем в распределяемой памяти каждого SM массив для хранения локальных сумм каждого потока блока
        unsigned shared_mem_size = threadsNum * sizeof(T);
        #ifdef DEBUG
        std::cout << "shared_mem_size = " << shared_mem_size << std::endl;
        #endif
        // Выделяем в RAM и глобальной памяти GPU массив для локальных сумм каждого блока
        T* block_sum = (T*)malloc(blocksNum * sizeof(T));
        T* dev_block_sum;
        cudaMalloc(&dev_block_sum, blocksNum * sizeof(T));
        kernel_sum<<<blocksNum, threadsNum, shared_mem_size>>>(dev_arr, length, dev_block_sum, dev_sum);

        //cudaMemcpy(&sum, dev_sum, sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum, dev_block_sum, blocksNum * sizeof(T), cudaMemcpyDeviceToHost);
        for(unsigned i=0; i<blocksNum;i++)
        {
            //std::cout << "block_sum[" << i << "] = " << block_sum[i] << std::endl;
            sum += block_sum[i];
        }

        #ifdef DEBUG
        std::cout << "Sum is " << sum << std::endl;
        #endif

        free(block_sum);
        cudaFree(dev_block_sum);

        return sum;

        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }

    template<typename T>
    static T SumCuda(T* data, size_t size, unsigned blocksNum, unsigned threadsNum)
    {
        return SumCuda(data, 0, size - 1, blocksNum, threadsNum);
    }

    ////////////////////////// Суммирование элементов массива (конец) /////////////////////////////

    /*  ---   Другие алгоритмы   ---  */

};


//////////////////////////////////////////////////////////////////////////////////////////

//////// Определения классов, моделирующих вектор в N-мерном пространстве ////////////////

template<typename T>
class Vector
{
public:
    virtual void InitByVal(T val) = 0;
    virtual void Print() const = 0;
    virtual size_t Size() const = 0;
};

template<typename T>
class VectorRam : public Vector<T>
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

    void InitByVal(T val) override
    {
        for (size_t i = 0; i < size; i++)
        {
            data[i] = val;
        }        
    }

    void Print() const override
    {
        for (size_t i = 0; i < size; i++)
        {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;     
    }

    size_t Size() const override
    {
        return size;
    }

};


/// @brief Вектор (в GPU) 
/// @tparam T Тип элементов вектора
template<typename T>
class VectorGpu : public Vector<T>
{
public:
    // Количество элементов вектора
    size_t _size = 0;
    // Указатель на массив в видеопамяти
    T* _dev_data = nullptr;
    // Флаг инициализации вектора
    // false - неинициализирован, true - инициализирован
    bool _isInitialized = false;

    VectorGpu(size_t size) : _size(size)
    {
        #ifdef __NVCC__
        //std::cout << "VectorGpu(size_t size) constructor started...\n";

        if (_size == 0)
        {
            std::string mes = "Cannot initialize vector of _size = 0";
            //std::cerr << mes << std::endl;
            throw std::logic_error(mes);
        }

        cudaError_t cudaResult = cudaMalloc(&_dev_data, size*sizeof(T));
        if (cudaResult != cudaSuccess)
        {
            std::string msg("Could not allocate device memory for VectorGpu: ");
            msg += cudaGetErrorString(cudaResult);
            throw std::runtime_error(msg);
        }

        //std::cout << "VectorGpu(size_t size): Device memory for VectorGpu allocated!\n";
        
        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }

    VectorGpu(VectorRam<T> vecRam) : _size(vecRam.GetSize())
    {
        #ifdef __NVCC__
        std::cout << "VectorGpu(VectorRam<T> vecRam) constructor started...\n";

        if (_size == 0)
        {
            std::string mes = "Cannot initialize vector of _size = 0";
            //std::cerr << mes << std::endl;
            throw std::logic_error(mes);
        }

        cudaError_t cudaResult = cudaMalloc(&_dev_data, _size*sizeof(T));
        if (cudaResult != cudaSuccess)
        {
            std::string msg("Could not allocate device memory for VectorGpu: ");
            msg += cudaGetErrorString(cudaResult);
            throw std::runtime_error(msg);
        }

        std::cout << "VectorGpu(VectorRam<T> vecRam): Device memory for VectorGpu allocated!\n";
    
        // Копируем данные в видеопамять
        cudaResult = cudaMemcpy(_dev_data, vecRam.Get_data_pointer(), _size*sizeof(T), cudaMemcpyHostToDevice);
        if (cudaResult != cudaSuccess)
        {
            std::string msg("Could not copy data from RAM to device memory: ");
            msg += cudaGetErrorString(cudaResult);
            throw std::runtime_error(msg);
        }
        //std::cout << "cudaMemCpy OK!\n";

        // Устанавливаем флаг инициализации вектора
        _isInitialized = true;

        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }

    ~VectorGpu()
    {
        //std::cout << "~VectorGpu(): " << this << " destructed!\n";
    }

    /// @brief Проверяет состояние вектора
    bool CheckState()
    {
        if(!_isInitialized)
            return false;

        if(_size < 1)
            return false;

        if(_dev_data == nullptr)
            return false;

        return true;
    }

    /// @brief Возвращает сумму элементов вектора
    /*FuncResultScalar<T> Sum(unsigned blocksNum, unsigned threadsNum)
    {
        if(!CheckState())
            throw std::logic_error("Vector is not initialized!");   
        
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();   

        T result = CudaHelper<T>::Sum(_dev_data, _size, blocksNum, threadsNum);       

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        //std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;

        FuncResultScalar<T> res{true, result, std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()};
        //res.Print();

        return res;
    }*/

    /// @brief Освобождаем массив в видеопамяти
    void Clear_dev_data()
    {
        #ifdef __NVCC__
        if(_dev_data != nullptr)
        {
            cudaFree(_dev_data);
            _dev_data = nullptr;
            _isInitialized = false;
            //std::cout << "Device memory for VectorGpu cleared!\n";
        }
        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }

    /// @brief Возвращает указатель на данные в видеопамяти
    /// @return 
    #ifdef __NVCC__
    __host__ __device__
    #endif
    T* Get_dev_data_pointer()
    {
        return _dev_data;
    }

    /*#ifdef __NVCC__
    __host__ __device__
    #endif*/
    size_t Size() const override
    {
        return _size;
    }
    
    /// @brief Инициализирует вектор числом
    void InitByVal(double value) override
    {
        #ifdef __NVCC__
        // Создаём временный массив
        T* tmp = new T[_size];
        
        // Инициализируем временный массив        
        for (size_t i = 0; i < _size; i++)
        {
            tmp[i] = value;
            //std::cout << tmp[i] << " ";
        }
        //std::cout << std::endl;

        // Копируем данные из временного массива в видеопамять
        cudaError_t cudaResult = cudaMemcpy(_dev_data, tmp, _size*sizeof(T), cudaMemcpyHostToDevice);
        if (cudaResult != cudaSuccess)
        {
            std::string msg("Could not copy data from RAM to device memory: ");
            msg += cudaGetErrorString(cudaResult);
            throw std::runtime_error(msg);
        }

        //std::cout << "cudaMemCpy OK!\n";

        // Освобождаем временный массив
        delete[] tmp;

        // Устанавливаем флаг инициализации вектора
        _isInitialized = true;

        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }

    /// @brief Инициализирует вектор числами из диапазона от start до end
    void InitVectorByRange(double start, double end)
    {
        #ifdef __NVCC__
        // Создаём временный массив
        T* tmp = new T[_size];
        size_t cnt = 0;

        // Инициализируем временный массив
        auto step = (end-start)/(_size-1);
        for (auto i = start; i < end+step/2; i+=step)
        {
            tmp[cnt++] = i;
            //std::cout << tmp[cnt-1] << " ";
        }
        std::cout << std::endl;

        // Копируем данные из временного массива в видеопамять
        cudaError_t cudaResult = cudaMemcpy(_dev_data, tmp, _size*sizeof(T), cudaMemcpyHostToDevice);
        if (cudaResult != cudaSuccess)
        {
            std::string msg("Could not copy data from RAM to device memory: ");
            msg += cudaGetErrorString(cudaResult);
            throw std::runtime_error(msg);
        }

        //std::cout << "cudaMemCpy OK!\n";

        // Освобождаем временный массив
        delete[] tmp;

        // Устанавливаем флаг инициализации вектора
        _isInitialized = true;

        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }

    void Print() const override
    {
        #ifdef __NVCC__
        print_kernel<T><<<1,1>>>(_dev_data, 0, _size);
        cudaDeviceSynchronize();
        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }
};


/////////////////////////////////////////////////

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
        T result = ArrayHelper::Sum(v.data, indStart, indEnd);
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
        T result = ArrayHelper::Sum(v.data, indStart, indEnd, threadsNum);
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
        T result = ArrayHelper::SumOpenMP(v.data, indStart, indEnd, threadsNum);
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

class VectorGpuHelper
{
public:
    template<typename T>
    static
    FuncResult<T> SumCuda(VectorGpu<T>& v, size_t indStart, size_t indEnd, unsigned NumBlocks, unsigned Nthreads)
    {
        bool calcStatus = true;
        auto start = high_resolution_clock::now();
        T result = ArrayHelper::SumCuda(v._dev_data, indStart, indEnd, NumBlocks, Nthreads);
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);        
        auto t = duration.count();

        return FuncResult<T>(calcStatus, result, t);
    }

    template<typename T>
    static
    FuncResult<T> SumCuda(VectorGpu<T>& v, unsigned NumBlocks, unsigned Nthreads)
    {
        return SumCuda(v, 0, v.Size() - 1, NumBlocks, Nthreads);
    }
};

///////////////////////////////////////////////////////////////////

/// @brief Параметры проведения численного эксперимента
struct TestParams
{
    unsigned IterNum = 20;// Количество итераций
    // --- Дополнить ---
};

/// @brief Вспомогательный класс для запуска численных экспериментов
class TestHelper
{
public:
    template<typename T>
    static std::vector<FuncResult<T>> LaunchSum(VectorRam<T>& v, TestParams p)
    {
        std::cout << "-------LaunchSum(VectorRam<T>& v) Start ------" << std::endl;
        auto iterNum = p.IterNum;
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
    static std::vector<FuncResult<T>> LaunchSum(VectorRam<T>& v, unsigned Nthreads, TestParams p)
    {
        std::cout << "-------LaunchSum(VectorRam<T>& v, unsigned Nthreads) Start ------" << std::endl;
        auto iterNum = p.IterNum;
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
    static std::vector<FuncResult<T>> LaunchSumOpenMP(VectorRam<T>& v, unsigned Nthreads, TestParams p)
    {
        std::cout << "-------LaunchSumOpenMP(VectorRam<T>& v, unsigned Nthreads) Start ------" << std::endl;
        auto iterNum = p.IterNum;
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

    
    template<typename T>
    static std::vector<FuncResult<T>> LaunchSumCuda(VectorGpu<T>& v, unsigned NumBlocks, unsigned Nthreads, TestParams p)
    {
        std::cout << "-------LaunchSumCuda(VectorGpu<T>& v, unsigned NumBlocks, unsigned Nthreads, TestParams p) Start ------" << std::endl;
        std::vector<FuncResult<T>> results;

        #ifdef __NVCC__

        auto iterNum = p.IterNum;
        for(unsigned i{0}; i < iterNum; i++)
        {
            FuncResult<T> res = VectorGpuHelper::SumCuda(v, NumBlocks, Nthreads);
            results.push_back(res);
        }

        #endif
        
        std::cout << "-------LaunchSumCuda(VectorGpu<T>& v, unsigned NumBlocks, unsigned Nthreads, TestParams p) End --------" << std::endl;
        return results;
    }
};



/// @brief Статистические параметры результатов численного эксперимента
struct CalculationStatistics
{
    // Количество запусков численного эксперимента
    unsigned numIter = 0;
    // Минимальное значение
    double minValue = 0;
    // Максимальное значение
    double maxValue = 0;
    // Среднее арифметическое
    double avg = 0;
    // Медиана
    double median = 0;
    // 95 процентиль
    double percentile_95 = 0;
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

///////////////////////////////////////////////////////////////////////////////////////


// Тестирование функций класса ArrayHelper
bool TestArrayHelper()
{
    // Вызов функции суммирования с помощью OpenMP
    try
    {
        size_t size = 10;
        double* data = new double[size];
        for (size_t i = 0; i < size; i++)
        {
            data[i] = 0.1;
        }
        

        int Nthreads = 4;
        double sum = ArrayHelper::SumOpenMP(data, 0, size, Nthreads);
        std::cout << "ArrayRamHelper::SumOpenMP(v.data, 0, v.size): " << sum << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return false;
    }
    return true;
}

/////////////////////////////////
/// Проверка работы VectorGpu ///
bool TestVectorGpu()
{
    // Добавить разные тесты
    try
    {
        VectorGpu<double> v1{350000};        
        v1.InitByVal(0.001);
        //v1.Print();        
    
        for(int i = 1; i <= 5; i++)
        {
            for(int j = 1; j <= 5; j++)
            {
                auto res = ArrayHelper::SumCuda(v1.Get_dev_data_pointer(), v1.Size(),i,j);
                std::cout << i << ", " << j << ": ";
                //res.Print();
                std::cout << res << std::endl;
            }
        }
        system("pause");

    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return false;
    }

    return true;
}
///////////////////////////////////////////////////////////
/// Тестирование функции суммирования элементов массива ///
bool TestSum()
{
    TestParams testParams;

    // 1. Подготовка данных
    unsigned Nthreads = 4;
    size_t size = 1000000000;
    double elVal = 0.001;
    VectorRam<double> v(size);
    v.InitByVal(elVal);
    //v.Print();

    VectorGpu<double>* vGpu_p = nullptr;
    try
    {
        vGpu_p = new VectorGpu<double>(size);
        vGpu_p->InitByVal(elVal);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    // 2. Запуск тестов и получение массива результатов
    // 2.1 Последовательный алгоритм
    auto testResults_seq = TestHelper::LaunchSum(v, testParams);
    std::cout << "Seq: testResults_seq size = " << testResults_seq.size() << std::endl;
    for(auto& res : testResults_seq)
        res.Print();
    // 2.2 Параллельный алгоритм std::thread
    auto testResults_par = TestHelper::LaunchSum(v, Nthreads, testParams);
    std::cout << "Parallel: testResults size = " << testResults_par.size() << std::endl;
    for(auto& res : testResults_par)
        res.Print();
    // 2.3 Параллельный алгоритм OpenMP
    auto testResults_par_OpenMP = TestHelper::LaunchSumOpenMP(v, Nthreads, testParams);
    std::cout << "Parallel OpenMP: testResults size = " << testResults_par_OpenMP.size() << std::endl;
    for(auto& res : testResults_par_OpenMP)
        res.Print();

    // 2.4 Параллельный алгоритм Cuda
    int numBlocks = 10;
    auto testResults_par_Cuda = TestHelper::LaunchSumCuda(*vGpu_p, numBlocks, Nthreads, testParams);
    std::cout << "Parallel CUDA: testResults size = " << testResults_par_Cuda.size() << std::endl;
    for(auto& res : testResults_par_Cuda)
        res.Print();

    // 3. Статистическая обработка результатов
    CalculationStatistics stat_seq{testResults_seq};
    std::cout << "CalculationStatistics seq: " << std::endl;
    stat_seq.Print();

    CalculationStatistics stat_par{testResults_par};
    std::cout << "CalculationStatistics parallel std::thread: " << std::endl;
    stat_par.Print();

    CalculationStatistics stat_par_OpenMP;
    try
    {
        stat_par_OpenMP = CalculationStatistics{testResults_par_OpenMP};
        std::cout << "CalculationStatistics parallel OpenMP: " << std::endl;
        stat_par_OpenMP.Print();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';        
    }

    CalculationStatistics stat_par_Cuda;
    try
    {
        stat_par_Cuda = CalculationStatistics{testResults_par_Cuda};
        std::cout << "CalculationStatistics parallel Cuda: " << std::endl;
        stat_par_Cuda.Print();
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
        return false;
    }

    try
    {
        std::cout << "--- CUDA ---" << std::endl;
        ParallelCalcIndicators parallelCalcIndicators_Cuda(stat_seq, stat_par_Cuda, numBlocks*Nthreads);
        parallelCalcIndicators_Cuda.Print();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return false;
    }

    return true;
}
//////////////////////////////////////////////////////


//////////////// Работа с консолью (начало) ///////////////////

/// @brief Вспомогательный класс для работы с консолью
struct ConsoleHelper
{
    /// @brief Запрашивает у пользователя целое число
    /// @param message Сообщение для пользователя
    /// @param errorMessage Сообщение об ошибке
    /// @return Введённое пользователем число
    static std::string GetStringFromUser(std::string message)
    {
        std::cout << message;
        std::string userInput;
        if(char(std::cin.peek()) == '\n')
            std::cin.ignore();

        if (std::cin.fail()) 
        {
            std::cin.clear();
            std::cin.ignore(32767, '\n');
        }
        getline(std::cin, userInput);
        return userInput;
    }

    /// @brief Запрашивает у пользователя целое число
    /// @param message Сообщение для пользователя
    /// @param errorMessage Сообщение об ошибке
    /// @return Введённое пользователем число
    static int GetIntFromUser(std::string message, std::string errorMessage = "Error! Enter integer number")
    {
        while (1)
        {
            try
            {
                std::cout << message;
                std::string userInput;
                std::cin >> userInput;
                int value = std::stoi(userInput);
            
                return value;
            }
            catch(const std::exception& e)
            {
                std::cout << errorMessage << std::endl;
            }
        }
    }
};
//////////////// Работа с консолью (конец) ///////////////////


//////////////// Файловая система (начало) ///////////////////

/// @brief Класс для работы с файловой системой
//#include <filesystem> // C++17
#include <fstream>
#include "sys/stat.h"
class FileSystemHelper
{
public:
    /// @brief Комбинирует имя папки и имя файла в путь к файлу
    /// @param dir_name 
    /// @param file_name 
    /// @return 
    static std::string CombinePath(const std::string& dir_name, const std::string& file_name)
    {
        return std::string{dir_name + "/" + file_name};
    }

    static std::string CombinePath()
    {
        std::cout << "Enter dir name: ";
        std::string dirName;
        std::cin >> dirName;

        std::cout << "Enter file name: ";
        std::string fileName;
        std::cin >> fileName;

        std::string path = CombinePath(dirName, fileName);
        std::cout << "Path: " << path << std::endl;

        return path;
    }


    /// @brief Проверяет существование файла
    /// @return true - существует; false - не существует
    static bool IsFileExists(const std::string& path_file)
    {
        std::ifstream iff(path_file);
        //std::cout << "iff.good()" << iff.good() << std::endl;
        return iff.good();

        // C++17 
        /*if(std::filesystem::exists(path_file))
            return true;
        else
            return false;*/
    }

    /// @brief Проверяет существование файла
    /// @return true - существует; false - не существует
    static bool IsFileExists(const std::string& dir_name, const std::string& file_name)
    {
        auto filePath = CombinePath(dir_name, file_name);
        if(IsFileExists(filePath))
            return true;
        else
            return false;
    }

    static bool IsFileExists()
    {
        std::cout << "Enter file name: ";
        std::string fileName;
        std::cin >> fileName;
        bool isExists = IsFileExists(fileName);
        if(isExists)
            std::cout << "File exists (true)" << std::endl;
        else
            std::cout << "File not exists (false)" << std::endl;

        return isExists;
    }

    /// @brief Проверяет существование каталога
    /// @return true - существует; false - не существует
    static bool IsDirExists(const std::string& path_dir)
    {
        std::string filePath = CombinePath(path_dir,"tmp");
        std::ofstream fout(filePath,std::ios::app);
        bool isExists = fout.good();
        fout.close();
        if (isExists)// Удаляем временный файл
        {
            remove(filePath.c_str());
        }
        return isExists;
        // C++17   
        //if(std::filesystem::exists(path_dir))
        //    return true;
        //return false;
    }

    static bool IsDirExists()
    {
        std::cout << "Enter dir name: ";
        std::string dirName;
        std::cin >> dirName;
        bool isExists = IsDirExists(dirName);
        if(isExists)
            std::cout << "Directory exists (true)" << std::endl;
        else
            std::cout << "Directory not exists (false)" << std::endl;

        return isExists;
    }


    /// @brief Создаёт каталог
    /// @return Результат создания нового каталога
    static bool CreateDir(const std::string& path_dir)
    {
        if(IsDirExists(path_dir))
            return false;

        int errCode = mkdir(path_dir.c_str(), S_IRWXU);
        bool result = !(bool)errCode;
        return result;
        //return std::experimental::create_directory(path_dir);
    }

    static bool CreateDir()
    {
        std::cout << "Enter dir name: ";
        std::string dirName;
        std::cin >> dirName;
        bool res = CreateDir(dirName);
        if(res)
            std::cout << "Directory created (true)" << std::endl;
        else
            std::cout << "Directory not created (false)" << std::endl;

        return res;
    }


    static bool CreateFile(const std::string& dir_name, const std::string& file_name, const std::string& string_data)
    {
        //auto filePath = CombinePath(dir_name, file_name);
        std::string filePath = file_name;
        if(dir_name.size()>0)
            filePath = CombinePath(dir_name, file_name);
        std::cout << "filePath: " << filePath << std::endl;
        std::ofstream fout(filePath);
        if(string_data != "")
            fout << string_data;
        fout.close();

        return true;
    }

    static bool CreateFile()
    {
        std::cout << "Enter dir name (. - current dir): ";
        std::string dirName;
        std::cin >> dirName;

        std::cout << "Enter file name: ";
        std::string fileName;
        std::cin >> fileName;

        bool res = CreateFile(dirName, fileName, "");
        if(res)
            std::cout << "File created (true)" << std::endl;
        else
            std::cout << "File not created (false)" << std::endl;

        return res;
    }


    static bool RemoveFile(const std::string& dir_name, const std::string& file_name)
    {
        auto filePath = CombinePath(dir_name, file_name);
        int errCode = remove(filePath.c_str());

        return !(bool)errCode;
    }

    static bool RemoveFile()
    {
        std::cout << "Enter dir name: ";
        std::string dirName;
        std::cin >> dirName;

        std::cout << "Enter file name: ";
        std::string fileName;
        std::cin >> fileName;

        bool res = RemoveFile(dirName, fileName);
        if(res)
            std::cout << "File removed (true)" << std::endl;
        else
            std::cout << "File not removed (false)" << std::endl;

        return res;
    }


    static bool RemoveDir(const std::string& dir_name)
    {        
        int errCode = remove(dir_name.c_str());

        return !(bool)errCode;
    }

    static bool RemoveDir()
    {
        std::cout << "Enter dir name: ";
        std::string dirName;
        std::cin >> dirName;
        
        bool res = RemoveDir(dirName);
        if(res)
            std::cout << "Directory removed (true)" << std::endl;
        else
            std::cout << "Directory not removed (false)" << std::endl;

        return res;
    }
};
/////////////////// Файловая система (конец) ///////////////////


//////////////// Вычислительная система (начало) //////////////

/// @brief Вычислительная система
class ComputingSystem
{
    int id{0};// Идентификатор вычислительной системы
    std::string name{"TestSystem"}; // Наименование вычислительной системы
    std::string description{"TestSystem description"}; // Описание вычислительной системы
    std::string file_name{"ComputingSystem.txt"};// Имя файла с описанием вычислительной системы
    

public:
    ComputingSystem()
    {}

    ComputingSystem(int id,
        std::string name,
        std::string description,
        std::string file_name = "ComputingSystem.txt"
        ) : id(id),
            name(name),
            description(description),
            file_name(file_name)
    {}

    /// @brief Выводит в консоль сведения о вычислительной системе
    void Print()
    {
        std::cout   << "Computing system details:"
                    << "\nid:          " << id
                    << "\nname:        " << name
                    << "\ndescription: " << description
                    << "\nfile_name:   " << file_name
                    << std::endl;
    }

    /// @brief Возвращает идентификатор вычислительной системы
    /// @return 
    int GetId() const
    {
        return id;
    }

    /// @brief Устанавливает идентификатор вычислительной системы
    /// @param id 
    void SetId(int id)
    {
        this->id = id;
    }

    /// @brief Записать сведения о вычислительной системе
    /// @param dir_name Каталог для записи
    /// @return 
    bool Serialize(const std::string& dir_name)
    {
        // Создаём каталог dir_name/id
        std::string path_dir = FileSystemHelper::CombinePath(dir_name, std::to_string(id));
        bool result = FileSystemHelper::CreateDir(path_dir);
        if (!result)
        {
            std::cerr << "Cannot create dir " << path_dir << std::endl;
            return false;
        }
        std::string data = std::to_string(id) + "\n" + name + "\n" + description + "\n";
        FileSystemHelper::CreateFile(path_dir, file_name, data);

        return true;
    }

    static ComputingSystem Deserialize(const std::string& dir_name,
                const int id,
                const std::string& file_name = "ComputingSystem.txt")
    {
        ComputingSystem computingSystem;

        std::string dir_Path = FileSystemHelper::CombinePath(dir_name,
                                std::to_string(id));
        std::string filePath = FileSystemHelper::CombinePath(dir_Path, file_name);
        std::ifstream fin(filePath);

        if(!fin.good())
            throw std::runtime_error("Error in opening file " + filePath);

        int f_id;
        std::string f_name;
        std::string f_description;

        //fin >> f_id;
        std::string f_id_str;
        std::getline(fin, f_id_str);
        f_id = std::stoi(f_id_str);
        if(f_id != id)
            throw std::runtime_error("Error in file " + filePath);

        //fin >> f_name;
        std::getline(fin, f_name);
        std::getline(fin, f_description);

        return ComputingSystem(f_id, f_name, f_description);
    }

    static ComputingSystem GetDataFromUser()
    {
        int id = ConsoleHelper::GetIntFromUser("Enter computing system id: ");
        std::string name = ConsoleHelper::GetStringFromUser("Enter computing system name: ");
        std::string description = ConsoleHelper::GetStringFromUser("Enter computing system description: ");

        return ComputingSystem(id, name, description);
    }
};


/// @brief Репозиторий сведений о вычислительных системах
class ComputingSystemRepository
{
    std::string dir_name = "ComputingSystemRepository";// Имя каталога со сведениями о вычислительных системах
    std::string file_name = "List.txt";// Имя файла со сведениями о вычислительных системах

    std::vector<int> computerSystemIds;// Вектор идентификаторов вычислительных систем

    /// @brief Проверка существования каталогов
    void CheckDirectories()
    {        
        if(!FileSystemHelper::IsDirExists(dir_name))
            FileSystemHelper::CreateDir(dir_name);
    }

    void CheckFiles()
    {
        if(!FileSystemHelper::IsFileExists(dir_name, file_name))
        {
            bool result = FileSystemHelper::CreateFile(dir_name, file_name, "ComputingSystemRepository");
            if (!result)
            {
                std::cerr << "File " + file_name + " in directory " + dir_name + " is not created!";
                exit(-1);
            }            
        }
    }

    /// @brief Считывает содержимое файла со сведениями о вычислительных системах
    /// @return 
    bool ReadFile()
    {
        std::string filePath = FileSystemHelper::CombinePath(dir_name, file_name);

        std::ifstream f(filePath);
        
        if(!f.is_open())
        {            
            std::string message = "File \"" + filePath + "\" is not opened!";
            std::cerr << message << std::endl;
            return false;
        }

        // Проверка формата файла
        std::string str;
        f >> str;
        if (str != "ComputingSystemRepository")
        {            
            std::string message = "File \"" + filePath + "\" format is not AppConfig!";
            std::cerr << message << std::endl;
            return false;
        }

        // Считываем пары "Параметр Значение"
        int value;
        while(f >> value)
        {
            //std::cout << value << std::endl;
            computerSystemIds.push_back(value);
        }

        return true;
    }

    /// @brief Записывает new_id в конец файла 
    /// @param new_id 
    /// @return 
    bool AddIdToFile(const int new_id)
    {
        std::string filePath = FileSystemHelper::CombinePath(dir_name, file_name);

        std::ofstream fout(filePath,std::ios::app);
        if(!fout.is_open())
        {
            return false;
        }

        fout << '\n' << new_id ;

        fout.close();
        return true;
    }

public:
    ComputingSystemRepository()
    {
        CheckDirectories();
        CheckFiles();
        ReadFile();
    }

    ComputingSystemRepository(std::string dir_name)
        : dir_name(dir_name)
    {
        CheckDirectories();
        CheckFiles();
        ReadFile();
    }

    bool IsExists(int computingSystemId) const
    {
        for(auto& id : computerSystemIds)
        {
            if(id == computingSystemId)
                return true;
        }

        return false;
    }

    bool TryAddComputingSystem(ComputingSystem& computingSystem)
    {
        int new_id = computingSystem.GetId();
        // Если уже есть информация о вычислительной системе
        // с таким идентификатором, информацию не добавляем
        // и возвращаем false
        if (IsExists(new_id))
            return false;

        // Записать данные о выч. системе в каталог dir_name
        computingSystem.Serialize(dir_name);

        AddIdToFile(new_id);

        computerSystemIds.push_back(new_id);

        return true;
    }

    ComputingSystem GetComputingSystem(int id)
    {
        if(!IsExists(id))
            throw std::logic_error("Computing system not found!");

        return ComputingSystem::Deserialize(dir_name, id);
    }

    /// @brief 2 Print config
    void PrintConfig()
    {
        std::cout << "dir_name: "  << dir_name  << "; ";
        std::cout << "file_name: " << file_name << std::endl;
    }

    /// @brief 3 Print computing system list
    void PrintList()
    {
        std::cout << "Computing system ids: [";
        for(auto& id : computerSystemIds)
            std::cout << id << " ";
        std::cout << "]" << std::endl;
    }
    
    /// @brief 4 Print computing system details
    void PrintDetails()
    {
        std::cout << "PrintDetails()" << std::endl;
        int id = ConsoleHelper::GetIntFromUser("Enter computing system id: ");

        if(!IsExists(id))
        {
            std::cout << "Not found!" << std::endl;
            return;
        }

        ComputingSystem computingSystem = GetComputingSystem(id);
        computingSystem.Print();
    }

    /// @brief 5 Add computing system
    void Add()
    {
        std::cout << "Add()" << std::endl;
        ComputingSystem computingSystem = ComputingSystem::GetDataFromUser();        

        if(TryAddComputingSystem(computingSystem))
        {
            std::cout << "Computing system " << computingSystem.GetId() << " added." << std::endl;
        }
        else
        {
            std::cout << "Error in adding computing system " << computingSystem.GetId() << "!" << std::endl;
        }
    }

    /// @brief 6 Change computing system
    void Change()
    {
        std::cout << "Change()" << std::endl;
    }

    /// @brief 7 Remove computing system
    void Remove()
    {
        std::cout << "Remove()" << std::endl;
    }

    /// @brief 8 Is computing system exists
    void IsExists()
    {
        int compSystemId = ConsoleHelper::GetIntFromUser("Enter computing system id: ", "Error! Enter integer number!");                
        bool isExists = IsExists(compSystemId);

        std::cout << "id: "       << compSystemId << "; ";
        std::cout << "isExists: " << isExists     << std::endl;
    }

};
//////////////// Вычислительная система (конец) //////////////



/////////////////////////////////////////////////

/// @brief Группы задач
enum class TaskGroup
{
    None,
    Vector,
    VecVec,
    Matrix,
    MatVec,
    VecMat,
    MatMat
};

/// @brief Задача
enum class Task
{
    None,
    Init,// Инициализация
    Copy,// Копирование
    Sum, // Суммирование
    Min, // Минимум
    Max  // Максимум
};

/// @brief Рузультаты тестового запуска алгоритма
struct AlgTestingResult
{
    // УИД тестового запуска
    size_t id = 0;
    // УИД вычислительной системы
    unsigned compSystemId = 0;
    // УИД группы задач (вектор, вектор-матрица и пр) | TaskGroup
    unsigned taskGroupId = 0;
    // УИД задачи (сумма элементов вектора, скалярное произведение векторов и пр) | Task
    unsigned taskId = 0;
    // УИД алгоритма
    unsigned algorithmId = 0;
    // Длина типа данных, используемая в алгоритме (float: 4; double: 8)
    unsigned algorithmDataTypeLength = 0;
    // Тип алгоритма:
    // 1 - последовательный CPU
    // 2 - последовательный GPU
    // 3 - параллельный CPU std::thread
    // 4 - параллельный CPU OpenMP
    // 5 - параллельный CUDA
    unsigned algorithmType = 0;
    // Количество потоков CPU
    unsigned threadsNumCpu = 0;
    // Количество блоков GPU
    unsigned threadBlocksNumGpu = 0;
    // Количество нитей GPU в блоке
    unsigned threadsNumGpu = 0;
    // Статистики вычислительного эксперимента
    CalculationStatistics calculationStatistics;
};


/// @brief Репозиторий результатов тестовых запусков алгоритмов
class AlgTestingResultRepository
{
    std::string dir_name = "AlgTestingResultRepository";// Каталог с данными
    std::string file_name = "data.txt";  // Файл с данными
    std::vector<AlgTestingResult> cache; // Кэш данных в памяти
    // Ключ - compSystemId;
    // значение - вектор индексов УИД тестовых запусков
    // вычислительной системы compSystemId
    std::map<unsigned, std::vector<size_t>> compSystemIndex;

    /// @brief Проверка существования каталогов
    void CheckDirectories()
    {        
        if(!FileSystemHelper::IsDirExists(dir_name))
            FileSystemHelper::CreateDir(dir_name);
    }

    public:
    AlgTestingResultRepository()
    {
        CheckDirectories();
    }

    bool Write(AlgTestingResult& data)
    {
        std::string filePath = FileSystemHelper::CombinePath(dir_name, "1.txt");
        std::ofstream fout(filePath, std::ios::app);
        fout << data.id << " "
             << data.compSystemId << " "
             << data.taskGroupId << " "
             << data.taskId << " "
             << data.algorithmId << " "
             << data.algorithmDataTypeLength << " "
             << data.algorithmType << " "
             << data.threadsNumCpu << " "
             << data.threadBlocksNumGpu << " "
             << data.threadsNumGpu << " "
             << data.calculationStatistics.minValue << " "
             << data.calculationStatistics.median << " "
             << data.calculationStatistics.avg << " "
             << data.calculationStatistics.percentile_95 << " "
             << data.calculationStatistics.maxValue << " "
             << data.calculationStatistics.stdDev << " "
             << data.calculationStatistics.numIter << " "
             << "\n";
        fout.close();

        return true;
    }

    /// @brief Запуск по команде меню
    void Write()
    {
        AlgTestingResult res;
        res.id = 111;
        res.compSystemId = 222;

        Write(res);
    }
};

/////////////////////////////////////////////////



//////////////// Конфигурация приложения (начало) //////////////

/// @brief Конфигурация приложения
class AppConfig
{
    bool isInitialized{true};// Статус инициализации объекта конфигурации приложения
    std::string message{"AppConfig status: OK"};// Строка с описанием статуса инициализации объекта конфигурации приложения

    std::string fileConfig{"config.txt"};// Имя файла конфигурации

    int compSystemId {1};// Идентификатор вычислительной системы
    std::string dir_calcTestResults{"CalcTestResults"}; // Каталог с данными о результатах вычислительных экспериментов
    std::string dir_computingSystemRepository{"ComputingSystemRepository"}; // Каталог с данными о вычислительных системах

    /// @brief Проверка существования каталогов
    void CheckDirectories()
    {        
        if(!FileSystemHelper::IsDirExists(dir_calcTestResults))
            FileSystemHelper::CreateDir(dir_calcTestResults);

        if(!FileSystemHelper::IsDirExists(dir_computingSystemRepository))
            FileSystemHelper::CreateDir(dir_computingSystemRepository);
    }

    /// @brief Считывает конфигурацию из файла
    /// @return true - успех; false - наличие ошибок считывания
    bool ReadConfigFile()
    {
        std::ifstream f(fileConfig);
        
        if(!f.is_open())
        {            
            message = "Config file \"" + fileConfig + "\" is not opened!";
            return false;
        }

        // Проверка формата файла
        std::string str;
        f >> str;
        if (str != "AppConfig")
        {            
            message = "Config file \"" + fileConfig + "\" format is not AppConfig!";
            return false;
        }

        // Считываем пары "Параметр Значение"
        std::string param, value;
        while(f >> param >> value)
        {
            //std::cout << param << ": " << value << std::endl;
            if(param == "compSystemId")
            {
                try
                {
                    compSystemId = std::stoi(value);
                    //std::cout << "!!! " << compSystemId << std::endl;
                }
                catch(const std::exception& e)
                {                    
                    message = "Config file \"" + fileConfig + "\": compSystemId parameter is not recognized!";
                    return false;
                }                
            }
            else if (param == "dir_calcTestResults")
                dir_calcTestResults = value;
            else if (param == "dir_computingSystemRepository")
                dir_computingSystemRepository = value;
            else
            {
                message = "Config file \"" + fileConfig + "\": parameter \"" + param + "\" with value \"" + value + "\" is not recognized!";
                return false;
            }
        }

        return true;
    }

public:
    AppConfig()
    {
        // Проверка существования каталогов
        CheckDirectories();
    }

    AppConfig(std::string fileName)
    {
        if(!FileSystemHelper::IsFileExists(fileName))
        {
            isInitialized = false;
            message = "Error! Config file \"" + fileName + "\" not found!";
            return;
        }
        fileConfig = fileName;

        bool result = ReadConfigFile();
        if(!result)
        {
            isInitialized = false;            
            return;
        }

        // Проверка существования каталогов
        CheckDirectories();
    }

    bool IsInitialized() const
    {
        return isInitialized;
    }

    std::string GetMessage() const
    {
        return message;
    }

    std::string GetDirComputingSystemRepository() const
    {
        return dir_computingSystemRepository;
    }

    void Print()
    {
        if(!isInitialized)
        {
            std::cout   << "AppConfig: ["
                        << "NOT INITIALIZED; "
                        << message
                        << "]" << std::endl;
            return;
        }

        std::cout   << "AppConfig: ["
                    << "compSystemId: " << compSystemId << "; "
                    << "dir_calcTestResults: " << dir_calcTestResults << "; "
                    << "dir_computingSystemRepository: " << dir_computingSystemRepository
                    << "]" << std::endl;
    }

};
//////////////// Конфигурация приложения (конец) //////////////

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

/// @brief Элемент меню
struct MenuCommandItem
{
    MenuCommand comm = MenuCommand::None;// Команда
    std::vector<std::string> keys;// Список ключей
    std::function<void()> func;// Вызываемая функция
    std::string desc;// Описание команды

    MenuCommandItem()
    {}

    MenuCommandItem(MenuCommand comm,
        std::vector<std::string> keys,
        std::function<void()> func,
        std::string desc)
            : comm(comm), keys(keys), func(func), desc(desc)
    {}

    void Reset()
    {
        comm = MenuCommand::None;
        keys = {};
        func = nullptr;
        desc = "Command not choosed!";
    }

    bool CheckKey(const std::string& str)
    {
        bool isKey = false;
        for(auto& key : keys)
        {
            if(key == str)
            {
                isKey = true;
                break;
            }
        }
        return isKey;
    }
};



/// @brief Функции меню
struct MenuFunctions
{    
    /// @brief Выводит параметры GPU
    static void PrintGpuParameters()
    {
        CudaHelper::PrintCudaDeviceProperties();
    }

    /// @brief Выводит в консоль список поддерживаемых библиотек
    static void PrintLibSupport()
    {
        // Определяем перечень поддерживаемых библиотек
        LibSupport support;
        support.Print();// Выводим список поддерживаемых библиотек
    }

    /// @brief Тестирование функций класса ArrayHelper
    static void Testing_TestArrayHelper()
    {
        if(TestArrayHelper())
            std::cout << "TestArrayHelper correct!" << std::endl;
        else
            std::cout << "TestArrayHelper not correct!" << std::endl;
    }

    /// @brief Записывает параметры видеокарт в текстовый файл gpu-specs.txt
    static void WriteGpuSpecsToTxtFile()
    {
        int cudaDeviceNumber = CudaHelper::GetCudaDeviceNumber();
        std::cout << "Cuda devices number: " << cudaDeviceNumber << std::endl;
        //CudaHelper::PrintCudaDeviceProperties();

        if(cudaDeviceNumber > 0)
        {
            for(int i = 0; i < cudaDeviceNumber; i++)
            {
                auto devProps = CudaHelper::GetCudaDeviceProperties();
                devProps.Print();
            }
            
            std::ofstream f("gpu-specs.txt");
            CudaHelper::WriteGpuSpecs(f);
            f.close();
        }
    }

    /// @brief Запускает тест работоспособности VectorGpu
    static void Testing_TestVectorGpu()
    {
        // Запускаем тест работоспособности VectorGpu
        if(TestVectorGpu())
            std::cout << "VectorGpu correct!" << std::endl;
        else
            std::cout << "VectorGpu not correct!" << std::endl;
    }

    /// @brief Запускает функцию тестирования суммирования массивов
    static void Testing_TestSum()
    {
        // Запускаем функцию тестирования суммирования массивов
        if(TestSum())
            std::cout << "TestSum correct!" << std::endl;
        else
            std::cout << "TestSum not correct!" << std::endl;
    }
    
    /// @brief Конфигурирование приложения
    static void Application_Config(AppConfig& config)
    {
        std::cout   << "----- Application configuration -----\n"
                    << "1 Back to main menu\n"
                    << "2 Print config" << std::endl;

        
        int command = 0;
        while(command != 1)
        {
            std::cout << ">> ";
            std::string commandString;
            std::cin >> commandString;
            
            try
            {
                command = std::stoi(commandString);
            }
            catch(const std::exception& e)
            {
                command = 0;
            }
                        
            switch (command)
            {
            case 1:
                std::cout << "Back to main menu" << std::endl;
                break;
            case 2:
                config.Print();
                break;
            
            default:
                std::cout << "Command not recognized!" << std::endl;
                break;
            }
        }
    
    }

    /// @brief Конфигурирование приложения
    static void ComputingSystemRepository_Config(ComputingSystemRepository& repo)
    {
        std::cout   << "----- Computing system repository configuration -----\n"
                    << "1 Back to main menu\n"
                    << "2 Print config\n"
                    << "3 Print computing system list\n"
                    << "4 Print computing system details\n"
                    << "5 Add computing system\n"
                    << "6 Change computing system\n"
                    << "7 Remove computing system\n"
                    << "8 Is computing system exists\n";

        int command = 0;
        while(command != 1)
        {
            std::cout << ">> ";
            std::string commandString;
            std::cin >> commandString;
            
            try
            {
                command = std::stoi(commandString);
            }
            catch(const std::exception& e)
            {
                command = 0;
            }
                        
            switch (command)
            {
            case 1:
                std::cout << "Back to main menu" << std::endl;
                break;
            case 2:
                std::cout   << "Command: 2 Print config\n";
                repo.PrintConfig();
                break;
            case 3:
                std::cout   << "Command: 3 Print computing system list\n";
                repo.PrintList();
                break;
            case 4:
                std::cout   << "Command: 4 Print computing system details\n";
                repo.PrintDetails();
                break;
            case 5:
                std::cout   << "Command: 5 Add computing system\n";
                repo.Add();
                break;
            case 6:
                std::cout   << "Command: 6 Change computing system\n";
                repo.Change();
                break;
            case 7:
                std::cout   << "Command: 7 Remove computing system\n";
                repo.Remove();
                break;
            case 8:
                std::cout   << "Command: 8 Is computing system exists\n";
                repo.IsExists();
                break;
            default:
                std::cout << "Command not recognized!" << std::endl;
                break;
            }
        }
    
    }

    static void AlgTestingResultRepository_Config(AlgTestingResultRepository& repo)
    {
        std::cout   << "----- AlgTestingResultRepository configuration -----\n"
                    << "1 Back to main menu\n"
                    << "2 Print config\n"
                    << "3 Print AlgTestingResultRepository list\n"
                    << "4 Print AlgTestingResultRepository details\n"
                    << "5 Add test alg result data\n"
                    << "6 Change AlgTestingResultRepository\n"
                    << "7 Remove AlgTestingResultRepository\n"
                    << "8 Is AlgTestingResultRepository exists\n";

        int command = 0;
        while(command != 1)
        {
            std::cout << ">> ";
            std::string commandString;
            std::cin >> commandString;
            
            try
            {
                command = std::stoi(commandString);
            }
            catch(const std::exception& e)
            {
                command = 0;
            }
                        
            switch (command)
            {
            case 1:
                std::cout << "Back to main menu" << std::endl;
                break;
            case 2:
                std::cout   << "Command: 2 Print config\n";
                //repo.PrintConfig();
                break;
            case 3:
                std::cout   << "Command: 3 Print computing system list\n";
                //repo.PrintList();
                break;
            case 4:
                std::cout   << "Command: 4 Print computing system details\n";
                //repo.PrintDetails();
                break;
            case 5:
                std::cout   << "Command: 5 Add test alg result data\n";
                repo.Write();
                break;
            case 6:
                std::cout   << "Command: 6 Change computing system\n";
                //repo.Change();
                break;
            case 7:
                std::cout   << "Command: 7 Remove computing system\n";
                //repo.Remove();
                break;
            case 8:
                std::cout   << "Command: 8 Is computing system exists\n";
                //repo.IsExists();
                break;
            default:
                std::cout << "Command not recognized!" << std::endl;
                break;
            }
        }
    }

    // Тестирование функциональности класса FileSystemHelper
    static void Testing_FileSystemHelper()
    {
        std::cout   << "----- FileSystemHelper -----\n"
                    << "1 Back to main menu\n"
                    << "2 CombinePath\n"
                    << "3 CreateFile\n"
                    << "4 IsFileExists\n"
                    << "5 CreateDir\n"
                    << "6 IsDirExists\n"
                    << "7 RemoveFile\n"
                    << "8 RemoveDir\n";

        int command = 0;
        while(command != 1)
        {
            std::cout << ">> ";
            std::string commandString;
            std::cin >> commandString;
            
            try
            {
                command = std::stoi(commandString);
            }
            catch(const std::exception& e)
            {
                command = 0;
            }
                        
            switch (command)
            {
            case 1:
                std::cout << "Back to main menu" << std::endl;
                break;
            case 2:
                std::cout   << "Command: 2 CombinePath\n";
                FileSystemHelper::CombinePath();
                break;
            case 3:
                std::cout   << "Command: 3 CreateFile\n";
                FileSystemHelper::CreateFile();
                break;
            case 4:
                std::cout   << "Command: 4 IsFileExists\n";
                FileSystemHelper::IsFileExists();
                break;
            case 5:
                std::cout   << "Command: 5 CreateDir\n";
                FileSystemHelper::CreateDir();                
                break;
            case 6:
                std::cout   << "Command: 6 IsDirExists\n";
                FileSystemHelper::IsDirExists();
                break;
            case 7:
                std::cout   << "Command: 7 RemoveFile\n";
                FileSystemHelper::RemoveFile();
                break;
            case 8:
                std::cout   << "Command: 8 RemoveDir\n";
                FileSystemHelper::RemoveDir();
                break;            
            default:
                std::cout << "Command not recognized!" << std::endl;
                break;
            }
        }
    }
};


/// @brief Главное меню приложения
class MainMenu
{
    // Список команд меню
    std::vector<MenuCommandItem> menuCommands;

    //MenuCommand command = MenuCommand::None;// Выбранная команда меню
    MenuCommandItem command;// Выбранная команда меню
    
    /// @brief Распознаёт команду
    /// @param commandString 
    /// @return 
    bool RecognizeCommand(std::string commandString)
    {        
        command.Reset();
        for(auto& menuItem : menuCommands)
        {
            if(menuItem.CheckKey(commandString))
            {
                command = menuItem;
                return true;
            }
        }

        return false;
    }

    /// @brief Исполняет команду
    void RunCommand()
    {
        if(command.func == nullptr)        
            return;
        
        std::cout << "----- Starting: " << command.desc << "-----------" << std::endl;
        command.func();
        std::cout << "-------------------------------------" << std::endl;
    }

    /// @brief Выводит в консоль справочную информацию
    void PrintHelp()
    {
        std::cout << "----- Command list -----" << std::endl;
        for(auto& menuItem : menuCommands)
        {            
            for(auto& key : menuItem.keys)
            {
                std::cout << key << " ";
            }
            std::cout << "\t" << menuItem.desc << std::endl;
        }
    }

public:

    MainMenu()
    {
        // Инициализация меню
        MenuCommandItem item1;
        item1.comm = MenuCommand::Help;
        item1.keys = {"1","?","h","help"};
        item1.func = nullptr;
        item1.desc = "Print help";
        menuCommands.push_back(item1);

        MenuCommandItem item2;
        item2.comm = MenuCommand::Exit;
        item2.keys = {"2","q","exit"};
        item2.func = nullptr;
        item2.desc = "Exit from menu";
        menuCommands.push_back(item2);
        
        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::PrintLibSupport,
                {"3","libs"},
                MenuFunctions::PrintLibSupport,
                "Print supported libs (OpenMP, Cuda etc.)"
            }
        );

        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::PrintGpuParameters,
                {"4","gpu"},
                MenuFunctions::PrintGpuParameters,
                "Print default (0) Cuda-device properties"
            }
        );

        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::WriteGpuSpecsToTxtFile,
                {"5","gpu"},
                MenuFunctions::WriteGpuSpecsToTxtFile,
                "Write GPU specification to txt file gpu-specs.txt"
            }
        );

        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::Testing_TestArrayHelper,
                {"6","test-arr-help"},
                MenuFunctions::Testing_TestArrayHelper,
                "Testing TestArrayHelper class"
            }
        );

        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::Testing_TestVectorGpu,
                {"7","test-vec-gpu"},
                MenuFunctions::Testing_TestVectorGpu,
                "Testing VectorGpu class"
            }
        );

        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::Testing_TestSum,
                {"8","test-sum"},
                MenuFunctions::Testing_TestSum,
                "Testing sum functions"
            }
        );

        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::Application_Config,
                {"9","app-conf"},
                nullptr,
                "Application configuration"
            }
        );

        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::ComputingSystemRepository_Config,
                {"10","cs-repo-conf"},
                nullptr,
                "Computing system repository configuration"
            }
        );

        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::AlgTestingResultRepository_Config,
                {"11","algtr-repo-conf"},
                nullptr,
                "AlgTestingResultRepository configuration"
            }
        );
        
        menuCommands.push_back(
            MenuCommandItem
            {
                MenuCommand::Testing_FileSystemHelper,
                {"12","fs-hlp"},
                MenuFunctions::Testing_FileSystemHelper,
                "Testing FileSystemHelper"
            }
        );
    }

    /// @brief Запуск главного меню
    void Start(AppConfig& appConfig, ComputingSystemRepository& compSysRepo, AlgTestingResultRepository& algTestingResultRepo)
    {
        std::cout << "--- Main Menu ('1', '?', 'h' or 'help' for print help)---" << std::endl;
        std::string commandString;// Введённая пользователем команда
        
        while(command.comm != MenuCommand::Exit)
        {
            std::cout << "> ";
            std::cin >> commandString;
            if ( !RecognizeCommand(commandString))// Распознаём команду
            {
                std::cout << "Error! Command not recognized! Please enter command again. '?' or 'help' for print help." << std::endl;
                continue;
            }

            switch (command.comm)
            {
            case MenuCommand::Help:
                PrintHelp();
                break;
            case MenuCommand::Application_Config:
                MenuFunctions::Application_Config(appConfig);
                break;
            case MenuCommand::ComputingSystemRepository_Config:
                MenuFunctions::ComputingSystemRepository_Config(compSysRepo);
                break;
            case MenuCommand::AlgTestingResultRepository_Config:
                MenuFunctions::AlgTestingResultRepository_Config(algTestingResultRepo);
            default:
                RunCommand();
                break;
            }            
        }
        std::cout << "--- Good bye! ---" << std::endl;
    }

};


/// @brief Приложение
class Application
{
    MainMenu menu; // Главное меню
    AppConfig appConfig;// Конфигурация приложения
    ComputingSystemRepository computingSystemRepository;// Репозиторий сведений о вычислительных сстемах
    AlgTestingResultRepository algTestingResultRepository;// Репозиторий сведений о тестовых запусках различных алгоритмов

public:

    AppConfig& GetAppConfig()
    {
        return appConfig;
    }

    void Start()
    {
        // 1. Считываем конфигурацию из файла
        std::string appConfigFileName {"config.txt"};
        appConfig = AppConfig(appConfigFileName);
        if(!appConfig.IsInitialized())
        {
            std::cerr << appConfig.GetMessage() << std::endl;
            exit(-1);
        }        
        std::cout << "Application initialization: OK" << std::endl;

        // 2. Считываем сведения о вычислительной системе
        computingSystemRepository = ComputingSystemRepository {appConfig.GetDirComputingSystemRepository()};
        std::cout << "Computing system repository initialization: OK" << std::endl;

        // 3. Запускаем главное меню
        menu.Start(appConfig,
            computingSystemRepository,
            algTestingResultRepository
        );
    }
};


//////////////////////////// main ////////////////////////////
int main()
{
    std::cout << "Starting application..." << std::endl;
    Application app;
    app.Start();
}