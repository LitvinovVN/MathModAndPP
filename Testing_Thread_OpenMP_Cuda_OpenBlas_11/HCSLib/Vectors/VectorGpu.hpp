#pragma once

/// @brief Вектор (в GPU) 
/// @tparam T Тип элементов вектора
template<typename T>
class VectorGpu : public IVector<T>
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
    size_t Length() const override
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
        kernel_print<T><<<1,1>>>(_dev_data, 0, _size);
        cudaDeviceSynchronize();
        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }

    /// @brief Выводит в консоль элементы вектора в заданном диапазоне
    void PrintData(unsigned long long indStart,
        unsigned long long length) const override
    {
        throw std::runtime_error("Not realized!");
    }

    /// @brief Выводит в консоль все элементы вектора
    void PrintData() const override
    {
        PrintData(0, Length());
    }

    /// @brief Возвращает значение элемента вектора, расположенного по указанному индексу
    T GetValue(unsigned long long index) const override
    {
        throw std::runtime_error("Not realized!");
    }

    /// @brief Устанавливает значение элемента вектора, расположенного по указанному индексу
    bool SetValue(unsigned long long index, T value) override
    {        
        throw std::runtime_error("Not realized!");
    }

    /// @brief Очищает массивы данных и устанавливает размер вектора в 0
    void ClearData() override
    {
        throw std::runtime_error("Not realized!");
        //delete[] data;
        //data = nullptr;
        //this->length = 0;
    }
};
