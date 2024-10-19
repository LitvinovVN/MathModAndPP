// nvcc main.cu -o app
// ./app

#include <iostream>

/// @brief Вектор (в GPU) 
/// @tparam T Тип элементов вектора
template<typename T = double>
class VectorGpu
{
    size_t _size;
    T* _dev_data;

public:
    VectorGpu(size_t size) : _size(size)
    {
        cudaError_t cudaResult = cudaMalloc(&_dev_data, size*sizeof(T));
        if (cudaResult != cudaSuccess)
        {
            std::string msg("Could not allocate device memory for VectorGpu: ");
            msg += cudaGetErrorString(cudaResult);
            throw std::runtime_error(msg);
        }

        std::cout << "Device memory for VectorGpu allocated!\n";
    }

    ~VectorGpu()
    {
        cudaFree(_dev_data);
        std::cout << "Device memory for VectorGpu cleared!\n";
    }
    

    /// @brief Возвращает указатель на данные в видеопамяти
    /// @return 
    __host__ __device__
    T* get_dev_data_pointer()
    {
        return _dev_data;
    }

    __host__ __device__
    size_t getSize() const
    {
        return _size;
    }
    
    void initVectorByRange(double start, double end)
    {
        // Создаём временный массив
        T* tmp = new T[_size];
        size_t cnt = 0;

        // Инициализируем временный массив
        auto step = (end-start)/(_size-1);
        for (auto i = start; i < end+step/2; i+=step)
        {
            tmp[cnt++] = i;
            std::cout << tmp[cnt-1] << " ";
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

        std::cout << "cudaMemCpy OK!\n";

        // Освобождаем временный массив
        delete[] tmp;
    }

};

template<typename T>
__global__
void kernel_vector(VectorGpu<T> vectorGpu)
{
    int th_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_i == 0)
    {
        printf("GPU: vectorGpu._size = %d\n", vectorGpu.getSize());
        T* _dev_data_pointer = vectorGpu.get_dev_data_pointer();
        for(size_t i=0; i<vectorGpu.getSize(); i++)
        {
            printf("%lf ", _dev_data_pointer[i]);
        }
    }
}

////////////////////////////////
int main()
{
    VectorGpu<> v1(10);
    v1.initVectorByRange(0.1,0.5);
    //v1.print();

    kernel_vector<double><<<2,2>>>(v1);
    cudaDeviceSynchronize();
}