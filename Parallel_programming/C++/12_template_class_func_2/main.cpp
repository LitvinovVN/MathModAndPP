#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>
#include <functional>

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
    T _result;
    long long _time;

    FuncResult(T result, double time) : 
        _result(result), _time(time)
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
        auto start = high_resolution_clock::now();
        T result = v.Sum(indStart, indEnd);
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);        
        auto t = duration.count();

        return FuncResult<T>(result, t);
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
        auto start = high_resolution_clock::now();
        T result = v.Sum(indStart, indEnd, threadsNum);
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);        
        auto t = duration.count();

        return FuncResult<T>(result, t);
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
        std::cout << "-------LaunchSum(T v) Start ------" << std::endl;
        auto iterNum = 10;
        std::vector<FuncResult<T>> results;
        for(unsigned i{0}; i < iterNum; i++)
        {
            FuncResult<T> res = VectorRamHelper::Sum(v);
            //res.Print();
            results.push_back(res);
            //results[i].Print();
        }
        
        std::cout << "-------LaunchSum(T v) End --------" << std::endl;
        return results;
    }
};

int main()
{
    // 1. Подготовка данных
    unsigned Nthreads = 4;
    size_t size = 100000000;
    double elVal = 0.001;
    VectorRam<double> v(size);
    v.InitByVal(elVal);
    //v.PrintToConsole();
    
    // 2. Запуск тестов и получение массива результатов
    auto testResults = TestHelper::LaunchSum(v);
    std::cout << "testResults size = " << testResults.size() << std::endl;
    testResults[0].Print();
    testResults[1].Print();

    // 3. Статистическая обработка результатов
    

    std::cout << "sum must be equal " << size * elVal << std::endl;
    auto sum_seq = v.Sum();
    std::cout << "sum_seq = " << sum_seq << std::endl;

    std::cout << "sum_seq_half must be equal " << (size / 2) * elVal << std::endl;
    auto sum_seq_half = v.Sum(0, size / 2);
    std::cout << "sum_seq_half = " << sum_seq_half << std::endl;
    ///////////////////////////////////////////////////

    auto sum_par = v.Sum(Nthreads);
    std::cout << "sum_par = " << sum_par << std::endl;

    auto sum_par_half = v.Sum(0, size / 2, Nthreads);
    std::cout << "sum_par_half = " << sum_par_half << std::endl;
    ///////////////////////////////////////////////////

    auto sumFR = VectorRamHelper::Sum(v);
    std::cout << "sumFR: ";
    sumFR.Print();

    auto sumFR_half = VectorRamHelper::Sum(v, 0, size / 2);
    std::cout << "sumFR_half: ";
    sumFR_half.Print();
    ///////////////////////////////////////////////////

    auto sumFR_par = VectorRamHelper::Sum(v, Nthreads);
    std::cout << "sumFR_par: ";
    sumFR_par.Print();

    auto sumFR_par_half = VectorRamHelper::Sum(v, 0, size / 2, Nthreads);
    std::cout << "sumFR_par_half: ";
    sumFR_par_half.Print();
    ///////////////////////////////////////////////////

    auto S = (double)sumFR._time / sumFR_par._time;
    std::cout << "S = " << S << std::endl;

    auto S_half = (double)sumFR_half._time / sumFR_par_half._time;
    std::cout << "S_half = " << S_half << std::endl;
    ///////////////////////////////////////////////////

    auto E = S / Nthreads;
    std::cout << "E = " << E << std::endl;

    auto E_half = S_half / Nthreads;
    std::cout << "E_half = " << E_half << std::endl;
    ///////////////////////////////////////////////////
}