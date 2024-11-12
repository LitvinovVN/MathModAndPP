#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>

using namespace std::chrono;

std::mutex m;

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

template<typename T>
void thread_sum(T* data, size_t indStart, size_t indEnd, T& sum)
{
    T local_sum = 0;

    for (size_t i = indStart; i < indEnd; i++)
    {
        local_sum += data[i];
    }
    
    m.lock();
    /*std::cout << "thread " << std::this_thread::get_id()
        << "| local_sum = " << local_sum
        << std::endl;*/
    sum += local_sum;
    m.unlock();
}

template<typename T>
class Vec
{
public:
    T* data;
    size_t size;

    Vec(size_t size) : size(size)
    {
        data = new T[size];
    }

    ~Vec()
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

    T Sum()
    {
        T result = 0;
        for (size_t i = 0; i < size; i++)
        {
            result += data[i];
        }
        return result;
    }

    T Sum(unsigned threadsNum)
    {
        T sum = 0;

        std::vector<std::thread> threads;
        size_t blockSize = size / threadsNum;
        for (size_t i = 0; i < threadsNum; i++)
        {            
            size_t indStart = i * blockSize;
            size_t indEnd = indStart + blockSize;
            if(i == threadsNum - 1)
                indEnd = size;

            threads.push_back(std::thread(thread_sum<T>, data, indStart, indEnd, std::ref(sum)));
        }
        
        for(auto& th : threads)
        {
            th.join();
        }

        return sum;
    }

    FuncResult<T> Sum2()
    {
        auto start = high_resolution_clock::now();
        T result = Sum();
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);        
        auto t = duration.count();

        return FuncResult<T>(result, t);
    }

    FuncResult<T> Sum2(unsigned threadsNum)
    {
        auto start = high_resolution_clock::now();
        T result = Sum(threadsNum);
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);        
        auto t = duration.count();

        return FuncResult<T>(result, t);
    }
};

int main()
{
    size_t size = 1000000;
    double elVal = 0.001;
    Vec<double> v(size);
    v.InitByVal(elVal);
    //v.PrintToConsole();

    std::cout << "sum must be equal " << size * elVal << std::endl;
    auto sum_seq = v.Sum();
    std::cout << "sum_seq = " << sum_seq << std::endl;

    auto sum_par = v.Sum(4);
    std::cout << "sum_par = " << sum_par << std::endl;

    auto sum2 = v.Sum2();
    sum2.Print();

    auto sum2_par = v.Sum2(4);
    sum2_par.Print();
}