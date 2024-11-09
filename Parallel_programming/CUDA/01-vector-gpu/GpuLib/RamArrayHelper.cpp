#pragma once

#include <mutex>
#include <functional>

void alg_array_sum_double(double* data, size_t indStart, size_t length, double& result, std::mutex& mutex)
{    
    double local_res = 0;

    for (size_t i = indStart; i < indStart + length; i++)
    {        
        local_res += 1;
    }

    mutex.lock();
    result += local_res;
    mutex.unlock();
}

// ???
template<typename T = double>
void alg_array_sum(T* data, size_t indStart, size_t length, T& result, std::mutex& mutex)
{
    T local_res = 0;

    for (size_t i = indStart; i < indStart + length; i++)
    {
        local_res += data[i];
    }

    mutex.lock();
    result += local_res;
    mutex.unlock();
}

template<typename T = double>
class RamArrayHelper
{
public:
    static T Sum(T* data, size_t size, size_t threadsNum = 1)
    {
        T sum = 0;
        double sum_double = 0;     

        if (threadsNum == 1)
        {            
            for (size_t i = 0; i < size; i++)
            {
                sum += data[i];
            }
            return sum;
        }

        std::mutex m;
        std::vector<std::thread> threads;
        for (size_t i = 0; i < threadsNum; i++)
        {
            size_t blockSize = size/threadsNum;
            //std::cout << "blockSize = " << blockSize << std:: endl;
            size_t indStart = i*blockSize;            
            if(i > 0 && i == threadsNum - 1)
            {
                blockSize = size - (threadsNum-1)*blockSize;
                //std::cout << "blockSize = " << blockSize << std:: endl;
            }
            //auto boundFunction = std::bind(alg_array_sum<T>, data, indStart, blockSize, sum, m);
            //threads.push_back(std::thread t(boundFunction));
            /*threads.push_back(
                std::thread(
                    [](auto data, auto indStart, auto blockSize, auto& sum, auto& m)
                        {
                            alg_array_sum(data, indStart, blockSize, sum, m);
                        },
                        data, indStart, blockSize, std::ref(sum), std::ref(m)
                )
            );*/

            threads.push_back(std::thread(alg_array_sum_double, data, indStart, blockSize, std::ref(sum_double), std::ref(m)));
        }
        
        for(auto& th : threads)
        {
            if (th.joinable())
                th.join();                
        }

        //double sum_double = 0;
        //std::thread t1(alg_array_sum_double, data, 0, size, std::ref(sum_double), std::ref(m));
        //t1.join();
        //std::cout << "sum_double = " << sum_double << std::endl;

        //return sum;
        return sum_double;
    }
};