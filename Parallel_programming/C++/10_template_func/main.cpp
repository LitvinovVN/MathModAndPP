#include <iostream>
#include <thread>

template<typename T>
void func(T val)
{
    std::cout << val << std::endl;
}

template<typename T>
void func2(T& val)
{
    val *= 2;
}

template<typename T>
void func3(T* arr, unsigned size)
{
    for(unsigned i = 0; i < size; i++)
    {
        arr[i] *= 2;
    }
}

int main()
{
    std::thread t1(func<double>, 3.45);
    t1.join();

    double tmp1 = 1.25;
    std::cout << "tmp1 = " << tmp1 << std::endl;
    std::thread t2(func2<double>, std::ref(tmp1));
    t2.join();
    std::cout << "tmp1 = " << tmp1 << std::endl;

    unsigned N = 10;
    double* arr = new double[N];
    for (size_t i = 0; i < N; i++)
    {
        arr[i] = 0.1*i;
    }
    for (size_t i = 0; i < N; i++) std::cout << arr[i] << " ";
    std::cout << std::endl;

    std::thread t3(func3<double>, arr, N);
    t3.join();
    for (size_t i = 0; i < N; i++) std::cout << arr[i] << " ";
    std::cout << std::endl;
}