// Блокировка
// Запуск:
// nvcc kernel.cu -o app -allow-unsupported-compiler
// ./app

#include <iostream>

struct Lock
{
    int *mutex;
    Lock()
    {
        cudaMalloc( (void**)&mutex, sizeof(int) );
        cudaMemset( mutex, 0, sizeof(int) );
    }

    ~Lock()
    {
        cudaFree( mutex );
    }

    __device__
    void lock()
    {
        while( atomicCAS( mutex, 0, 1 ) != 0 );
	    __threadfence();
    }

    __device__
    void unlock()
    {
        __threadfence();
        atomicExch( mutex, 0 );
    }
};

__device__
int cnt = 0;

__global__ void cuda_hello(Lock lock)
{
    auto tid = threadIdx.x + blockIdx.x*blockDim.x;

    lock.lock();
    cnt++;
    printf("tid %d: 1 | cnt = %d \n", tid, cnt);
    lock.unlock();// Закомментировать и проанализировать вывод
    
    lock.lock();  // Закомментировать и проанализировать вывод 
    cnt--;
    printf("tid %d: 2 | cnt = %d \n", tid, cnt);
    lock.unlock();
}

int main()
{
    Lock lock;
    cuda_hello<<<30,1>>>(lock);// th > 1 - deadlock
    return 0;
}