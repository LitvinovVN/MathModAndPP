// Задача 06. Mutex
// Источник: https://medium.com/swlh/c-mutex-write-your-first-concurrent-code-69ac8b332288
// Запуск:
// g++ main.cpp -std=c++11 -pthread -o app
// nvcc main.cpp -o app
// ./app

#include <mutex>
#include <vector>
#include <queue>

class threadSafe_queue
{

    std::queue<int> rawQueue; // shared structure between all threads
    std::mutex m; // rawQueue's red door

public:

    int& retrieve_and_delete() {
        int front_value = 0; // if empty return 0
        m.lock();
        // From now on, the current thread is the only one that can access rawQueue
        if( !rawQueue.empty() ) {
            front_value = rawQueue.front();
            rawQueue.pop();
        }
        m.unlock();
        // other threads can lock the mutex now
        return front_value;
    }

    void push(int val) {
        m.lock();
        rawQueue.push(val);
        m.unlock();
    }

};

int main()
{
    std::mutex door;    // mutex declaration
    std::vector<int> v; // shared data
    door.lock();
    /*-----------------------*/
    /* This is a thread-safe zone: just one thread at the time allowed
     *
     * Unique ownership of vector v guaranteed
     */
    /*-----------------------*/
    door.unlock();

    return 1;
}