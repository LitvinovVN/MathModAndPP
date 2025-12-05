#include <chrono>
#include <iostream>
#include <semaphore>
#include <thread>
#include <vector>

const unsigned numThreads{2};
const unsigned numSteps{15};

// global semaphore instances
// object counts are set to zero
// objects are in non-signaled state
std::counting_semaphore<numThreads>
    smphSignalMainToThread{0},
    smphSignalThreadToMain{0};
 
void ThreadProc()
{
    for(int i=0; i<numSteps;i++)
    {
        // wait for a signal from the main proc
        // by attempting to decrement the semaphore
        smphSignalMainToThread.acquire();
     
        // this call blocks until the semaphore's count
        // is increased from the main proc
     
        std::cout << "[thread] Calculation starting...\n"; // response message
     
        // wait to imitate some work
        using namespace std::literals;
        std::this_thread::sleep_for(100ms);
     
        std::cout << "[thread] Calulation completed!\n"; // message
     
        // signal the main proc back
        smphSignalThreadToMain.release();
    }
}

void signalToWorkers()
{
    // signal the worker threads to start working
    // by increasing the semaphore's count
    smphSignalMainToThread.release(numThreads);
}

void waitingSygnalFromWorkers()
{
    // wait until the worker threads is done doing the work
    // by attempting to decrement the semaphore's count
    for(int i=0; i<numThreads;i++)
        smphSignalThreadToMain.acquire();
        
    std::cout << "[main] Got the signals from all workers\n"; // response message
}
 
int main()
{
    std::vector<std::jthread> threads;
    threads.reserve(numThreads);
 
    for (auto id{0U}; id != numThreads; ++id)
        threads.push_back(std::jthread(ThreadProc));
    
    
    for(int s = 0; s < numSteps; s++)
    {
        std::cout << "[main] Step " << s <<"\n"; // message
        signalToWorkers();
        waitingSygnalFromWorkers();
    }
    
    
    std::cout << "[main] The end\n";
}