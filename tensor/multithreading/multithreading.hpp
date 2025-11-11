
#include <thread>

class LinkedJob{
    public:
        bool completed = false;
        int ready = 0; // negative numbers allow for multiple dependencies
        LinkedJob** Next = nullptr; // Next job

};

class MultithreadingEngine {
    public:
        std::thread* threads;
        int thread_count;
        MultithreadingEngine(int NumThreads){
            thread_count = NumThreads;
        };
        
        void start(){};
        void stop(){};

};