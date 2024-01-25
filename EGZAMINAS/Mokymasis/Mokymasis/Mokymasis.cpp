//Matas Palujanskas IFF-1/8
#include <iostream>
#include <mutex>
#include <stack>
#include <condition_variable>
#include <thread>

class DataMonitor {
public:
    std::mutex mtx;
    std::condition_variable cv;
    bool end = false;
    std::stack<int>dataStack;

    DataMonitor() {}

    void pushToStack(int value) {
        std::unique_lock<std::mutex>lock(mtx);
        dataStack.push(value);
        cv.notify_one();
    }

    int popFromStack() {
        std::unique_lock<std::mutex>lock(mtx);
        cv.wait(lock, [this] {return !dataStack.empty();});

        int value = dataStack.top();
        dataStack.pop();
        return value;
    }
};

void PrintThreadId(int id) {
    for (int i = 0; i < 5; i++) {
        std::cout << "Thread: " << id << ", his value: " << std::this_thread::get_id() << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}


int main()
{
    const int thread_count = 11;
    std::thread threads[thread_count];
    int count[thread_count] = {};
    DataMonitor dm;

    //Worker threads
    for (int i = 0; i < thread_count - 1; i++) {
        threads[i] = std::thread([i, &dm, &count]() {
            PrintThreadId(i);
            while (!dm.end) {
                int value = i * 5 + count[i];
                dm.pushToStack(value);

                count[i]++;

                if (count[i] == 5) {
                    dm.end = true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            });
    }

    //Background thread
    threads[thread_count - 1] = std::thread([&dm]() {
        for (int i = 0; i < 10; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (!dm.dataStack.empty()) {
                int max = dm.popFromStack();
                std::cout << "Max = " << max << std::endl;
            }
        }
        });

    //Join threads
    for (int i = 0; i < thread_count; i++) {
        threads[i].join();
    }

    return 0;  
}

