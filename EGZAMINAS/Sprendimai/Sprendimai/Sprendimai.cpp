#include <iostream>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <stack>

class DataMonitor {
public:
    std::mutex mtx;
    std::condition_variable cv;
    bool end = false;
    std::stack<int> dataStack; // Stack to store values

    DataMonitor() {}

    void PushToStack(int value) {
        std::unique_lock<std::mutex> lock(mtx);
        dataStack.push(value);
        cv.notify_one();
    }

    int PopFromStack() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return !dataStack.empty(); });

        int value = dataStack.top();
        dataStack.pop();
        return value;
    }
};

void PrintThreadId(int id) {
    for (int i = 0; i < 5; ++i) {
        std::cout << "Thread " << id << ": " << std::this_thread::get_id() << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

int main() {
    const int thread_count = 11; // 10 worker threads and 1 background thread
    std::thread threads[thread_count];
    int count[thread_count] = { 0 };
    DataMonitor dm;

    // Worker threads
    for (int i = 0; i < thread_count - 1; i++) {
        threads[i] = std::thread([i, &dm, &count]() {
            PrintThreadId(i);
            while (!dm.end) {
                int value = i * 5 + count[i];
                dm.PushToStack(value);

                count[i]++;

                if (count[i] == 5) {
                    dm.end = true;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            });
    }

    // Background thread
    threads[thread_count - 1] = std::thread([&dm]() {
        for (int i = 0; i < 10; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (!dm.dataStack.empty()) {
                int max = dm.PopFromStack();
                std::cout << "Background Thread: Max value from stack: " << max << std::endl;
            }
        }
        });

    // Join all threads
    for (int i = 0; i < thread_count; i++) {
        threads[i].join();
    }

    return 0;
}
