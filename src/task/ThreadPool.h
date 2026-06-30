#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

class ThreadPool
{
public:
    // 构造函数：numThreads 线程数，maxQueueSize 任务队列的最大长度
    ThreadPool(size_t numThreads, size_t maxQueueSize);

    // 析构函数：停止线程池并等待线程结束
    ~ThreadPool();

    // 提交任务到线程池，队列满时返回 false
    bool submit(std::function<void()> task);

private:
    // 工作线程的列表
    std::vector<std::thread> workers;
    // 任务队列
    std::queue<std::function<void()>> taskQueue;

    // 互斥量，保护队列
    std::mutex queueMutex;

    // 条件变量，用于通知线程执行任务
    std::condition_variable condition;

    // 控制线程池停止的标志
    bool stop;

    // 最大任务队列长度
    size_t maxQueueSize;
};

#endif // THREAD_POOL_H
