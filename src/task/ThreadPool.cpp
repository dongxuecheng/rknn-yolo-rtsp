#include "ThreadPool.h"
#include <iostream>

ThreadPool::ThreadPool(size_t numThreads, size_t maxQueueSize)
    : maxQueueSize(maxQueueSize), stop(false)
{
    // 启动线程池，创建 numThreads 个线程
    for (size_t i = 0; i < numThreads; ++i)
    {
        workers.emplace_back([this] {
            while (true)
            {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(queueMutex);
                    // 等待直到任务队列非空，或者线程池停止
                    condition.wait(lock, [this] {
                        return stop || !taskQueue.empty();
                    });

                    if (stop && taskQueue.empty())
                        return;

                    // 获取队列中的第一个任务
                    task = std::move(taskQueue.front());
                    taskQueue.pop();
                }

                // 执行任务
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    condition.notify_all();  // 通知所有线程退出
    for (std::thread &worker : workers)
    {
        worker.join();  // 等待每个线程完成
    }
}

bool ThreadPool::submit(std::function<void()> task)
{
    std::unique_lock<std::mutex> lock(queueMutex);

    // 如果队列已满，丢弃任务
    if (taskQueue.size() >= maxQueueSize)
    {
        return false;
    }

    taskQueue.push(std::move(task));  // 将任务添加到队列中
    condition.notify_one();  // 通知空闲线程执行任务
    return true;
}
