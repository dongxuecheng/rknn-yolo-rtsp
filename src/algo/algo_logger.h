#ifndef ALGO_LOGGER_H
#define ALGO_LOGGER_H

#include <string>
#include <string>
#include <mutex>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace rknn_yolo {

class AlgoLogger {
public:
    static AlgoLogger& getInstance();
    void init(const std::string &log_dir);
    void info(const std::string &msg);
    void error(const std::string &msg);
    void warning(const std::string &msg);

private:
    AlgoLogger() = default;
    ~AlgoLogger();
    AlgoLogger(const AlgoLogger&) = delete;
    AlgoLogger& operator=(const AlgoLogger&) = delete;

    void ensureOpen();
    std::string currentDateStr() const;
    std::string currentTimeStr() const;

    std::string log_dir_;
    std::string current_date_;
    std::ofstream ofs_;
    std::mutex mtx_;
    bool initialized_ = false;
};

#define ALGO_LOG_INFO(msg)  rknn_yolo::AlgoLogger::getInstance().info(msg)
#define ALGO_LOG_ERROR(msg) rknn_yolo::AlgoLogger::getInstance().error(msg)
#define ALGO_LOG_WARN(msg)  rknn_yolo::AlgoLogger::getInstance().warning(msg)

} // namespace rknn_yolo

#endif // ALGO_LOGGER_H
