#include "algo_logger.h"
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>

namespace rknn_yolo {

AlgoLogger& AlgoLogger::getInstance() {
    static AlgoLogger instance;
    return instance;
}

AlgoLogger::~AlgoLogger() {
    if (ofs_.is_open()) {
        ofs_.close();
    }
}

void AlgoLogger::init(const std::string &log_dir) {
    std::lock_guard<std::mutex> lock(mtx_);
    log_dir_ = log_dir;
    if (log_dir_.empty() || log_dir_ == ".") {
        log_dir_ = "";
    }
    // mkdir recursively if log_dir is a real subdirectory
    if (!log_dir_.empty()) {
        size_t pos = 0;
        while ((pos = log_dir_.find('/', pos + 1)) != std::string::npos) {
            std::string sub = log_dir_.substr(0, pos);
            if (!sub.empty() && access(sub.c_str(), F_OK) != 0) {
                mkdir(sub.c_str(), 0777);
            }
        }
        if (access(log_dir_.c_str(), F_OK) != 0) {
            mkdir(log_dir_.c_str(), 0777);
        }
    }
    initialized_ = true;
    ensureOpen();
}

std::string AlgoLogger::currentDateStr() const {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
    localtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d");
    return oss.str();
}

std::string AlgoLogger::currentTimeStr() const {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
    localtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

void AlgoLogger::ensureOpen() {
    if (!initialized_) return;
    std::string date = currentDateStr();
    if (date != current_date_ || !ofs_.is_open()) {
        if (ofs_.is_open()) {
            ofs_.close();
        }
        current_date_ = date;
        std::string path = log_dir_.empty() ? ("algo_" + date + ".log") : (log_dir_ + "/algo_" + date + ".log");
        ofs_.open(path, std::ios::app);
        if (!ofs_.is_open()) {
            std::cerr << "[AlgoLogger] Failed to open log file: " << path << std::endl;
        }
    }
}

void AlgoLogger::info(const std::string &msg) {
    std::lock_guard<std::mutex> lock(mtx_);
    ensureOpen();
    if (ofs_.is_open()) {
        ofs_ << "[" << currentTimeStr() << "] [INFO] " << msg << std::endl;
    }
}

void AlgoLogger::error(const std::string &msg) {
    std::lock_guard<std::mutex> lock(mtx_);
    ensureOpen();
    if (ofs_.is_open()) {
        ofs_ << "[" << currentTimeStr() << "] [ERROR] " << msg << std::endl;
    }
}

void AlgoLogger::warning(const std::string &msg) {
    std::lock_guard<std::mutex> lock(mtx_);
    ensureOpen();
    if (ofs_.is_open()) {
        ofs_ << "[" << currentTimeStr() << "] [WARN] " << msg << std::endl;
    }
}

} // namespace rknn_yolo
