#include "alert_uploader.h"
#include "algo_logger.h"
#include "algo_engine.h"
#include "httplib.h"
#include <sstream>
#include <iomanip>
#include <chrono>

namespace rknn_yolo {

AlertUploader::AlertUploader() = default;
AlertUploader::~AlertUploader() { stop(); }

void AlertUploader::start(int queue_size, int timeout_seconds) {
    queue_size_ = queue_size > 0 ? queue_size : 10;
    timeout_seconds_ = timeout_seconds > 0 ? timeout_seconds : 10;
    if (started_) return;
    stopped_ = false;
    worker_ = std::thread(&AlertUploader::workerLoop, this);
    started_ = true;
    ALGO_LOG_INFO("AlertUploader started");
}

void AlertUploader::stop() {
    if (!started_) return;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stopped_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    started_ = false;
    ALGO_LOG_INFO("AlertUploader stopped");
}

static std::string formatUploadTime(const std::chrono::system_clock::time_point &tp) {
    auto t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm;
    localtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

void AlertUploader::addAlert(const std::vector<uchar> &png_buf, const AlertContext &ctx,
                             const std::string &upload_url, int level_id) {
    if (!started_ || png_buf.empty()) {
        ALGO_LOG_WARN("Drop alert, uploader not ready or empty png: " + ctx.algorithm_id);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mtx_);
        if ((int)queue_.size() >= queue_size_) {
            queue_.pop();
            ALGO_LOG_WARN("Upload queue full, dropped oldest");
        }
        UploadTask task;
        task.image_png = png_buf;
        task.context = ctx;
        task.upload_url = upload_url;
        task.level_id = level_id;
        task.timeout_seconds = timeout_seconds_;
        queue_.push(std::move(task));
    }
    cv_.notify_one();
}

void AlertUploader::workerLoop() {
    while (!stopped_) {
        UploadTask task;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait_for(lock, std::chrono::seconds(1), [this]() { return stopped_ || !queue_.empty(); });
            if (stopped_ && queue_.empty()) break;
            if (queue_.empty()) continue;
            task = std::move(queue_.front());
            queue_.pop();
        }
        uploadOne(task);
    }
}

bool AlertUploader::uploadOne(const UploadTask &task) {
    try {
        // 将完整 URL 拆分为 base host 和 api path
        // httplib::Client 的构造函数只接受 scheme://host:port，不接受 path
        std::string base_url;
        std::string api_path;
        {
            const std::string &url = task.upload_url;
            size_t scheme_end = url.find("://");
            if (scheme_end == std::string::npos) {
                ALGO_LOG_ERROR("Upload invalid url (no scheme): " + url);
                return false;
            }
            size_t host_start = scheme_end + 3;
            size_t path_start = url.find('/', host_start);
            if (path_start == std::string::npos) {
                base_url = url;
                api_path = "/";
            } else {
                base_url = url.substr(0, path_start);
                api_path = url.substr(path_start);
            }
        }

        httplib::Client cli(base_url.c_str());
        cli.set_connection_timeout(task.timeout_seconds, 0);
        cli.set_read_timeout(task.timeout_seconds, 0);
        cli.set_follow_location(true);

        // Build query string manually because httplib has no Post(path, params, items)
        std::string path = api_path + "?";
        auto append = [&path, &api_path](const std::string &k, const std::string &v) {
            if (path.size() > api_path.size() + 1) path += "&";
            path += httplib::encode_query_component(k);
            path += "=";
            path += httplib::encode_query_component(v);
        };
        append("channel", task.context.channel);
        append("classIndex", task.context.algorithm_code);
        append("ip", task.context.nvr_ip);
        append("videoTime", formatUploadTime(task.context.trigger_time));
        append("levelId", std::to_string(task.level_id));

        httplib::UploadFormDataItems items;
        std::string img_data(task.image_png.begin(), task.image_png.end());
        items.push_back({"file", img_data, "violation.png", "image/png"});

        ALGO_LOG_INFO("Uploading " + task.context.algorithm_id + " code=" + task.context.algorithm_code + " url=" + base_url + path);
        auto res = cli.Post(path.c_str(), items);
        if (!res) {
            ALGO_LOG_ERROR("Upload no response: " + task.context.algorithm_id);
            return false;
        }
        if (res->status != 200) {
            ALGO_LOG_ERROR("Upload HTTP " + std::to_string(res->status) + ": " + task.context.algorithm_id + " body=" + res->body);
            return false;
        }
        ALGO_LOG_INFO("Upload success: " + task.context.algorithm_id + " body=" + res->body);
        return true;
    } catch (const std::exception &e) {
        ALGO_LOG_ERROR(std::string("Upload exception: ") + e.what());
        return false;
    }
}

} // namespace rknn_yolo
