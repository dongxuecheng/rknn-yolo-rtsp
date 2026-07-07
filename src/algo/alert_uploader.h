#ifndef ALERT_UPLOADER_H
#define ALERT_UPLOADER_H

#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "algorithm.h"

namespace rknn_yolo {

struct UploadTask {
    std::vector<uchar> image_png;
    AlertContext context;
    std::string upload_url;
    int level_id;
    int timeout_seconds;
};

class AlertUploader {
public:
    AlertUploader();
    ~AlertUploader();

    void start(int queue_size, int timeout_seconds);
    void stop();
    // 直接接收 PNG 编码后的 buffer，避免在 uploader 线程中调用 cv::imencode
    void addAlert(const std::vector<uchar> &png_buf, const AlertContext &ctx,
                  const std::string &upload_url, int level_id);

private:
    void workerLoop();
    bool uploadOne(const UploadTask &task);

    int queue_size_ = 10;
    int timeout_seconds_ = 10;
    std::queue<UploadTask> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::thread worker_;
    std::atomic<bool> stopped_{false};
    bool started_ = false;
};

} // namespace rknn_yolo

#endif // ALERT_UPLOADER_H
