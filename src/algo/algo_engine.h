#ifndef ALGO_ENGINE_H
#define ALGO_ENGINE_H

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "algorithm.h"
#include "alert_uploader.h"

namespace rknn_yolo {

class AlgoEngine {
public:
    AlgoEngine();
    ~AlgoEngine();

    bool init(const std::string &config_path);
    void stop();

    // 每帧调用：传入所有模型结果和原始 RGB 图
    void evaluate(const std::vector<ModelResult> &all_results, const cv::Mat &origin_rgb);

private:
    AlgorithmSettings settings_;
    AlertUploader uploader_;
    std::mutex evaluate_mtx_;  // 保护算法引擎内部状态，evaluate 串行执行

    struct DetectedBox {
        image_rect_t box;
        std::string cls_name;
        float score;
        std::string model_name;
    };

    struct PersonAssociation {
        DetectedBox person;
        DetectedBox matched_head;
        DetectedBox matched_helmet;
        bool has_head = false;
        bool has_helmet = false;
    };

    bool extractBoxes(const std::vector<ModelResult> &all_results);
    void evaluateNoHelmet(const cv::Mat &origin_rgb);
    void evaluatePhoneInZone(const cv::Mat &origin_rgb);
    void evaluatePersonIntrusion(const cv::Mat &origin_rgb);
    void evaluateAbsence(const cv::Mat &origin_rgb);

    bool triggerAlert(const AlgorithmConfig &algo, const std::string &reason,
                      const cv::Mat &origin_rgb,
                      const std::vector<DrawnBox> &boxes_to_draw);
    bool isCooldownOver(const std::string &algo_id);
    static float iou(const image_rect_t &a, const image_rect_t &b);
    static bool boxContainsPoint(const image_rect_t &box, int x, int y);
    static cv::Point boxCenter(const image_rect_t &box);

    std::vector<DetectedBox> persons_;
    std::vector<DetectedBox> phones_;
    std::vector<DetectedBox> heads_;
    std::vector<DetectedBox> helmets_;

    std::map<std::string, std::chrono::system_clock::time_point> last_alert_time_;
    std::map<std::string, std::chrono::system_clock::time_point> last_person_in_fence_time_;
    std::map<std::string, bool> absence_alert_triggered_;
};

} // namespace rknn_yolo

// 线程安全地保存 OpenCV 图像（封装 cv::imwrite，避免多线程并发调用 libjpeg 崩溃）
extern std::mutex g_rknn_yolo_image_io_mutex;
bool rknn_yolo_safe_imwrite(const std::string &path, const cv::Mat &image);

#endif // ALGO_ENGINE_H
