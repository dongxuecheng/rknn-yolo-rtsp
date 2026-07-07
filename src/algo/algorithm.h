#ifndef ALGORITHM_CONFIG_H
#define ALGORITHM_CONFIG_H

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "yolov8.h"

namespace rknn_yolo {

// 2D point for polygon
struct Point2i {
    int x = 0;
    int y = 0;
};

// 电子围栏（支持任意多边形，也支持全画面模式）
struct Fence {
    std::string id;
    bool full_frame = false;
    std::vector<Point2i> points;

    Fence() = default;
    explicit Fence(bool whole_screen) : full_frame(whole_screen) {}

    // 判断点是否在多边形内（射线法），full_frame=true 时始终返回 true
    bool contains(int x, int y) const;
    // 获取 cv::Point 列表，用于绘制
    std::vector<cv::Point> cvPoints() const;
    // 获取中心点
    cv::Point center() const;
};

// 算法关联类别配置
struct RelatedClasses {
    std::string person_model;  // 如 "yolo11"
    std::string phone_model;   // 如 "yolo11"
    std::string head_model;    // 如 "yolov5"
    std::string person_class = "person";
    std::string phone_class = "phone";
    std::string head_class = "head";
    std::string helmet_class = "helmet";
};

// 单个算法配置
struct AlgorithmConfig {
    std::string id;                  // no_helmet / phone_in_zone / person_intrusion / absence
    bool enabled = true;
    std::string algorithm_code;      // 平台接口的 classIndex
    float conf_threshold = 0.45f;
    int cooldown_seconds = 10;
    std::string nvr_ip;
    std::string channel;
    std::string fence_id;            // phone_in_zone / person_intrusion / absence 需要
    int absence_seconds = 0;         // 仅 absence
    std::string logic;               // 逻辑类型标识
    RelatedClasses related;
};

// 全局配置
struct GlobalConfig {
    std::string upload_url = "http://36.7.84.146:28801/open/api/operate/upload";
    int level_id = 1;
    std::string log_dir = "./logs";
    std::string alert_image_dir = "./alerts";
    int upload_queue_size = 10;
    int upload_timeout_seconds = 10;
};

// 算法配置集合
struct AlgorithmSettings {
    GlobalConfig global;
    std::vector<AlgorithmConfig> algorithms;
    std::map<std::string, Fence> fences;

    bool loadFromFile(const std::string &path);
    const AlgorithmConfig* getAlgorithm(const std::string &id) const;
    const Fence* getFence(const std::string &id) const;
};

// 单个模型推理结果（封装自 object_detect_result_list）
struct ModelResult {
    std::string model_name;  // yolo11 / yolov5 / yolo_world / ...
    std::vector<std::string> class_names;  // 该模型的类别名称列表，用于把 cls_id 映射成名称
    object_detect_result_list results;

    std::string getClassName(int cls_id) const {
        if (cls_id >= 0 && cls_id < (int)class_names.size()) {
            return class_names[cls_id];
        }
        return "Unknown";
    }
};

// 报警上下文
struct AlertContext {
    std::string algorithm_id;
    std::string algorithm_code;
    std::string nvr_ip;
    std::string channel;
    std::string reason;
    std::chrono::system_clock::time_point trigger_time;
};

// 用于绘制的检测框
struct DrawnBox {
    image_rect_t box;
    std::string label;
    float score;
    cv::Scalar color;
};

} // namespace rknn_yolo

#endif // ALGORITHM_CONFIG_H
