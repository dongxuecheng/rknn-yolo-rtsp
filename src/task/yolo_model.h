#ifndef YOLO_MODEL_H
#define YOLO_MODEL_H

#include "yolov8.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include <opencv2/opencv.hpp>
#include "conf/Config.h"

// YOLO-World headers
#include "clip_text/clip_text.h"
#include "yolo_world/yolo_world_postprocess.h"

enum class ModelType {
    YOLO11,
    YOLO_WORLD,
    YOLOV8_POSE,
    YOLO26N
};

class YoloModel {
public:
    YoloModel();  // 构造函数
    ~YoloModel(); // 析构函数

    int load_model(const char* model_path, ModelType type = ModelType::YOLO11); // 加载图像模型
    int load_text_model(const char* text_model_path, const std::vector<std::string>& descriptions); // YOLO-World 特有

    int infer(cv::Mat img, object_detect_result_list &od_results); // 执行推理，接收 cv::Mat 图像
    void release(); // 释放资源
    std::string getClassName(int cls_id);
    ModelType getModelType() const { return model_type_; }

    // 设置该模型的类别名称（用于自定义模型，替代全局Config的classes）
    void setClassNames(const std::vector<std::string>& names) { class_names_ = names; }
    // 设置该模型的独立阈值
    void setThresholds(float obj_thresh, float nms_thresh) { object_threshold_ = obj_thresh; nms_threshold_ = nms_thresh; }

private:
    ModelType model_type_ = ModelType::YOLO11;
    rknn_app_context_t rknn_app_ctx_;  // RKNN上下文（图像模型共用）

    // YOLO-World 特有成员
    rknn_clip_context* clip_ctx_ = nullptr;
    std::vector<float> text_embeddings_;
    std::vector<std::string> class_names_;
    bool text_model_loaded_ = false;

    // 每模型独立阈值（<=0 时使用全局Config默认值）
    float object_threshold_ = -1.0f;
    float nms_threshold_ = -1.0f;

    int convert_to_image_buffer(cv::Mat img, image_buffer_t &img_buffer);
    int infer_yolo_world(cv::Mat img, object_detect_result_list &od_results);
    int infer_yolov8_pose(cv::Mat img, object_detect_result_list &od_results);
    int infer_yolo26(cv::Mat img, object_detect_result_list &od_results);
};

#endif // YOLO_MODEL_H
