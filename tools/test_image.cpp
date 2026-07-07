/*-------------------------------------------
        单张图片推理测试工具
        支持模型：yolo11 / yolov8_pose / yolo26 / yolov5
        用法：
            ./test_image <model_path> <image_path> <model_type> [classes] [obj_thresh] [nms_thresh]
        示例：
            ./test_image model/yolo11.rknn test.jpg yolo11 person,car 0.45 0.25
            ./test_image model/yolov8s-pose.rknn test.jpg yolov8_pose person 0.45 0.25
            ./test_image model/yolov5s.rknn test.jpg yolov5 person,car 0.45 0.25
            ./test_image model/yolo26n.rknn test.jpg yolo26 0.45 0.45
-------------------------------------------*/
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <cstring>

#include <chrono>

#include <opencv2/opencv.hpp>

#include "task/yolo_model.h"
#include "image_utils.h"

static cv::Mat convert_to_opencv_image(image_buffer_t *img_buffer)
{
    cv::Mat mat;
    cv::Mat rgb;
    switch (img_buffer->format)
    {
    case IMAGE_FORMAT_GRAY8:
        mat = cv::Mat(img_buffer->height, img_buffer->width, CV_8UC1, img_buffer->virt_addr);
        break;
    case IMAGE_FORMAT_RGB888:
        mat = cv::Mat(img_buffer->height, img_buffer->width, CV_8UC3, img_buffer->virt_addr);
        break;
    case IMAGE_FORMAT_RGBA8888:
        mat = cv::Mat(img_buffer->height, img_buffer->width, CV_8UC4, img_buffer->virt_addr);
        break;
    case IMAGE_FORMAT_YUV420SP_NV21:
    case IMAGE_FORMAT_YUV420SP_NV12:
        mat = cv::Mat(img_buffer->height + img_buffer->height / 2, img_buffer->width, CV_8UC1, img_buffer->virt_addr);
        cv::cvtColor(mat, rgb, cv::COLOR_YUV2BGR_NV21);
        mat = rgb;
        break;
    default:
        printf("Unsupported image format\n");
        return cv::Mat();
    }
    return mat;
}

static void print_usage(const char *prog)
{
    std::cout << "Usage: " << prog << " <model_path> <image_path> <model_type> <classes> [obj_thresh] [nms_thresh]" << std::endl;
    std::cout << "  model_type: yolo11 | yolo_world | yolov8_pose | yolo26 | yolov5" << std::endl;
    std::cout << "  classes:    comma separated class names, or \"coco\" for COCO 80 default" << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << prog << " model/yolo11.rknn bus.jpg yolo11 coco 0.45 0.25" << std::endl;
    std::cout << "  " << prog << " model/phone-yolo11-1280.rknn head_phone.jpg yolo11 phone 0.15 0.45" << std::endl;
}

static std::vector<std::string> split(const std::string &s, char delim)
{
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        if (!item.empty())
            tokens.push_back(item);
    }
    return tokens;
}

static std::vector<std::string> load_coco_labels()
{
    std::vector<std::string> labels;
    labels.reserve(80);
    labels = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    return labels;
}

static ModelType parse_model_type(const std::string &type_str)
{
    if (type_str == "yolo11") return ModelType::YOLO11;
    if (type_str == "yolo_world") return ModelType::YOLO_WORLD;
    if (type_str == "yolov8_pose") return ModelType::YOLOV8_POSE;
    if (type_str == "yolo26") return ModelType::YOLO26;
    if (type_str == "yolov5") return ModelType::YOLOV5;
    return ModelType::YOLO11;
}

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        print_usage(argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];
    const std::string model_type_str = argv[3];
    const std::string classes_str = argv[4];
    ModelType model_type = parse_model_type(model_type_str);

    std::vector<std::string> class_names;
    if (classes_str == "coco")
    {
        class_names = load_coco_labels();
    }
    else
    {
        class_names = split(classes_str, ',');
    }

    float obj_thresh = 0.25f;
    float nms_thresh = 0.45f;
    if (argc >= 6)
    {
        obj_thresh = std::stof(argv[5]);
    }
    if (argc >= 7)
    {
        nms_thresh = std::stof(argv[6]);
    }

    // 1. 加载图片
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(src_image));
    int ret = read_image(image_path, &src_image);
    if (ret != 0)
    {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }

    cv::Mat rgb_img = convert_to_opencv_image(&src_image);
    if (rgb_img.empty())
    {
        std::cerr << "Failed to convert image to OpenCV format." << std::endl;
        if (src_image.virt_addr) free(src_image.virt_addr);
        return -1;
    }

    // 2. 加载模型
    YoloModel model;
    std::cout << "Loading model: " << model_path << " (type=" << model_type_str << ")" << std::endl;
    ret = model.load_model(model_path, model_type);
    if (ret != 0)
    {
        std::cerr << "load_model failed, ret=" << ret << std::endl;
        if (src_image.virt_addr) free(src_image.virt_addr);
        return -1;
    }
    model.setClassNames(class_names);
    model.setNumClasses((int)class_names.size());
    model.setThresholds(obj_thresh, nms_thresh);

    // 3. 推理
    object_detect_result_list od_results;
    memset(&od_results, 0, sizeof(od_results));

    auto start = std::chrono::high_resolution_clock::now();
    ret = model.infer(rgb_img, od_results);
    auto end = std::chrono::high_resolution_clock::now();
    int infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (ret != 0)
    {
        std::cerr << "infer failed, ret=" << ret << std::endl;
        return -1;
    }

    std::cout << "Inference done in " << infer_ms << " ms, detected " << od_results.count << " objects." << std::endl;

    // 4. 打印并绘制结果（绘制到 BGR 图片用于保存）
    cv::Mat draw_img;
    if (rgb_img.channels() == 3)
    {
        cv::cvtColor(rgb_img, draw_img, cv::COLOR_RGB2BGR);
    }
    else if (rgb_img.channels() == 4)
    {
        cv::cvtColor(rgb_img, draw_img, cv::COLOR_RGBA2BGR);
    }
    else if (rgb_img.channels() == 1)
    {
        cv::cvtColor(rgb_img, draw_img, cv::COLOR_GRAY2BGR);
    }
    else
    {
        draw_img = rgb_img.clone();
    }

    for (int i = 0; i < od_results.count; i++)
    {
        object_detect_result *det = &(od_results.results[i]);
        std::string name = model.getClassName(det->cls_id);
        std::cout << "[" << i << "] " << name
                  << " @ (" << det->box.left << ", " << det->box.top
                  << ", " << det->box.right << ", " << det->box.bottom << ") "
                  << "score=" << det->prop << std::endl;

        int x1 = det->box.left;
        int y1 = det->box.top;
        int x2 = det->box.right;
        int y2 = det->box.bottom;

        cv::rectangle(draw_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);

        char text[256];
        snprintf(text, sizeof(text), "%s %.2f", name.c_str(), det->prop);
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(draw_img,
                      cv::Point(x1, y1 - text_size.height - 4),
                      cv::Point(x1 + text_size.width, y1),
                      cv::Scalar(0, 255, 0), -1);
        cv::putText(draw_img, text, cv::Point(x1, y1 - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    // 5. 保存结果
    std::string out_path = "out_test_image.png";
    cv::imwrite(out_path, draw_img);
    std::cout << "Result saved to: " << out_path << std::endl;

    model.release();
    if (src_image.virt_addr) free(src_image.virt_addr);
    return 0;
}
