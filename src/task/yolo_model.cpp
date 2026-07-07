#include "yolo_model.h"
#include <opencv2/opencv.hpp>

#include "yolo_world/yolo_world.h"
#include "yolo_world/yolo_world_postprocess.h"
#include "yolo_world/yolo_world.h"
#include "yolov8_pose/yolov8_pose.h"
#include "yolov8_pose/yolov8_pose_postprocess.h"
#include "yolo26/yolo26.h"
#include "yolo26/yolo26_postprocess.h"
#include "yolov5/yolov5.h"
#include "yolov5/yolov5_postprocess.h"
#include "clip_text/clip_text.h"

YoloModel::YoloModel()
{
    memset(&rknn_app_ctx_, 0, sizeof(rknn_app_context_t));
}

YoloModel::~YoloModel()
{
    release();
}

int YoloModel::load_model(const char *model_path, ModelType type)
{
    int ret;
    model_type_ = type;

    if (model_type_ == ModelType::YOLO11)
    {
        ret = init_yolov8_model(model_path, &rknn_app_ctx_);
        if (ret != 0)
        {
            printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
            return ret;
        }
    }
    else if (model_type_ == ModelType::YOLO_WORLD)
    {
        ret = init_yolo_world_model(&rknn_app_ctx_, model_path);
        if (ret != 0)
        {
            printf("init_yolo_world_model fail! ret=%d model_path=%s\n", ret, model_path);
            return ret;
        }
        yolo_world_init_post_process();
    }
    else if (model_type_ == ModelType::YOLOV8_POSE)
    {
        ret = init_yolov8_pose_model(model_path, &rknn_app_ctx_);
        if (ret != 0)
        {
            printf("init_yolov8_pose_model fail! ret=%d model_path=%s\n", ret, model_path);
            return ret;
        }
        yolov8_pose_init_post_process();
    }
    else if (model_type_ == ModelType::YOLO26)
    {
        ret = init_yolo26_model(model_path, &rknn_app_ctx_);
        if (ret != 0)
        {
            printf("init_yolo26_model fail! ret=%d model_path=%s\n", ret, model_path);
            return ret;
        }
    }
    else if (model_type_ == ModelType::YOLOV5)
    {
        ret = init_yolov5_model(model_path, &rknn_app_ctx_);
        if (ret != 0)
        {
            printf("init_yolov5_model fail! ret=%d model_path=%s\n", ret, model_path);
            return ret;
        }
    }
    return 0;
}

int YoloModel::load_text_model(const char *text_model_path, const std::vector<std::string> &descriptions)
{
    if (model_type_ != ModelType::YOLO_WORLD)
    {
        printf("load_text_model only used for YOLO_WORLD model\n");
        return -1;
    }

    int ret;
    if (clip_ctx_ == nullptr)
    {
        clip_ctx_ = new rknn_clip_context();
        memset(clip_ctx_, 0, sizeof(rknn_clip_context));
    }

    ret = init_clip_text_model(clip_ctx_, text_model_path);
    if (ret != 0)
    {
        printf("init_clip_text_model fail! ret=%d\n", ret);
        return ret;
    }

    // 将 std::vector<std::string> 转为 char**
    int text_num = descriptions.size();
    char **texts = (char **)malloc(text_num * sizeof(char *));
    for (int i = 0; i < text_num; i++)
    {
        texts[i] = (char *)descriptions[i].c_str();
    }

    int embed_dim = clip_ctx_->output_attrs[0].dims[1];  // 512
    int max_text_num = rknn_app_ctx_.input_attrs[1].dims[1]; // model max classes, e.g. 80
    if (max_text_num <= 0) max_text_num = 80;

    // YOLO-World model expects fixed size text input: [1, max_text_num, embed_dim]
    text_embeddings_.resize(max_text_num * embed_dim, 0.0f);

    // Only compute embeddings for user-provided descriptions
    int actual_text_num = std::min(text_num, max_text_num);
    std::vector<float> temp_embeds(actual_text_num * embed_dim);

    ret = inference_clip_text_model(clip_ctx_, texts, actual_text_num, temp_embeds.data());
    if (ret == 0)
    {
        for (int i = 0; i < actual_text_num; i++)
        {
            memcpy(text_embeddings_.data() + i * embed_dim,
                   temp_embeds.data() + i * embed_dim,
                   embed_dim * sizeof(float));
        }
    }
    if (ret != 0)
    {
        printf("inference_clip_text_model fail! ret=%d\n", ret);
        free(texts);
        return ret;
    }

    free(texts);

    class_names_ = descriptions;
    text_model_loaded_ = true;
    printf("YOLO-World text model loaded, classes=%zu\n", class_names_.size());
    return 0;
}

int YoloModel::infer(cv::Mat img, object_detect_result_list &od_results)
{
    if (rknn_app_ctx_.rknn_ctx == 0)
    {
        printf("YoloModel::infer error: rknn_ctx is invalid, model not loaded properly.\n");
        return -1;
    }
    if (model_type_ == ModelType::YOLO11)
    {
        int ret;
        image_buffer_t src_image;
        memset(&src_image, 0, sizeof(image_buffer_t));

        ret = convert_to_image_buffer(img, src_image);
        if (ret != 0)
        {
            printf("Failed to convert cv::Mat to image_buffer_t\n");
            return ret;
        }

        ret = inference_yolov8_model(&rknn_app_ctx_, &src_image, &od_results, object_threshold_, nms_threshold_);
        if (ret != 0)
        {
            printf("inference_yolov8_model fail! ret=%d\n", ret);
            free(src_image.virt_addr);
            return ret;
        }

        if (src_image.virt_addr != NULL)
        {
            free(src_image.virt_addr);
        }
        return 0;
    }
    else if (model_type_ == ModelType::YOLOV8_POSE)
    {
        return infer_yolov8_pose(img, od_results);
    }
    else if (model_type_ == ModelType::YOLO26)
    {
        return infer_yolo26(img, od_results);
    }
    else if (model_type_ == ModelType::YOLOV5)
    {
        return infer_yolov5(img, od_results);
    }
    else
    {
        return infer_yolo_world(img, od_results);
    }
}

int YoloModel::infer_yolo_world(cv::Mat img, object_detect_result_list &od_results)
{
    if (!text_model_loaded_)
    {
        printf("YOLO-World text model not loaded, call load_text_model first!\n");
        return -1;
    }

    int ret;
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));

    ret = convert_to_image_buffer(img, src_image);
    if (ret != 0)
    {
        printf("Failed to convert cv::Mat to image_buffer_t\n");
        return ret;
    }

    yolo_world_object_detect_result_list yw_results;
    memset(&yw_results, 0, sizeof(yolo_world_object_detect_result_list));

    ret = inference_yolo_world_model(&rknn_app_ctx_, &src_image,
                                     text_embeddings_.data(),
                                     (int)text_embeddings_.size(),
                                     &yw_results, object_threshold_, nms_threshold_);
    if (ret != 0)
    {
        printf("inference_yolo_world_model fail! ret=%d\n", ret);
        free(src_image.virt_addr);
        return ret;
    }

    // 转换结果类型：yolo_world -> yolo11 通用格式
    memset(&od_results, 0, sizeof(object_detect_result_list));
    int count = yw_results.count;
    if (count > OBJ_NUMB_MAX_SIZE)
    {
        count = OBJ_NUMB_MAX_SIZE;
    }
    od_results.count = count;
    for (int i = 0; i < count; i++)
    {
        od_results.results[i].box = yw_results.results[i].box;
        od_results.results[i].prop = yw_results.results[i].prop;
        od_results.results[i].cls_id = yw_results.results[i].cls_id;
    }

    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);
    }
    return 0;
}

int YoloModel::infer_yolov8_pose(cv::Mat img, object_detect_result_list &od_results)
{
    int ret;
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));

    ret = convert_to_image_buffer(img, src_image);
    if (ret != 0)
    {
        printf("Failed to convert cv::Mat to image_buffer_t\n");
        return ret;
    }

    ret = inference_yolov8_pose_model(&rknn_app_ctx_, &src_image, &od_results, object_threshold_, nms_threshold_);
    if (ret != 0)
    {
        printf("inference_yolov8_pose_model fail! ret=%d\n", ret);
        free(src_image.virt_addr);
        return ret;
    }

    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);
    }
    return 0;
}

int YoloModel::infer_yolo26(cv::Mat img, object_detect_result_list &od_results)
{
    int ret;
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));

    ret = convert_to_image_buffer(img, src_image);
    if (ret != 0)
    {
        printf("Failed to convert cv::Mat to image_buffer_t\n");
        return ret;
    }

    yolo26_object_detect_result_list y26_results;
    memset(&y26_results, 0, sizeof(yolo26_object_detect_result_list));

    ret = inference_yolo26_model(&rknn_app_ctx_, &src_image, &y26_results, object_threshold_, nms_threshold_);
    if (ret != 0)
    {
        printf("inference_yolo26_model fail! ret=%d\n", ret);
        free(src_image.virt_addr);
        return ret;
    }

    // 转换结果类型：yolo26 -> 通用格式
    memset(&od_results, 0, sizeof(object_detect_result_list));
    int count = y26_results.count;
    if (count > OBJ_NUMB_MAX_SIZE)
    {
        count = OBJ_NUMB_MAX_SIZE;
    }
    od_results.count = count;
    for (int i = 0; i < count; i++)
    {
        od_results.results[i].box = y26_results.results[i].box;
        od_results.results[i].prop = y26_results.results[i].prop;
        od_results.results[i].cls_id = y26_results.results[i].cls_id;
    }

    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);
    }
    return 0;
}

void YoloModel::release()
{
    if (model_type_ == ModelType::YOLO11)
    {
        release_yolov8_model(&rknn_app_ctx_);
    }
    else if (model_type_ == ModelType::YOLO_WORLD)
    {
        release_yolo_world_model(&rknn_app_ctx_);
        yolo_world_deinit_post_process();
        if (clip_ctx_ != nullptr)
        {
            release_clip_text_model(clip_ctx_);
            delete clip_ctx_;
            clip_ctx_ = nullptr;
        }
    }
    else if (model_type_ == ModelType::YOLOV8_POSE)
    {
        release_yolov8_pose_model(&rknn_app_ctx_);
        yolov8_pose_deinit_post_process();
    }
    else if (model_type_ == ModelType::YOLO26)
    {
        release_yolo26_model(&rknn_app_ctx_);
    }
    else if (model_type_ == ModelType::YOLOV5)
    {
        release_yolov5_model(&rknn_app_ctx_);
    }
}

int YoloModel::convert_to_image_buffer(cv::Mat img, image_buffer_t &img_buffer)
{
    img_buffer.width = img.cols;
    img_buffer.height = img.rows;
    img_buffer.format = IMAGE_FORMAT_RGB888;

    img_buffer.virt_addr = (unsigned char *)malloc(img.rows * img.cols * 3);
    if (img_buffer.virt_addr == nullptr)
    {
        printf("Failed to allocate memory for image buffer\n");
        return -1;
    }
    memcpy(img_buffer.virt_addr, img.data, img.rows * img.cols * 3);
    return 0;
}

std::string YoloModel::getClassName(int cls_id)
{
    if (cls_id >= 0 && cls_id < (int)class_names_.size())
    {
        return class_names_[cls_id];
    }
    else
    {
        return "Unknown";
    }
}
int YoloModel::infer_yolov5(cv::Mat img, object_detect_result_list &od_results)
{
    int ret;
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));

    ret = convert_to_image_buffer(img, src_image);
    if (ret != 0)
    {
        printf("Failed to convert cv::Mat to image_buffer_t\n");
        return ret;
    }

    yolov5_object_detect_result_list y5_results;
    memset(&y5_results, 0, sizeof(yolov5_object_detect_result_list));

    ret = inference_yolov5_model(&rknn_app_ctx_, &src_image, &y5_results, object_threshold_, nms_threshold_);
    if (ret != 0)
    {
        printf("inference_yolov5_model fail! ret=%d\n", ret);
        free(src_image.virt_addr);
        return ret;
    }

    // 转换结果类型：yolov5 -> 通用格式
    memset(&od_results, 0, sizeof(object_detect_result_list));
    int count = y5_results.count;
    if (count > OBJ_NUMB_MAX_SIZE)
    {
        count = OBJ_NUMB_MAX_SIZE;
    }
    od_results.count = count;
    for (int i = 0; i < count; i++)
    {
        od_results.results[i].box = y5_results.results[i].box;
        od_results.results[i].prop = y5_results.results[i].prop;
        od_results.results[i].cls_id = y5_results.results[i].cls_id;
    }

    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);
    }
    return 0;
}
