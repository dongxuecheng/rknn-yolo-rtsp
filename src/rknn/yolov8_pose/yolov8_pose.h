#ifndef YOLOV8_POSE_H
#define YOLOV8_POSE_H

#include "yolov8.h"

int init_yolov8_pose_model(const char* model_path, rknn_app_context_t* app_ctx);
int release_yolov8_pose_model(rknn_app_context_t* app_ctx);
int inference_yolov8_pose_model(rknn_app_context_t* app_ctx, image_buffer_t* img, object_detect_result_list* od_results, float conf_threshold = -1.0f, float nms_threshold = -1.0f);

#endif
