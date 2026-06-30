// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#ifndef _RKNN_DEMO_YOLO_WORLD_H_
#define _RKNN_DEMO_YOLO_WORLD_H_

#include "rknn_api.h"
#include "common.h"
#include "yolov8.h"  // 复用 yolov8 中定义的 rknn_app_context_t
#include "yolo_world_postprocess.h"


int init_yolo_world_model(rknn_app_context_t* app_ctx, const char* model_path);

int release_yolo_world_model(rknn_app_context_t* app_ctx);

int inference_yolo_world_model(rknn_app_context_t* app_ctx, image_buffer_t* img, float* text_input, int text_size, yolo_world_object_detect_result_list* od_results, float conf_threshold = -1.0f, float nms_threshold = -1.0f);

// 为 yolo_world.cc 提供 NMS/BOX 阈值宏兼容
#define YOLO_WORLD_NMS_THRESH 0.45
#define YOLO_WORLD_BOX_THRESH 0.25

#endif //_RKNN_DEMO_YOLO_WORLD_H_