// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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

#ifndef _RKNN_YOLOV5_DEMO_POSTPROCESS_H_
#define _RKNN_YOLOV5_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rknn_api.h"
#include "common.h"
#include "image_utils.h"
#include "yolov8.h"  // for rknn_app_context_t

#define YOLOV5_OBJ_NAME_MAX_SIZE 64
#define YOLOV5_OBJ_NUMB_MAX_SIZE 128
#define YOLOV5_OBJ_CLASS_NUM_DEFAULT 80
#define YOLOV5_NMS_THRESH 0.45
#define YOLOV5_BOX_THRESH 0.25

// class rknn_app_context_t;

typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
} yolov5_object_detect_result;

typedef struct {
    int id;
    int count;
    yolov5_object_detect_result results[YOLOV5_OBJ_NUMB_MAX_SIZE];
} yolov5_object_detect_result_list;

int yolov5_init_post_process(int num_classes = YOLOV5_OBJ_CLASS_NUM_DEFAULT);
void yolov5_deinit_post_process();
char *yolov5_cls_to_name(int cls_id);
int yolov5_post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, yolov5_object_detect_result_list *od_results, int num_classes = YOLOV5_OBJ_CLASS_NUM_DEFAULT);

void yolov5_deinitPostProcess();
#endif //_RKNN_YOLOV5_DEMO_POSTPROCESS_H_
