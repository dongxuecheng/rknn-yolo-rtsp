#ifndef _RKNN_DEMO_YOLO_WORLD_POSTPROCESS_H_
#define _RKNN_DEMO_YOLO_WORLD_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rknn_api.h"
#include "common.h"
#include "image_utils.h"
#include "yolov8.h"  // for rknn_app_context_t

#define YOLO_WORLD_OBJ_NAME_MAX_SIZE 64
#define YOLO_WORLD_OBJ_NUMB_MAX_SIZE 128
#define YOLO_WORLD_OBJ_CLASS_NUM 80
#define YOLO_WORLD_NMS_THRESH 0.45
#define YOLO_WORLD_BOX_THRESH 0.25

typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
} yolo_world_object_detect_result;

typedef struct {
    int id;
    int count;
    yolo_world_object_detect_result results[YOLO_WORLD_OBJ_NUMB_MAX_SIZE];
} yolo_world_object_detect_result_list;

int yolo_world_init_post_process();
void yolo_world_deinit_post_process();
char *yolo_world_cls_to_name(int cls_id);
int yolo_world_post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, yolo_world_object_detect_result_list *od_results);

#endif //_RKNN_DEMO_YOLO_WORLD_POSTPROCESS_H_
