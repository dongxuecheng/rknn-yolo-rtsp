#ifndef _RKNN_DEMO_YOLO26_POSTPROCESS_H_
#define _RKNN_DEMO_YOLO26_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rknn_api.h"
#include "common.h"
#include "image_utils.h"
#include "yolov8.h"  // for rknn_app_context_t

#define YOLO26_OBJ_NAME_MAX_SIZE 64
#define YOLO26_OBJ_NUMB_MAX_SIZE 128
#define YOLO26_OBJ_CLASS_NUM 80
#define YOLO26_NMS_THRESH 0.45
#define YOLO26_BOX_THRESH 0.25

typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
} yolo26_object_detect_result;

typedef struct {
    int id;
    int count;
    yolo26_object_detect_result results[YOLO26_OBJ_NUMB_MAX_SIZE];
} yolo26_object_detect_result_list;

int yolo26_init_post_process();
void yolo26_deinit_post_process();
char *yolo26_cls_to_name(int cls_id);
int yolo26_post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, yolo26_object_detect_result_list *od_results);

#endif // _RKNN_DEMO_YOLO26_POSTPROCESS_H_
