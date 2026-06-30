#ifndef YOLOV8_POSE_POSTPROCESS_H
#define YOLOV8_POSE_POSTPROCESS_H

#include "yolov8.h"

#define YOLOV8_POSE_OBJ_CLASS_NUM 1
#define YOLOV8_POSE_NMS_THRESH 0.4
#define YOLOV8_POSE_BOX_THRESH 0.5

int yolov8_pose_init_post_process();
void yolov8_pose_deinit_post_process();
char *yolov8_pose_cls_to_name(int cls_id);
int yolov8_pose_post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results);

#endif
