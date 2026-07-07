// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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

#include "yolov5_postprocess.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <algorithm>
#include <set>
#include <vector>

#include "yolov8.h"  // for rknn_app_context_t
#define LABEL_NALE_TXT_PATH "./model/coco_80_labels_list.txt"

static char *yolov5_labels[256];

const int yolov5_anchor[3][6] = {{10, 13, 16, 30, 33, 23},
                          {30, 61, 62, 45, 59, 119},
                          {116, 90, 156, 198, 373, 326}};

inline static int yolov5_clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

static char *yolov5_readLine(FILE *fp, char *buffer, int *len)
{
    int ch;
    int i = 0;
    size_t buff_len = 0;

    buffer = (char *)malloc(buff_len + 1);
    if (!buffer)
        return NULL; // Out of memory

    while ((ch = fgetc(fp)) != '\n' && ch != EOF)
    {
        buff_len++;
        void *tmp = realloc(buffer, buff_len + 1);
        if (tmp == NULL)
        {
            free(buffer);
            return NULL; // Out of memory
        }
        buffer = (char *)tmp;

        buffer[i] = (char)ch;
        i++;
    }
    buffer[i] = '\0';

    *len = buff_len;

    // Detect end
    if (ch == EOF && (i == 0 || ferror(fp)))
    {
        free(buffer);
        return NULL;
    }
    return buffer;
}

static int yolov5_readLines(const char *fileName, char *lines[], int max_line)
{
    FILE *file = fopen(fileName, "r");
    char *s;
    int i = 0;
    int n = 0;

    if (file == NULL)
    {
        printf("Open %s fail!\n", fileName);
        return -1;
    }

    while ((s = yolov5_readLine(file, s, &n)) != NULL)
    {
        lines[i++] = s;
        if (i >= max_line)
            break;
    }
    fclose(file);
    return i;
}

static int yolov5_loadLabelName(const char *locationFilename, char *label[], int num_classes)
{
    printf("load lable %s\n", locationFilename);
    yolov5_readLines(locationFilename, label, num_classes);
    return 0;
}

static float yolov5_CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int yolov5_nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId)
        {
            continue;
        }
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId)
            {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = yolov5_CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}

static int yolov5_quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        yolov5_quick_sort_indice_inverse(input, left, low - 1, indices);
        yolov5_quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

static float yolov5_sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

static float yolov5_unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

inline static int32_t yolov5_clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t yolov5_qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)yolov5_clip(dst_val, -128, 127);
    return res;
}

static uint8_t yolov5_qnt_f32_to_affine_u8(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    uint8_t res = (uint8_t)yolov5_clip(dst_val, 0, 255);
    return res;
}

static float yolov5_deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }
static float yolov5_deqnt_affine_u8_to_f32(uint8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static int yolov5_process_u8(uint8_t *input, int *anchor, int grid_h, int grid_w, int height, int width, int stride,
                      std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                      int32_t zp, float scale, int num_classes)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int prop_box_size = 5 + num_classes;
    uint8_t thres_u8 = yolov5_qnt_f32_to_affine_u8(threshold, zp, scale);
    for (int a = 0; a < 3; a++)
    {
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                uint8_t box_confidence = input[(prop_box_size * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= thres_u8)
                {
                    int offset = (prop_box_size * a) * grid_len + i * grid_w + j;
                    uint8_t *in_ptr = input + offset;
                    float box_x = (yolov5_deqnt_affine_u8_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                    float box_y = (yolov5_deqnt_affine_u8_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                    float box_w = (yolov5_deqnt_affine_u8_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                    float box_h = (yolov5_deqnt_affine_u8_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    uint8_t maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < num_classes; ++k)
                    {
                        uint8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs)
                        {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    float limit_score = (yolov5_deqnt_affine_u8_to_f32(maxClassProbs, zp, scale)) * (yolov5_deqnt_affine_u8_to_f32(box_confidence, zp, scale));
                    if (limit_score >= threshold)
                    {
                        objProbs.push_back(limit_score);
                        classId.push_back(maxClassId);
                        validCount++;
                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                    }
                }
            }
        }
    }
    return validCount;
}

static int yolov5_process_i8(int8_t *input, int *anchor, int grid_h, int grid_w, int height, int width, int stride,
                      std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                      int32_t zp, float scale, int num_classes)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int prop_box_size = 5 + num_classes;
    int8_t thres_i8 = yolov5_qnt_f32_to_affine(threshold, zp, scale);
    for (int a = 0; a < 3; a++)
    {
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                int8_t box_confidence = input[(prop_box_size * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= thres_i8)
                {
                    int offset = (prop_box_size * a) * grid_len + i * grid_w + j;
                    int8_t *in_ptr = input + offset;
                    float box_x = (yolov5_deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                    float box_y = (yolov5_deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                    float box_w = (yolov5_deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                    float box_h = (yolov5_deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    int8_t maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < num_classes; ++k)
                    {
                        int8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs)
                        {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    float limit_score = (yolov5_deqnt_affine_to_f32(maxClassProbs, zp, scale)) * (yolov5_deqnt_affine_to_f32(box_confidence, zp, scale));
                    if (limit_score >= threshold)
                    {
                        objProbs.push_back(limit_score);
                        classId.push_back(maxClassId);
                        validCount++;
                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                    }
                }
            }
        }
    }
    return validCount;
}

static int yolov5_process_i8_rv1106(int8_t *input, int *anchor, int grid_h, int grid_w, int height, int width, int stride,
                      std::vector<float> &boxes, std::vector<float> &boxScores, std::vector<int> &classId, float threshold,
                      int32_t zp, float scale, int num_classes) {
    int validCount = 0;
    int8_t thres_i8 = yolov5_qnt_f32_to_affine(threshold, zp, scale);

    int anchor_per_branch = 3;
    int prop_box_size = 5 + num_classes;
    int align_c = prop_box_size * anchor_per_branch;

    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            for (int a = 0; a < anchor_per_branch; a++) {
                int hw_offset = h * grid_w * align_c + w * align_c + a * prop_box_size;
                int8_t *hw_ptr = input + hw_offset;
                int8_t box_confidence = hw_ptr[4];

                if (box_confidence >= thres_i8) {
                    int8_t maxClassProbs = hw_ptr[5];
                    int maxClassId = 0;
                    for (int k = 1; k < num_classes; ++k) {
                        int8_t prob = hw_ptr[5 + k];
                        if (prob > maxClassProbs) {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }

                    float box_conf_f32 = yolov5_deqnt_affine_to_f32(box_confidence, zp, scale);
                    float class_prob_f32 = yolov5_deqnt_affine_to_f32(maxClassProbs, zp, scale);
                    float limit_score = box_conf_f32 * class_prob_f32;

                    if (limit_score > threshold) {
                        float box_x, box_y, box_w, box_h;

                        box_x = yolov5_deqnt_affine_to_f32(hw_ptr[0], zp, scale) * 2.0 - 0.5;
                        box_y = yolov5_deqnt_affine_to_f32(hw_ptr[1], zp, scale) * 2.0 - 0.5;
                        box_w = yolov5_deqnt_affine_to_f32(hw_ptr[2], zp, scale) * 2.0;
                        box_h = yolov5_deqnt_affine_to_f32(hw_ptr[3], zp, scale) * 2.0;
                        box_w = box_w * box_w;
                        box_h = box_h * box_h;


                        box_x = (box_x + w) * (float)stride;
                        box_y = (box_y + h) * (float)stride;
                        box_w *= (float)anchor[a * 2];
                        box_h *= (float)anchor[a * 2 + 1];

                        box_x -= (box_w / 2.0);
                        box_y -= (box_h / 2.0);

                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                        boxScores.push_back(limit_score);
                        classId.push_back(maxClassId);
                        validCount++;
                    }
                }
            }
        }
    }
    return validCount;
}

static int yolov5_process_fp32(float *input, int *anchor, int grid_h, int grid_w, int height, int width, int stride,
                        std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                        int num_classes)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int prop_box_size = 5 + num_classes;

    for (int a = 0; a < 3; a++)
    {
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                float box_confidence = input[(prop_box_size * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= threshold)
                {
                    int offset = (prop_box_size * a) * grid_len + i * grid_w + j;
                    float *in_ptr = input + offset;
                    float box_x = *in_ptr * 2.0 - 0.5;
                    float box_y = in_ptr[grid_len] * 2.0 - 0.5;
                    float box_w = in_ptr[2 * grid_len] * 2.0;
                    float box_h = in_ptr[3 * grid_len] * 2.0;
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    float maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < num_classes; ++k)
                    {
                        float prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs)
                        {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    if (maxClassProbs > threshold)
                    {
                        objProbs.push_back(maxClassProbs * box_confidence);
                        classId.push_back(maxClassId);
                        validCount++;
                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                    }
                }
            }
        }
    }
    return validCount;
}

int yolov5_post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, yolov5_object_detect_result_list *od_results, int num_classes)
{
    rknn_output *_outputs = (rknn_output *)outputs;
    int model_in_w = app_ctx->model_width;
    int model_in_h = app_ctx->model_height;
    float box_conf_thresh = (conf_threshold > 0.0f) ? conf_threshold : YOLOV5_BOX_THRESH;
    float nms_thresh = (nms_threshold > 0.0f) ? nms_threshold : YOLOV5_NMS_THRESH;

    if (num_classes <= 0)
    {
        num_classes = YOLOV5_OBJ_CLASS_NUM_DEFAULT;
    }

    memset(od_results, 0, sizeof(yolov5_object_detect_result_list));

    // ---------------------------------------------------------------
    // Format B: classic multi-branch raw YOLOv5 export (rknn_model_zoo
    // style). 3 outputs shaped [1, 3*(5+nc), gh, gw] for strides 8/16/32.
    // The exported model already applies sigmoid (output quant range is
    // [0,1]); we request float output (want_float=1) and decode per branch
    // with anchors.
    // ---------------------------------------------------------------
    if (app_ctx->io_num.n_output > 1)
    {
        std::vector<float> filterBoxes;
        std::vector<float> objProbs;
        std::vector<int> classId;
        int validCount = 0;

        for (int i = 0; i < app_ctx->io_num.n_output; i++)
        {
            int grid_h = app_ctx->output_attrs[i].dims[2];
            int grid_w = app_ctx->output_attrs[i].dims[3];
            int stride = model_in_h / grid_h;
            validCount += yolov5_process_fp32((float *)_outputs[i].buf, (int *)yolov5_anchor[i],
                                              grid_h, grid_w, model_in_h, model_in_w, stride,
                                              filterBoxes, objProbs, classId, box_conf_thresh, num_classes);
        }

        if (validCount <= 0)
        {
            return 0;
        }

        std::vector<int> indexArray;
        for (int i = 0; i < validCount; ++i)
        {
            indexArray.push_back(i);
        }
        yolov5_quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

        std::set<int> class_set(std::begin(classId), std::end(classId));
        for (auto c : class_set)
        {
            yolov5_nms(validCount, filterBoxes, classId, indexArray, c, nms_thresh);
        }

        int last_count = 0;
        for (int i = 0; i < validCount; ++i)
        {
            if (indexArray[i] == -1 || last_count >= YOLOV5_OBJ_NUMB_MAX_SIZE)
            {
                continue;
            }
            int n = indexArray[i];

            float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
            float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
            float x2 = x1 + filterBoxes[n * 4 + 2];
            float y2 = y1 + filterBoxes[n * 4 + 3];
            int id = classId[n];
            float obj_conf = objProbs[i];

            od_results->results[last_count].box.left   = (int)(yolov5_clamp(x1, 0, model_in_w) / letter_box->scale);
            od_results->results[last_count].box.top    = (int)(yolov5_clamp(y1, 0, model_in_h) / letter_box->scale);
            od_results->results[last_count].box.right  = (int)(yolov5_clamp(x2, 0, model_in_w) / letter_box->scale);
            od_results->results[last_count].box.bottom = (int)(yolov5_clamp(y2, 0, model_in_h) / letter_box->scale);
            od_results->results[last_count].prop = obj_conf;
            od_results->results[last_count].cls_id = id;
            last_count++;
        }
        od_results->count = last_count;
        return 0;
    }

    // ---------------------------------------------------------------
    // Format A: single-output decoded export [1, num_anchor, 5 + num_classes]
    // values already decoded (xywh in letterbox pixel space, sigmoid applied).
    // Use the caller-provided num_classes unless it was auto-detected above.
    // ---------------------------------------------------------------
    rknn_tensor_attr *out_attr = &app_ctx->output_attrs[0];
    int num_anchor = out_attr->dims[1];  // e.g. 25200
    int box_len = out_attr->dims[2];     // e.g. 7 = 5 + num_classes
    int detected_num_classes = box_len - 5;
    if (detected_num_classes > 0)
    {
        num_classes = detected_num_classes;
    }
    if (num_anchor <= 0 || num_classes <= 0)
    {
        printf("yolov5_post_process: invalid output dims [1, %d, %d], num_classes=%d\n", num_anchor, box_len, num_classes);
        return -1;
    }

    float *out = (float *)_outputs[0].buf;

    struct Candidate
    {
        float x, y, w, h;
        float score;
        int cls_id;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(300);

    for (int i = 0; i < num_anchor; i++)
    {
        const float *row = out + (size_t)i * box_len;
        float obj_conf = row[4];
        if (obj_conf <= box_conf_thresh)
        {
            continue;  // fast reject on objectness before scanning classes
        }

        float best_cls = 0.0f;
        int best_id = 0;
        for (int c = 0; c < num_classes; c++)
        {
            float p = row[5 + c];
            if (p > best_cls)
            {
                best_cls = p;
                best_id = c;
            }
        }

        float score = obj_conf * best_cls;
        if (score > box_conf_thresh)
        {
            candidates.push_back({row[0], row[1], row[2], row[3], score, best_id});
        }
    }

    if (candidates.empty())
    {
        return 0;
    }

    // Sort by score descending
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &a, const Candidate &b) { return a.score > b.score; });

    std::vector<bool> suppressed(candidates.size(), false);
    auto iou = [](const Candidate &a, const Candidate &b) -> float {
        float ax1 = a.x - a.w * 0.5f, ay1 = a.y - a.h * 0.5f;
        float ax2 = a.x + a.w * 0.5f, ay2 = a.y + a.h * 0.5f;
        float bx1 = b.x - b.w * 0.5f, by1 = b.y - b.h * 0.5f;
        float bx2 = b.x + b.w * 0.5f, by2 = b.y + b.h * 0.5f;
        float inter_w = std::max(0.0f, std::min(ax2, bx2) - std::max(ax1, bx1));
        float inter_h = std::max(0.0f, std::min(ay2, by2) - std::max(ay1, by1));
        float inter = inter_w * inter_h;
        float area_a = (ax2 - ax1) * (ay2 - ay1);
        float area_b = (bx2 - bx1) * (by2 - by1);
        return inter / (area_a + area_b - inter + 1e-6f);
    };

    for (size_t i = 0; i < candidates.size(); i++)
    {
        if (suppressed[i]) continue;
        for (size_t j = i + 1; j < candidates.size(); j++)
        {
            if (suppressed[j]) continue;
            if (candidates[i].cls_id != candidates[j].cls_id) continue;
            if (iou(candidates[i], candidates[j]) > nms_thresh)
            {
                suppressed[j] = true;
            }
        }
    }

    int last_count = 0;
    for (size_t i = 0; i < candidates.size() && last_count < YOLOV5_OBJ_NUMB_MAX_SIZE; i++)
    {
        if (suppressed[i]) continue;
        const Candidate &c = candidates[i];

        float x1 = c.x - c.w * 0.5f;
        float y1 = c.y - c.h * 0.5f;
        float x2 = c.x + c.w * 0.5f;
        float y2 = c.y + c.h * 0.5f;

        // Reverse letterbox: subtract pad, clamp to model input size, divide by scale
        x1 = (float)(yolov5_clamp(x1 - letter_box->x_pad, 0, model_in_w)) / letter_box->scale;
        y1 = (float)(yolov5_clamp(y1 - letter_box->y_pad, 0, model_in_h)) / letter_box->scale;
        x2 = (float)(yolov5_clamp(x2 - letter_box->x_pad, 0, model_in_w)) / letter_box->scale;
        y2 = (float)(yolov5_clamp(y2 - letter_box->y_pad, 0, model_in_h)) / letter_box->scale;

        od_results->results[last_count].box.left   = (int)(x1 + 0.5f);
        od_results->results[last_count].box.top    = (int)(y1 + 0.5f);
        od_results->results[last_count].box.right  = (int)(x2 + 0.5f);
        od_results->results[last_count].box.bottom = (int)(y2 + 0.5f);
        od_results->results[last_count].prop = c.score;
        od_results->results[last_count].cls_id = c.cls_id;
        last_count++;
    }
    od_results->count = last_count;
    return 0;
}

int yolov5_init_post_process(int num_classes)
{
    int ret = 0;
    if (num_classes <= 0)
    {
        num_classes = YOLOV5_OBJ_CLASS_NUM_DEFAULT;
    }
    ret = yolov5_loadLabelName(LABEL_NALE_TXT_PATH, yolov5_labels, num_classes);
    if (ret < 0)
    {
        printf("Load %s failed!\n", LABEL_NALE_TXT_PATH);
        return -1;
    }
    return 0;
}

char *yolov5_cls_to_name(int cls_id)
{
    if (cls_id >= 256)
    {
        return (char *)"null";
    }
    if (yolov5_labels[cls_id])
    {
        return yolov5_labels[cls_id];
    }
    return (char *)"null";
}

void yolov5_deinit_post_process()
{
    for (int i = 0; i < 256; i++)
    {
        if (yolov5_labels[i] != nullptr)
        {
            free(yolov5_labels[i]);
            yolov5_labels[i] = nullptr;
        }
    }
}
