#include "yolo26_postprocess.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <algorithm>
#include <vector>

#define YOLO26_LABEL_TXT_PATH "./model/coco_80_labels_list.txt"

static char *yolo26_labels[YOLO26_OBJ_CLASS_NUM];

inline static int yolo26_clamp(float val, int min, int max) {
    return val > min ? (val < max ? val : max) : min;
}

static char *yolo26_readLine(FILE *fp, char *buffer, int *len) {
    int ch;
    int i = 0;
    size_t buff_len = 0;
    buffer = (char *)malloc(buff_len + 1);
    if (!buffer) return NULL;
    while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
        buff_len++;
        void *tmp = realloc(buffer, buff_len + 1);
        if (tmp == NULL) {
            free(buffer);
            return NULL;
        }
        buffer = (char *)tmp;
        buffer[i] = (char)ch;
        i++;
    }
    buffer[i] = '\0';
    *len = buff_len;
    if (ch == EOF && (i == 0 || ferror(fp))) {
        free(buffer);
        return NULL;
    }
    return buffer;
}

static int yolo26_readLines(const char *fileName, char *lines[], int max_line) {
    FILE *file = fopen(fileName, "r");
    char *s;
    int i = 0;
    int n = 0;
    if (file == NULL) {
        printf("Open %s fail!\n", fileName);
        return -1;
    }
    while ((s = yolo26_readLine(file, s, &n)) != NULL) {
        lines[i++] = s;
        if (i >= max_line) break;
    }
    fclose(file);
    return i;
}

static int yolo26_loadLabelName(const char *locationFilename, char *label[]) {
    printf("load label %s\n", locationFilename);
    yolo26_readLines(locationFilename, label, YOLO26_OBJ_CLASS_NUM);
    return 0;
}

int yolo26_post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box,
                        float conf_threshold, float nms_threshold,
                        yolo26_object_detect_result_list *od_results) {
    rknn_output *_outputs = (rknn_output *)outputs;
    int model_in_w = app_ctx->model_width;
    int model_in_h = app_ctx->model_height;

    memset(od_results, 0, sizeof(yolo26_object_detect_result_list));

    // yolo26 has single output: [1, 84, 8400]
    if (app_ctx->io_num.n_output != 1) {
        printf("yolo26_post_process: expected 1 output, got %d\n", app_ctx->io_num.n_output);
        return -1;
    }

    rknn_tensor_attr *out_attr = &app_ctx->output_attrs[0];
    int num_ch = out_attr->dims[1];      // 84
    int num_anchor = out_attr->dims[2];  // 8400

    if (num_ch != 84) {
        printf("yolo26_post_process: expected 84 channels, got %d\n", num_ch);
        return -1;
    }

    float *out = (float *)_outputs[0].buf;
    float box_conf_thresh = (conf_threshold > 0.0f) ? conf_threshold : YOLO26_BOX_THRESH;

    struct Candidate {
        float x, y, w, h;
        float score;
        int cls_id;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(300);

    for (int i = 0; i < num_anchor; i++) {
        // layout: channel-major
        float x = out[0 * num_anchor + i];
        float y = out[1 * num_anchor + i];
        float w = out[2 * num_anchor + i];
        float h = out[3 * num_anchor + i];

        // skip invalid boxes
        if (w <= 0.0f || h <= 0.0f || x < 0.0f || y < 0.0f) continue;

        float best_score = 0.0f;
        int best_cls = -1;
        for (int c = 4; c < num_ch; c++) {
            float score = out[c * num_anchor + i];
            if (score > best_score) {
                best_score = score;
                best_cls = c - 4;
            }
        }

        if (best_score > box_conf_thresh) {
            candidates.push_back({x, y, w, h, best_score, best_cls});
        }
    }

    if (candidates.empty()) {
        return 0;
    }

    // Sort by score descending
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &a, const Candidate &b) { return a.score > b.score; });

    // NMS (yolo26 still produces duplicated boxes in practice)
    float nms_thresh = (nms_threshold > 0.0f) ? nms_threshold : YOLO26_NMS_THRESH;
    std::vector<bool> suppressed(candidates.size(), false);

    auto iou = [](const Candidate& a, const Candidate& b) -> float {
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

    for (size_t i = 0; i < candidates.size(); i++) {
        if (suppressed[i]) continue;
        for (size_t j = i + 1; j < candidates.size(); j++) {
            if (suppressed[j]) continue;
            if (candidates[i].cls_id != candidates[j].cls_id) continue;
            if (iou(candidates[i], candidates[j]) > nms_thresh) {
                suppressed[j] = true;
            }
        }
    }

    int last_count = 0;
    for (size_t i = 0; i < candidates.size() && last_count < YOLO26_OBJ_NUMB_MAX_SIZE; i++) {
        if (suppressed[i]) continue;
        const Candidate &c = candidates[i];

        float x1 = c.x - c.w * 0.5f;
        float y1 = c.y - c.h * 0.5f;
        float x2 = c.x + c.w * 0.5f;
        float y2 = c.y + c.h * 0.5f;

        // Reverse letterbox: subtract pad first, clamp to model input size, then divide by scale
        x1 = (float)(yolo26_clamp(x1 - letter_box->x_pad, 0, model_in_w)) / letter_box->scale;
        y1 = (float)(yolo26_clamp(y1 - letter_box->y_pad, 0, model_in_h)) / letter_box->scale;
        x2 = (float)(yolo26_clamp(x2 - letter_box->x_pad, 0, model_in_w)) / letter_box->scale;
        y2 = (float)(yolo26_clamp(y2 - letter_box->y_pad, 0, model_in_h)) / letter_box->scale;

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

int yolo26_init_post_process() {
    int ret = yolo26_loadLabelName(YOLO26_LABEL_TXT_PATH, yolo26_labels);
    if (ret < 0) {
        printf("Load %s failed!\n", YOLO26_LABEL_TXT_PATH);
        return -1;
    }
    return 0;
}

char *yolo26_cls_to_name(int cls_id) {
    if (cls_id >= YOLO26_OBJ_CLASS_NUM) {
        return (char *)"null";
    }
    if (yolo26_labels[cls_id]) {
        return yolo26_labels[cls_id];
    }
    return (char *)"null";
}

void yolo26_deinit_post_process() {
    for (int i = 0; i < YOLO26_OBJ_CLASS_NUM; i++) {
        if (yolo26_labels[i] != nullptr) {
            free(yolo26_labels[i]);
            yolo26_labels[i] = nullptr;
        }
    }
}
