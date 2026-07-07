#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>

#include "yolov8.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

/*-------------------------------------------
                  Main Function
-------------------------------------------*/

cv::Mat convert_to_opencv_image(image_buffer_t *img_buffer)
{
    cv::Mat mat;
    cv::Mat rgb; // 将 rgb 定义移到外部，确保所有情况都可以使用它
    // 根据图片格式转换
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
        // YUV -> RGB转换
        mat = cv::Mat(img_buffer->height + img_buffer->height / 2, img_buffer->width, CV_8UC1, img_buffer->virt_addr);
        cv::cvtColor(mat, rgb, cv::COLOR_YUV2BGR_NV21); // 使用 NV21 转换，如果是 NV12 则改为 COLOR_YUV2BGR_NV12
        mat = rgb;
        break;

    default:
        printf("Unsupported image format\n");
        return cv::Mat(); // 返回一个空的 Mat 表示失败
    }

    return mat;
}


int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("%s <model_path> <image_path>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    ret = init_yolov8_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
        return -1;
        // goto out;
    }

    // Reload labels with actual class count after model init
    deinit_post_process();
    init_post_process(rknn_app_ctx.num_classes);

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    ret = read_image(image_path, &src_image);

    if (ret != 0)
    {
        printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
        return -1;
        // goto out;
    }

    object_detect_result_list od_results;
    ret = inference_yolov8_model(&rknn_app_ctx, &src_image, &od_results);
    if (ret != 0)
    {
        printf("init_yolov8_model fail! ret=%d\n", ret);
        return -1;
        // goto out;
    }
    // 使用 OpenCV 转换图像
    cv::Mat img = convert_to_opencv_image(&src_image);

    // 如果转换失败，处理并退出
    if (img.empty())
    {
        printf("Error converting image to OpenCV format.\n");
        // goto out;
        return -1;
    }

    // 绘制框和概率
    for (int i = 0; i < od_results.count; i++)
    {
        object_detect_result *det_result = &(od_results.results[i]);
        printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
               det_result->box.left, det_result->box.top,
               det_result->box.right, det_result->box.bottom,
               det_result->prop);

        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        // 使用 OpenCV 绘制矩形框和文本
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 3);

        char text[256];
        sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
        cv::putText(img, text, cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    }

    // 保存结果
    cv::imwrite("out.png", img);

    deinit_post_process();

    ret = release_yolov8_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolov8_model fail! ret=%d\n", ret);
    }

    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);
    }

    return 0;
}
