/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <signal.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <stdexcept> // for std::runtime_error
#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include <atomic>
#include <fstream>

#include "im2d.h"
#include "rga.h"
#include "RgaUtils.h"
#include "rknn_api.h"

#include "rkmedia/utils/mpp_decoder.h"
#include "rkmedia/utils/mpp_encoder.h"

#include "mk_mediakit.h"
#include "task/yolo_model.h"
#include "alarm/SerialAlarm.h"

#include <future>
#include <thread>
#include <functional>

#include "task/ThreadPool.h"

#include "yolov8.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#include <unistd.h>
#include <sys/stat.h> // For mkdir (Linux/Unix)

#include <nlohmann/json.hpp>

#define OUT_VIDEO_PATH "out.h264"

// 流级上下文（扩展原 rknn_app_ctx）
struct StreamContext {
    FILE *out_fp = nullptr;
    MppDecoder *decoder = nullptr;
    MppEncoder *encoder = nullptr;
    mk_media media = nullptr;
    mk_pusher pusher = nullptr;
    mk_player player = nullptr;
    std::string pull_url;
    std::string push_url;
    uint64_t pts = 0;
    uint64_t dts = 0;

    // 流级配置
    std::string stream_id;
    std::string save_dir;
    int frame_num = 5;
    std::vector<YoloModel*> models;

    // ROI 报警配置
    cv::Rect roi;
    std::vector<std::string> roi_model_types;

    // 运行时状态
    std::atomic<int> frame_index{0};
};

struct StreamModelConfig {
    std::string type;
    std::string mode;
    std::string textModel;
    std::vector<std::string> textDescriptions;
    std::vector<std::string> classes;
    float objectThreshold = -1.0f;
    float nmsThreshold = -1.0f;
};

struct StreamJsonConfig {
    std::string id;
    std::string url;
    std::string save_dir;
    int frame_num = 5;
    cv::Rect roi;
    std::vector<std::string> roi_model_types;
    std::vector<StreamModelConfig> models;
};

static ThreadPool *g_pool = nullptr;
static std::vector<std::unique_ptr<StreamContext>> g_streams;
static bool g_mk_inited = false;

void release_media(mk_media *ptr)
{
    if (ptr && *ptr)
    {
        mk_media_release(*ptr);
        *ptr = NULL;
    }
}

void release_pusher(mk_pusher *ptr)
{
    if (ptr && *ptr)
    {
        mk_pusher_release(*ptr);
        *ptr = NULL;
    }
}

void release_player(mk_player *ptr)
{
    if (ptr && *ptr)
    {
        mk_player_release(*ptr);
        *ptr = NULL;
    }
}

// 获取当前时间戳字符串
std::string GetCurrentTimestamp()
{
    std::time_t now = std::time(0); // 获取当前时间（时间戳）
    return std::to_string(now);     // 返回时间戳（秒）
}

void createDirectoryIfNotExists(const std::string &dir)
{
    // 检查目录是否存在
    struct stat info;
    if (stat(dir.c_str(), &info) != 0)
    {
        // 目录不存在，递归创建父目录
        size_t pos = 0;
        while ((pos = dir.find('/', pos + 1)) != std::string::npos)
        {
            std::string subdir = dir.substr(0, pos);
            if (stat(subdir.c_str(), &info) != 0)
            {
                if (mkdir(subdir.c_str(), 0777) != 0)
                {
                    std::cerr << "Error creating directory: " << subdir << std::endl;
                    return;
                }
            }
        }
        // 最终创建目标目录
        if (mkdir(dir.c_str(), 0777) != 0)
        {
            std::cerr << "Error creating directory: " << dir << std::endl;
        }
        else
        {
            std::cout << "Directory created: " << dir << std::endl;
        }
    }
    else if (info.st_mode & S_IFDIR)
    {
        // 目录已经存在
    }
    else
    {
        std::cerr << dir << " is not a directory." << std::endl;
    }
}

// 从 config.json 读取多流配置（兼容旧格式）
std::vector<StreamJsonConfig> load_stream_configs(const std::string& path)
{
    std::vector<StreamJsonConfig> result;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Could not open config file: " << path << std::endl;
        return result;
    }

    nlohmann::json config;
    try {
        file >> config;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse config.json: " << e.what() << std::endl;
        return result;
    }

    if (!config.contains("streams") || !config["streams"].is_array()) {
        std::cerr << "config.json must contain a 'streams' array." << std::endl;
        return result;
    }
    for (const auto& s : config["streams"]) {
        StreamJsonConfig sc;
        sc.id = s.value("id", "stream");
        sc.url = s.value("url", "");
        sc.save_dir = s.value("saveDir", "./output/" + sc.id);
        sc.frame_num = s.value("frameNum", 5);
        if (s.contains("models") && s["models"].is_array()) {
            for (const auto& m : s["models"]) {
                StreamModelConfig smc;
                smc.type = m.value("type", "yolo11");
                smc.mode = m.value("mode", "");
                smc.textModel = m.value("textModel", "");
                if (m.contains("textDescriptions") && m["textDescriptions"].is_array()) {
                    smc.textDescriptions = m["textDescriptions"].get<std::vector<std::string>>();
                }
                if (m.contains("classes") && m["classes"].is_array()) {
                    smc.classes = m["classes"].get<std::vector<std::string>>();
                }
                smc.objectThreshold = m.value("objectThreshold", -1.0f);
                smc.nmsThreshold = m.value("nmsThreshold", -1.0f);
                sc.models.push_back(smc);
            }
        }
        // 解析 ROI
        if (s.contains("roi") && s["roi"].is_object()) {
            auto r = s["roi"];
            sc.roi.x = r.value("x1", 0);
            sc.roi.y = r.value("y1", 0);
            sc.roi.width = r.value("x2", 0) - sc.roi.x;
            sc.roi.height = r.value("y2", 0) - sc.roi.y;
            if (sc.roi.width < 0) sc.roi.width = 0;
            if (sc.roi.height < 0) sc.roi.height = 0;
        }
        // 解析参与 ROI 报警判定的模型白名单
        if (s.contains("roiModels") && s["roiModels"].is_array()) {
            sc.roi_model_types = s["roiModels"].get<std::vector<std::string>>();
        } else {
            // 默认所有模型都参与报警判定
            for (const auto& m : sc.models) {
                sc.roi_model_types.push_back(m.type);
            }
        }
        result.push_back(sc);
    }
    return result;
}

// 为指定流加载模型（每流独立实例，因为 RKNN 上下文非线程安全）
bool load_models_for_stream(const StreamJsonConfig& cfg, StreamContext* stream)
{
    for (const auto& mcfg : cfg.models) {
        YoloModel* model = new YoloModel();
        int ret = -1;
        if (mcfg.type == "yolo_world") {
            std::cout << "[" << cfg.id << "] Loading YOLO-World model: " << mcfg.mode << std::endl;
            ret = model->load_model(mcfg.mode.c_str(), ModelType::YOLO_WORLD);
            if (ret == 0 && !mcfg.textModel.empty()) {
                ret = model->load_text_model(mcfg.textModel.c_str(), mcfg.textDescriptions);
            }
        } else if (mcfg.type == "yolov8_pose") {
            std::cout << "[" << cfg.id << "] Loading YOLOv8-Pose model: " << mcfg.mode << std::endl;
            ret = model->load_model(mcfg.mode.c_str(), ModelType::YOLOV8_POSE);
        } else if (mcfg.type == "yolo26n") {
            std::cout << "[" << cfg.id << "] Loading YOLO26N model: " << mcfg.mode << std::endl;
            ret = model->load_model(mcfg.mode.c_str(), ModelType::YOLO26N);
        } else {
            std::cout << "[" << cfg.id << "] Loading YOLO11 model: " << mcfg.mode << std::endl;
            ret = model->load_model(mcfg.mode.c_str(), ModelType::YOLO11);
        }

        if (ret != 0) {
            std::cerr << "[" << cfg.id << "] Failed to load model " << mcfg.mode << ", skip." << std::endl;
            delete model;
            continue;
        }

        // 设置该模型的自定义类别名称（yolo11 / yolov8_pose 必须在 config.json 中显式配置）
        if (!mcfg.classes.empty()) {
            model->setClassNames(mcfg.classes);
        } else if (mcfg.type == "yolo11" || mcfg.type == "yolov8_pose") {
            std::cerr << "[WARNING] [" << cfg.id << "] Model " << mcfg.mode 
                      << " (type=" << mcfg.type << ") has no 'classes' configured. "
                      << "Please add 'classes' to this model in config.json." << std::endl;
        }
        // 设置该模型的独立阈值（如果配置中有）
        if (mcfg.objectThreshold > 0.0f || mcfg.nmsThreshold > 0.0f) {
            model->setThresholds(mcfg.objectThreshold, mcfg.nmsThreshold);
        }

        stream->models.push_back(model);
    }
    stream->roi = cfg.roi;
    stream->roi_model_types = cfg.roi_model_types;
    return !stream->models.empty();
}

// 线程池任务处理函数

// 线程池任务处理函数
void process_frame(cv::Mat origin_mat, StreamContext* stream)
{
    try
    {
        auto process_start = std::chrono::high_resolution_clock::now();

        // 1. 级联推理：依次运行该流配置的所有模型
        struct ModelInferResult {
            YoloModel* model;
            std::string name;
            object_detect_result_list results;
        };
        std::vector<ModelInferResult> all_results;

        for (auto* model : stream->models)
        {
            object_detect_result_list od_results;
            memset(&od_results, 0, sizeof(od_results));

            auto infer_start = std::chrono::high_resolution_clock::now();
            int ret = model->infer(origin_mat, od_results);
            auto infer_end = std::chrono::high_resolution_clock::now();
            int infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_start).count();

            std::string type_str;
            switch (model->getModelType()) {
                case ModelType::YOLO11: type_str = "yolo11"; break;
                case ModelType::YOLO_WORLD: type_str = "yolo_world"; break;
                case ModelType::YOLOV8_POSE: type_str = "yolov8_pose"; break;
                case ModelType::YOLO26N: type_str = "yolo26n"; break;
            }
            std::cout << "[" << stream->stream_id << "] [TIMING] " << type_str 
                      << " infer: " << infer_ms << "ms, objects: " << od_results.count << std::endl;

            if (ret == 0) {
                all_results.push_back({model, type_str, od_results});
            }
        }

        // ==========================================
        // ROI 报警判定（与保存逻辑独立）
        // ==========================================
        bool has_person_in_roi = false;
        if (stream->roi.width > 0 && stream->roi.height > 0) {
            for (auto& r : all_results) {
                // 检查该模型是否在 roi_model_types 白名单中
                bool model_in_roi_list = false;
                for (const auto& mt : stream->roi_model_types) {
                    if (mt == r.name) { model_in_roi_list = true; break; }
                }
                if (!model_in_roi_list) continue;

                for (int i = 0; i < r.results.count; i++) {
                    auto& det = r.results.results[i];
                    if (r.model->getClassName(det.cls_id) != "person") continue;
                    // 相交法：检测框与 ROI 是否有交集
                    if (!(det.box.right < stream->roi.x ||
                          det.box.left > stream->roi.x + stream->roi.width ||
                          det.box.bottom < stream->roi.y ||
                          det.box.top > stream->roi.y + stream->roi.height)) {
                        has_person_in_roi = true;
                        break;
                    }
                }
                if (has_person_in_roi) break;
            }
        }
        SerialAlarm::getInstance().report(stream->stream_id, has_person_in_roi);
        std::cout << "[" << stream->stream_id << "] [ALARM] person_in_roi=" << has_person_in_roi << std::endl;

        // ==========================================
        // 检测逻辑位置：此处判定该流是否保存图片
        // 当前逻辑：该流配置的【所有模型】均检测到人，才保存
        // 后期如需修改判定条件，请直接修改下方代码
        // ==========================================
        bool should_save = false;
        if (!all_results.empty()) {
            bool all_has_person = true;
            for (auto& r : all_results) {
                bool has_person = false;
                for (int i = 0; i < r.results.count; i++) {
                    if (r.model->getClassName(r.results.results[i].cls_id) == "person") {
                        has_person = true;
                        break;
                    }
                }
                if (!has_person) { all_has_person = false; break; }
            }
            should_save = all_has_person;
        }

        std::cout << "[" << stream->stream_id << "] [LOGIC] should_save=" << should_save << std::endl;

        if (!should_save)
        {
            std::cout << "[" << stream->stream_id << "] Logic check failed, skip saving." << std::endl;
            return;
        }

        // 3. 绘制并保存（仅在逻辑判定通过后执行）
        if (origin_mat.channels() == 3)
        {
            cv::cvtColor(origin_mat, origin_mat, cv::COLOR_RGB2BGR);
        }

        std::string dir = stream->save_dir.empty() ? "./output" : stream->save_dir;
        createDirectoryIfNotExists(dir);
        std::string timestamp = GetCurrentTimestamp();
        std::string tid = stream->stream_id;

        // COCO-pose skeleton edges (0-based indices)
        static const int skeleton[38] = {
            15,13, 13,11, 16,14, 14,12, 11,12, 5,11, 6,12, 5,6, 5,7,
            6,8, 7,9, 8,10, 1,2, 0,1, 0,2, 1,3, 2,4, 3,5, 4,6
        };

        for (auto& r : all_results)
        {
            ModelType mtype = r.model->getModelType();
            for (int i = 0; i < r.results.count; i++)
            {
                object_detect_result *det_result = &(r.results.results[i]);
                int x1 = det_result->box.left;
                int y1 = det_result->box.top;
                int x2 = det_result->box.right;
                int y2 = det_result->box.bottom;
                x1 = std::max(x1, 0);
                y1 = std::max(y1, 0);
                x2 = std::min(x2, origin_mat.cols);
                y2 = std::min(y2, origin_mat.rows);

                // 按模型类型分配不同颜色（BGR格式）
                cv::Scalar box_color, text_color;
                if (mtype == ModelType::YOLO11) {
                    box_color = cv::Scalar(0, 0, 255);    // 红色
                    text_color = cv::Scalar(0, 0, 255);   // 红色
                } else if (mtype == ModelType::YOLO_WORLD) {
                    box_color = cv::Scalar(0, 255, 0);    // 绿色
                    text_color = cv::Scalar(0, 255, 0);   // 绿色
                } else if (mtype == ModelType::YOLO26N) {
                    box_color = cv::Scalar(0, 255, 255);  // 黄色
                    text_color = cv::Scalar(0, 255, 255); // 黄色
                } else { // YOLOV8_POSE
                    box_color = cv::Scalar(255, 0, 0);    // 蓝色
                    text_color = cv::Scalar(255, 0, 0);   // 蓝色
                }

                if (mtype == ModelType::YOLOV8_POSE)
                {
                    // Pose 模型：绘制框 + 骨骼 + 关键点
                    cv::rectangle(origin_mat, cv::Point(x1, y1), cv::Point(x2, y2), box_color, 2);
                    std::string label = r.model->getClassName(det_result->cls_id);
                    char text[256];
                    snprintf(text, sizeof(text), "%s %.1f%%", label.c_str(), det_result->prop * 100);
                    cv::putText(origin_mat, text, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);

                    for (int j = 0; j < 19; ++j)
                    {
                        int kpt_a = skeleton[2*j];
                        int kpt_b = skeleton[2*j+1];
                        if (det_result->keypoints[kpt_a][2] > 0.5f && det_result->keypoints[kpt_b][2] > 0.5f)
                        {
                            cv::Point pa((int)det_result->keypoints[kpt_a][0], (int)det_result->keypoints[kpt_a][1]);
                            cv::Point pb((int)det_result->keypoints[kpt_b][0], (int)det_result->keypoints[kpt_b][1]);
                            cv::line(origin_mat, pa, pb, cv::Scalar(0, 255, 255), 2); // 骨骼：黄色
                        }
                    }
                    for (int j = 0; j < 17; ++j)
                    {
                        if (det_result->keypoints[j][2] > 0.5f)
                        {
                            cv::circle(origin_mat,
                                       cv::Point((int)det_result->keypoints[j][0], (int)det_result->keypoints[j][1]),
                                       3, cv::Scalar(255, 255, 0), -1); // 关键点：青色
                        }
                    }
                }
                else
                {
                    // YOLO11 / YOLO-World：只绘制检测框和标签
                    cv::rectangle(origin_mat, cv::Point(x1, y1), cv::Point(x2, y2), box_color, 2);
                    std::string label = r.model->getClassName(det_result->cls_id);
                    char text[256];
                    snprintf(text, sizeof(text), "%s %.1f%%", label.c_str(), det_result->prop * 100);
                    cv::putText(origin_mat, text, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
                }
            }
        }

        std::ostringstream file_name;
        file_name << dir << "/" << tid << "#" << timestamp << "#detected.png";
        std::cout << "[" << stream->stream_id << "] save_image: " << file_name.str() << std::endl;
        cv::imwrite(file_name.str(), origin_mat);

        auto process_end = std::chrono::high_resolution_clock::now();
        std::cout << "[" << stream->stream_id << "] process_frame total: " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(process_end - process_start).count() << "ms" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[" << stream->stream_id << "] Error in processing frame: " << e.what() << '\n';
    }
}

void mpp_decoder_frame_callback(void *userdata, int width_stride, int height_stride, int width, int height, int format, int fd, void *data)
{
    try
    {
        StreamContext *stream = (StreamContext *)userdata;
        int ret = 0;
        int frame_num = stream->frame_num;
        int idx = stream->frame_index.fetch_add(1) + 1;

        // 每隔 frame_num 帧提交一次推理任务给线程池
        if (idx % frame_num == 0)
        {
            // 使用 RK RGA 的格式转换：YUV420SP -> RGB888
            rga_buffer_t origin = wrapbuffer_fd(fd, width, height, RK_FORMAT_YCbCr_420_SP, width_stride, height_stride);
            cv::Mat origin_mat = cv::Mat::zeros(height, width, CV_8UC3);
            rga_buffer_t rgb_img = wrapbuffer_virtualaddr((void *)origin_mat.data, width, height, RK_FORMAT_RGB_888);
            imcopy(origin, rgb_img);
            // 提交推理任务到线程池
            if (!g_pool->submit([=]()
                              { process_frame(std::move(origin_mat), stream); }))
            {
                std::cerr << "[" << stream->stream_id << "] Task discarded, thread pool is full\n";
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "mpp_decoder_frame_callback err:" << e.what() << '\n';
    }
}

void API_CALL on_track_frame_out(void *user_data, mk_frame frame)
{
    try
    {
        StreamContext *stream = (StreamContext *)user_data;
        const char *data = mk_frame_get_data(frame);
        stream->dts = mk_frame_get_dts(frame);
        stream->pts = mk_frame_get_pts(frame);
        size_t size = mk_frame_get_data_size(frame);
        stream->decoder->Decode((uint8_t *)data, size, 0);
    }
    catch (const std::exception &e)
    {
        std::cerr << "on_track_frame_out err :" << e.what() << '\n';
    }
}

/**
 * 处理视频播放事件的回调函数
 */
static void init_video_decoder(StreamContext *stream, mk_track track)
{
    const char *codec_name = mk_track_codec_name(track);
    log_info("got video track: %s", codec_name);

    int video_type = 264;
    if (strstr(codec_name, "265") || strstr(codec_name, "HEVC") || strstr(codec_name, "hevc"))
    {
        video_type = 265;
    }
    else if (strstr(codec_name, "264") || strstr(codec_name, "AVC") || strstr(codec_name, "avc"))
    {
        video_type = 264;
    }
    else
    {
        printf("warning: unknown video codec %s, default to H264\n", codec_name);
    }
    printf("auto detect video type: %d (codec: %s)\n", video_type, codec_name);

    if (stream->decoder == NULL)
    {
        MppDecoder *decoder = new MppDecoder();
        decoder->Init(video_type, 25, stream);
        decoder->SetCallback(mpp_decoder_frame_callback);
        stream->decoder = decoder;
    }
    else
    {
        stream->decoder->Reset();
    }
}

void API_CALL on_mk_play_event_func(void *user_data, int err_code, const char *err_msg, mk_track tracks[], int track_count)
{
    try
    {
        StreamContext *stream = (StreamContext *)user_data;
        release_media(&(stream->media));
        if (err_code == 0)
        {
            printf("play success!");
            int i;
            // 使用流ID作为stream name，避免多流冲突
            stream->media = mk_media_create("__defaultVhost__", "live", stream->stream_id.c_str(), 0, 0, 0);
            for (i = 0; i < track_count; ++i)
            {
                if (mk_track_is_video(tracks[i]))
                {
                    init_video_decoder(stream, tracks[i]);
                    mk_media_init_track(stream->media, tracks[i]);
                    mk_track_add_delegate(tracks[i], on_track_frame_out, user_data);
                }
            }
            mk_media_init_complete(stream->media);
        }
        else
        {
            printf("play failed: %d %s", err_code, err_msg);
            mk_player_play(stream->player, stream->pull_url.c_str());
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "on_mk_play_event_func err: " << e.what() << '\n';
    }
}

void API_CALL on_mk_shutdown_func(void *user_data, int err_code, const char *err_msg, mk_track tracks[], int track_count)
{
    try
    {
        printf("play interrupted: %d %s", err_code, err_msg);
        StreamContext *stream = (StreamContext *)user_data;
        release_media(&(stream->media));
        if (err_code == 0)
        {
            printf("play success!");
            int i;
            stream->media = mk_media_create("__defaultVhost__", "live", stream->stream_id.c_str(), 0, 0, 0);
            for (i = 0; i < track_count; ++i)
            {
                if (mk_track_is_video(tracks[i]))
                {
                    init_video_decoder(stream, tracks[i]);
                    mk_media_init_track(stream->media, tracks[i]);
                    mk_track_add_delegate(tracks[i], on_track_frame_out, user_data);
                }
            }
            mk_media_init_complete(stream->media);
        }
        else
        {
            printf("on_mk_shutdown_func failed: %d %s", err_code, err_msg);
            mk_player_play(stream->player, stream->pull_url.c_str());
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "on_mk_shutdown_func err: " << e.what() << '\n';
    }
}

void player_create(StreamContext *stream)
{
    stream->player = mk_player_create();
    mk_player_set_on_result(stream->player, on_mk_play_event_func, stream);
    mk_player_set_on_shutdown(stream->player, on_mk_shutdown_func, stream);
    mk_player_play(stream->player, stream->pull_url.c_str());
}

/**
 * 处理 RTSP 视频流
 */
int process_video_rtsp(StreamContext *stream)
{
    if (!g_mk_inited) {
        mk_config config;
        memset(&config, 0, sizeof(mk_config));
        config.log_mask = LOG_CONSOLE;
        mk_env_init(&config);
        g_mk_inited = true;
    }
    player_create(stream);
    return 0;
}

static int exit_flag = 0;
void s_on_exit(int signum)
{
    std::cout << "Received signal " << signum << ", exiting...\n";
    exit(signum);
}

int main(int argc, char **argv)
{
    int status = 0;
    int ret;

    try
    {
        // 1. 读取多流配置（兼容旧格式）
        auto stream_configs = load_stream_configs("config.json");
        if (stream_configs.empty()) {
            std::cerr << "No valid stream config found, exit." << std::endl;
            return -1;
        }
        std::cout << "Loaded " << stream_configs.size() << " stream(s) from config." << std::endl;

        // 2. 为每流创建上下文并加载模型
        for (const auto& sc : stream_configs) {
            auto ctx = std::make_unique<StreamContext>();
            ctx->stream_id = sc.id;
            ctx->save_dir = sc.save_dir;
            ctx->frame_num = sc.frame_num;
            ctx->pull_url = sc.url;

            if (!load_models_for_stream(sc, ctx.get())) {
                std::cerr << "Stream " << sc.id << " has no valid models, skip." << std::endl;
                continue;
            }

            // 创建输出文件（按流区分）
            std::string out_path = std::string(OUT_VIDEO_PATH) + "." + sc.id;
            FILE* fp = fopen(out_path.c_str(), "w");
            if (fp) ctx->out_fp = fp;

            g_streams.push_back(std::move(ctx));
        }

        if (g_streams.empty()) {
            std::cerr << "No stream initialized, exit." << std::endl;
            return -1;
        }

        // 3. 初始化串口报警模块
        if (!SerialAlarm::getInstance().init("/dev/ttyS4")) {
            std::cerr << "WARNING: Failed to init serial alarm, alarm function disabled." << std::endl;
        }

        // 4. 创建全局线程池（容量按流数动态调整）
        int pool_size = std::max(4, (int)g_streams.size() * 2);
        g_pool = new ThreadPool(pool_size, pool_size);
        std::cout << "ThreadPool created with " << pool_size << " threads." << std::endl;

        // 4. 分别启动各流的 RTSP
        for (auto& s : g_streams) {
            std::cout << "Starting stream [" << s->stream_id << "] URL: " << s->pull_url << std::endl;
            process_video_rtsp(s.get());
        }

        signal(SIGINT, s_on_exit);
        signal(SIGTERM, s_on_exit);
        pause();
        printf("waiting finish\n");

        // 5. 清理资源
        for (auto& s : g_streams) {
            if (s->out_fp) {
                fflush(s->out_fp);
                fclose(s->out_fp);
            }
            if (s->decoder) {
                delete s->decoder;
                s->decoder = nullptr;
            }
            if (s->encoder) {
                delete s->encoder;
                s->encoder = nullptr;
            }
            release_media(&s->media);
            release_player(&s->player);
            for (auto* model : s->models) {
                model->release();
                delete model;
            }
        }
        g_streams.clear();

        if (g_pool) {
            delete g_pool;
            g_pool = nullptr;
        }

        SerialAlarm::getInstance().deinit();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
