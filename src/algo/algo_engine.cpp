#include "algo_engine.h"
#include "algo_logger.h"
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <math.h>
#include <set>

namespace rknn_yolo {

namespace {

void mkdir_p(const std::string &dir) {
    size_t pos = 0;
    while ((pos = dir.find('/', pos + 1)) != std::string::npos) {
        std::string sub = dir.substr(0, pos);
        if (!sub.empty() && access(sub.c_str(), F_OK) != 0) {
            mkdir(sub.c_str(), 0777);
        }
    }
    if (access(dir.c_str(), F_OK) != 0) {
        mkdir(dir.c_str(), 0777);
    }
}

std::string timestampFilename(const std::string &stream_id) {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
    localtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d%H%M%S");
    return stream_id + "_" + oss.str();
}

std::string currentDateDirName() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
    localtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d");
    return oss.str();
}

} // anonymous namespace

bool AlgorithmSettings::loadFromFile(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open algorithms config: " << path << std::endl;
        return false;
    }
    try {
        nlohmann::json j;
        file >> j;

        if (j.contains("global")) {
            const auto &g = j["global"];
            global.upload_url = g.value("upload_url", global.upload_url);
            global.level_id = g.value("level_id", global.level_id);
            global.log_dir = g.value("log_dir", global.log_dir);
            global.alert_image_dir = g.value("alert_image_dir", global.alert_image_dir);
            global.upload_queue_size = g.value("upload_queue_size", global.upload_queue_size);
            global.upload_timeout_seconds = g.value("upload_timeout_seconds", global.upload_timeout_seconds);
        }

        if (j.contains("fences") && j["fences"].is_array()) {
            for (const auto &f : j["fences"]) {
                Fence fence;
                fence.id = f.value("id", "");
                fence.full_frame = f.value("full_frame", false);
                if (f.contains("points") && f["points"].is_array()) {
                    for (const auto &p : f["points"]) {
                        if (p.is_array() && p.size() >= 2) {
                            fence.points.push_back({p[0].get<int>(), p[1].get<int>()});
                        }
                    }
                }
                if (!fence.id.empty()) fences[fence.id] = fence;
            }
        }

        if (j.contains("algorithms") && j["algorithms"].is_array()) {
            for (const auto &a : j["algorithms"]) {
                AlgorithmConfig ac;
                ac.id = a.value("id", "");
                ac.enabled = a.value("enabled", true);
                ac.algorithm_code = a.value("algorithm_code", "");
                ac.conf_threshold = a.value("conf_threshold", 0.45f);
                ac.cooldown_seconds = a.value("cooldown_seconds", 10);
                ac.nvr_ip = a.value("nvr_ip", "");
                ac.channel = a.value("channel", "1");
                ac.fence_id = a.value("fence_id", "");
                ac.absence_seconds = a.value("absence_seconds", 0);
                ac.logic = a.value("logic", "");

                if (a.contains("related_classes")) {
                    const auto &rc = a["related_classes"];
                    ac.related.person_model = rc.value("person_model", "");
                    ac.related.phone_model = rc.value("phone_model", "");
                    ac.related.head_model = rc.value("head_model", "");
                    ac.related.person_class = rc.value("person_class", "person");
                    ac.related.phone_class = rc.value("phone_class", "phone");
                    ac.related.head_class = rc.value("head_class", "head");
                    ac.related.helmet_class = rc.value("helmet_class", "helmet");
                }

                algorithms.push_back(ac);
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "Failed to parse algorithms config: " << e.what() << std::endl;
        return false;
    }
    return true;
}

const AlgorithmConfig* AlgorithmSettings::getAlgorithm(const std::string &id) const {
    for (const auto &a : algorithms) {
        if (a.id == id) return &a;
    }
    return nullptr;
}

const Fence* AlgorithmSettings::getFence(const std::string &id) const {
    auto it = fences.find(id);
    if (it != fences.end()) return &(it->second);
    return nullptr;
}

AlgoEngine::AlgoEngine() = default;
AlgoEngine::~AlgoEngine() { stop(); }

bool AlgoEngine::init(const std::string &config_path) {
    if (!settings_.loadFromFile(config_path)) return false;
    AlgoLogger::getInstance().init(settings_.global.log_dir);
    uploader_.start(settings_.global.upload_queue_size, settings_.global.upload_timeout_seconds);
    ALGO_LOG_INFO("AlgoEngine initialized: " + std::to_string(settings_.algorithms.size()) + " algorithms");
    return true;
}

void AlgoEngine::stop() { uploader_.stop(); }

void AlgoEngine::evaluate(const std::vector<ModelResult> &all_results, const cv::Mat &origin_rgb) {
    std::lock_guard<std::mutex> lock(evaluate_mtx_);
    extractBoxes(all_results);
    evaluateNoHelmet(origin_rgb);
    evaluatePhoneInZone(origin_rgb);
    evaluatePersonIntrusion(origin_rgb);
    evaluateAbsence(origin_rgb);
}

bool AlgoEngine::extractBoxes(const std::vector<ModelResult> &all_results) {
    persons_.clear();
    phones_.clear();
    heads_.clear();
    helmets_.clear();

    for (const auto &mr : all_results) {
        for (int i = 0; i < mr.results.count; ++i) {
            const auto &r = mr.results.results[i];
            std::string cls_name = mr.getClassName(r.cls_id);
            DetectedBox db{r.box, cls_name, r.prop, mr.model_name};
            if (cls_name == "person") persons_.push_back(db);
            else if (cls_name == "phone") phones_.push_back(db);
            else if (cls_name == "head") heads_.push_back(db);
            else if (cls_name == "helmet") helmets_.push_back(db);
        }
    }

    ALGO_LOG_INFO("extractBoxes: persons=" + std::to_string(persons_.size()) +
                  " phones=" + std::to_string(phones_.size()) +
                  " heads=" + std::to_string(heads_.size()) +
                  " helmets=" + std::to_string(helmets_.size()));
    return !persons_.empty();
}

float AlgoEngine::iou(const image_rect_t &a, const image_rect_t &b) {
    int x1 = std::max(a.left, b.left);
    int y1 = std::max(a.top, b.top);
    int x2 = std::min(a.right, b.right);
    int y2 = std::min(a.bottom, b.bottom);
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    int inter = (x2 - x1) * (y2 - y1);
    int area_a = (a.right - a.left) * (a.bottom - a.top);
    int area_b = (b.right - b.left) * (b.bottom - b.top);
    return (float)inter / (float)(area_a + area_b - inter + 1e-6f);
}

bool AlgoEngine::boxContainsPoint(const image_rect_t &box, int x, int y) {
    return x >= box.left && x <= box.right && y >= box.top && y <= box.bottom;
}

cv::Point AlgoEngine::boxCenter(const image_rect_t &box) {
    return cv::Point((box.left + box.right) / 2, (box.top + box.bottom) / 2);
}

bool AlgoEngine::triggerAlert(const AlgorithmConfig &algo, const std::string &reason,
                              const cv::Mat &origin_rgb,
                              const std::vector<DrawnBox> &boxes_to_draw) {
    auto now = std::chrono::system_clock::now();
    {
        auto it = last_alert_time_.find(algo.id);
        if (it != last_alert_time_.end()) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - it->second).count();
            if (elapsed < algo.cooldown_seconds) return false;
        }
    }

    if (origin_rgb.empty() || origin_rgb.cols <= 0 || origin_rgb.rows <= 0) {
        ALGO_LOG_WARN("Alert " + algo.id + ": empty origin image, skip drawing");
        return false;
    }

    std::string path;
    cv::Mat alert_img;
    std::vector<uchar> png_buf;
    {
        std::lock_guard<std::mutex> io_lock(g_rknn_yolo_image_io_mutex);

        if (origin_rgb.channels() == 3) cv::cvtColor(origin_rgb, alert_img, cv::COLOR_RGB2BGR);
        else alert_img = origin_rgb.clone();

        if (alert_img.empty() || alert_img.cols <= 0 || alert_img.rows <= 0 || alert_img.type() != CV_8UC3) {
            ALGO_LOG_WARN("Alert " + algo.id + ": invalid alert image after cvtColor");
            return false;
        }

        for (const auto &db : boxes_to_draw) {
            int x1 = std::max(db.box.left, 0);
            int y1 = std::max(db.box.top, 0);
            int x2 = std::min(db.box.right, alert_img.cols);
            int y2 = std::min(db.box.bottom, alert_img.rows);
            cv::rectangle(alert_img, cv::Point(x1, y1), cv::Point(x2, y2), db.color, 2);
            char text[256];
            snprintf(text, sizeof(text), "%s %.1f%%", db.label.c_str(), db.score * 100);
            cv::putText(alert_img, text, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, db.color, 2);
        }

        const Fence *fence = nullptr;
        if (!algo.fence_id.empty()) fence = settings_.getFence(algo.fence_id);
        if (fence && !fence->full_frame && !fence->points.empty()) {
            auto pts = fence->cvPoints();
            const cv::Point *ppts = pts.data();
            int npts = (int)pts.size();
            cv::polylines(alert_img, &ppts, &npts, 1, true, cv::Scalar(0, 255, 255), 2);
        }

        mkdir_p(settings_.global.alert_image_dir);
        std::string date_dir = settings_.global.alert_image_dir + "/" + currentDateDirName();
        std::string algo_dir = date_dir + "/" + algo.id;
        mkdir_p(date_dir);
        mkdir_p(algo_dir);
        path = algo_dir + "/" + algo.id + "_" + timestampFilename("A") + ".png";
        try {
            // 使用 PNG 格式保存，绕过 libjpeg 在 aarch64 静态链接下的崩溃问题
            if (!cv::imwrite(path, alert_img)) {
                ALGO_LOG_ERROR("Failed to write alert image: " + path);
                return false;
            }
            if (!cv::imencode(".png", alert_img, png_buf)) {
                ALGO_LOG_ERROR("Failed to encode alert image to PNG buffer");
                return false;
            }
        } catch (const std::exception &e) {
            ALGO_LOG_ERROR(std::string("cv::imwrite/imencode exception: ") + e.what());
            return false;
        }
    }

    // 只有在图片真正保存成功后才更新最后一次报警时间
    last_alert_time_[algo.id] = now;

    AlertContext ctx;
    ctx.algorithm_id = algo.id;
    ctx.algorithm_code = algo.algorithm_code;
    ctx.nvr_ip = algo.nvr_ip;
    ctx.channel = algo.channel;
    ctx.reason = reason;
    ctx.trigger_time = now;

    uploader_.addAlert(png_buf, ctx, settings_.global.upload_url, settings_.global.level_id);
    ALGO_LOG_INFO("Alert " + algo.id + ": " + reason + " image=" + path);
    return true;
}

void AlgoEngine::evaluateNoHelmet(const cv::Mat &origin_rgb) {
    const auto *algo = settings_.getAlgorithm("no_helmet");
    if (!algo || !algo->enabled || persons_.empty()) return;

    std::vector<DrawnBox> violation_boxes;
    int violation_count = 0;

    for (const auto &person : persons_) {
        if (person.score < algo->conf_threshold) continue;
        if (algo->related.person_model != person.model_name) continue;

        // find best matching head and helmet by IoU
        float best_head_iou = 0.0f;
        const DetectedBox *best_head = nullptr;
        float best_helmet_iou = 0.0f;
        const DetectedBox *best_helmet = nullptr;

        for (const auto &head : heads_) {
            if (head.model_name != algo->related.head_model) continue;
            float i = iou(person.box, head.box);
            if (i > best_head_iou) { best_head_iou = i; best_head = &head; }
        }
        for (const auto &helmet : helmets_) {
            if (helmet.model_name != algo->related.head_model) continue;
            float i = iou(person.box, helmet.box);
            if (i > best_helmet_iou) { best_helmet_iou = i; best_helmet = &helmet; }
        }

        bool has_helmet = (best_helmet && best_helmet_iou > 0.1f && best_helmet->score >= algo->conf_threshold);
        if (best_head && !has_helmet) {
            violation_boxes.push_back({person.box, "person", person.score, cv::Scalar(0, 0, 255)});
            violation_boxes.push_back({best_head->box, "head", best_head->score, cv::Scalar(255, 255, 0)});
            if (best_helmet) {
                violation_boxes.push_back({best_helmet->box, "helmet", best_helmet->score, cv::Scalar(0, 255, 0)});
            }
            ++violation_count;
        }
    }

    if (violation_count > 0) {
        triggerAlert(*algo, std::to_string(violation_count) + " person(s) with head but no helmet",
                     origin_rgb, violation_boxes);
    }
}

void AlgoEngine::evaluatePhoneInZone(const cv::Mat &origin_rgb) {
    const auto *algo = settings_.getAlgorithm("phone_in_zone");
    if (!algo || !algo->enabled || persons_.empty() || phones_.empty()) return;
    const Fence *fence = settings_.getFence(algo->fence_id);
    if (!fence) return;

    std::vector<DrawnBox> violation_boxes;
    bool triggered = false;

    // 使用集合避免同一个 person/phone 被重复绘制
    std::set<const DetectedBox*> drawn_persons;
    std::set<const DetectedBox*> drawn_phones;

    for (const auto &person : persons_) {
        if (person.score < algo->conf_threshold) continue;
        if (algo->related.person_model != person.model_name) continue;
        cv::Point pc = boxCenter(person.box);
        if (!fence->contains(pc.x, pc.y)) continue;

        for (const auto &phone : phones_) {
            if (phone.model_name != algo->related.phone_model) continue;
            if (phone.score < algo->conf_threshold) continue;
            if (iou(person.box, phone.box) > 0.0f || boxContainsPoint(person.box, boxCenter(phone.box).x, boxCenter(phone.box).y)) {
                if (drawn_persons.insert(&person).second) {
                    violation_boxes.push_back({person.box, "person", person.score, cv::Scalar(0, 0, 255)});
                }
                if (drawn_phones.insert(&phone).second) {
                    violation_boxes.push_back({phone.box, "phone", phone.score, cv::Scalar(255, 0, 255)});
                }
                triggered = true;
            }
        }
    }

    if (triggered) {
        triggerAlert(*algo, "person playing phone in zone", origin_rgb, violation_boxes);
    }
}

void AlgoEngine::evaluatePersonIntrusion(const cv::Mat &origin_rgb) {
    const auto *algo = settings_.getAlgorithm("person_intrusion");
    if (!algo || !algo->enabled || persons_.empty()) return;
    const Fence *fence = settings_.getFence(algo->fence_id);
    if (!fence) return;

    std::vector<DrawnBox> violation_boxes;
    for (const auto &person : persons_) {
        if (person.score < algo->conf_threshold) continue;
        if (algo->related.person_model != person.model_name) continue;
        cv::Point pc = boxCenter(person.box);
        if (fence->contains(pc.x, pc.y)) {
            violation_boxes.push_back({person.box, "person", person.score, cv::Scalar(0, 0, 255)});
        }
    }

    if (!violation_boxes.empty()) {
        triggerAlert(*algo, std::to_string(violation_boxes.size()) + " person(s) in forbidden zone",
                     origin_rgb, violation_boxes);
    }
}

void AlgoEngine::evaluateAbsence(const cv::Mat &origin_rgb) {
    const auto *algo = settings_.getAlgorithm("absence");
    if (!algo || !algo->enabled) return;
    const Fence *fence = settings_.getFence(algo->fence_id);
    if (!fence) return;

    auto now = std::chrono::system_clock::now();
    bool person_in_zone = false;
    for (const auto &person : persons_) {
        if (person.score < algo->conf_threshold) continue;
        if (algo->related.person_model != person.model_name) continue;
        cv::Point pc = boxCenter(person.box);
        if (fence->contains(pc.x, pc.y)) {
            person_in_zone = true;
            break;
        }
    }

    if (person_in_zone) {
        last_person_in_fence_time_[algo->fence_id] = now;
        absence_alert_triggered_[algo->fence_id] = false;
        return;
    }

    auto it = last_person_in_fence_time_.find(algo->fence_id);
    if (it == last_person_in_fence_time_.end()) {
        // 启动后第一次在该围栏内未检测到人，以当前时间开始计时
        last_person_in_fence_time_[algo->fence_id] = now;
        return;
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - it->second).count();
    if (elapsed >= algo->absence_seconds) {
        bool &triggered = absence_alert_triggered_[algo->fence_id];
        if (!triggered) {
            std::vector<DrawnBox> boxes;
            triggerAlert(*algo, "no person in zone for " + std::to_string(elapsed) + "s", origin_rgb, boxes);
            triggered = true;
        }
    }
}

} // namespace rknn_yolo

std::mutex g_rknn_yolo_image_io_mutex;

bool rknn_yolo_safe_imwrite(const std::string &path, const cv::Mat &image) {
    std::lock_guard<std::mutex> lock(g_rknn_yolo_image_io_mutex);
    try {
        return cv::imwrite(path, image);
    } catch (const std::exception &e) {
        std::cerr << "cv::imwrite exception for " << path << ": " << e.what() << std::endl;
        return false;
    }
}
