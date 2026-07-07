# RKNN YOLO 多流推理（rknn-yolo）

基于 [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo) 修改的 Rockchip NPU 目标检测、姿态估计推理框架。支持从 RTSP 视频流实时拉流、硬件解码、多模型级联推理，并自动保存检测结果。

---

## 目录

- [简介](#简介)
- [特性](#特性)
- [支持平台](#支持平台)
- [项目架构](#项目架构)
- [依赖与编译](#依赖与编译)
  - [前置依赖](#前置依赖)
  - [MPP 编译](#mpp-编译)
  - [ZLMediaKit 编译](#zlmediakit-编译)
  - [CMake 构建](#cmake-构建)
- [模型准备](#模型准备)
  - [模型导出（ONNX）](#模型导出onnx)
  - [模型转换（RKNN）](#模型转换rknn)
- [算法配置（algorithms.json）](#算法配置algorithmsjson)
- [流配置（config.json）](#流配置configjson)
- [运行](#运行)
  - [单图片推理 Demo](#单图片推理-demo)
  - [RTSP 多流推理](#rtsp-多流推理)
- [YOLOv5 说明](#yolov5-说明)
- [YOLO26 说明](#yolo26-说明)
- [目录结构](#目录结构)
- [参考链接](#参考链接)

---

## 简介

本项目在 rknn_model_zoo 的 YOLOv8 Demo 基础上进行扩展，新增以下能力：

- **RTSP 多路视频流并发处理**：通过 `ZLMediaKit` 拉取多路摄像头 RTSP 码流，使用 Rockchip `MPP` 进行硬件解码。
- **多模型级联推理**：单路视频流可同时挂载多个模型（如 YOLO11 + YOLO-World + YOLOv8-Pose + YOLO26 + YOLOv5），级联执行推理。
- **异步线程池**：解码后的帧通过线程池异步提交推理任务，避免阻塞解码线程。
- **结果保存**：检测到目标时自动绘制检测框并保存截图。
- **RGA 硬件加速**：使用 Rockchip RGA 进行 YUV420SP 到 RGB888 的高效格式转换。

---

## 特性

| 功能 | 说明 |
|------|------|
| **多路 RTSP** | 支持同时接入多路摄像头/视频流，通过 `config.json` 配置 |
| **多模型级联** | 每路流可配置多个模型，依次推理，独立阈值 |
| **模型类型** | 支持 `YOLO11`、`YOLO-World`、`YOLOv8-Pose`、`YOLO26`、`YOLOv5` |
| **硬件解码** | 基于 MPP 的 H.264 / H.265 硬件解码 |
| **硬件加速** | RGA 图像格式转换，NPU 模型推理 |
| **异步推理** | 线程池调度，避免解码帧堆积 |
| **可扩展保存逻辑** | 检测逻辑位于 `process_frame`，可按需自定义 |

---

## 支持平台

- `RK3562`
- `RK3566`
- `RK3568`
- `RK3576`
- `RK3588`

> 编译前请在 `CMakeLists.txt` 中设置对应的 `LIB_ARCH` 和 `TARGET_SOC`。

---

## 项目架构

```
rknn-yolo/
├── 3rdparty/               # 第三方依赖（OpenCV、librknnrt、librga、jpeg_turbo 等）
├── build/                  # CMake 构建输出目录
├── model/                  # 预置 RKNN 模型、标签文件、测试图片
├── mpp_libs/               # MPP 与 ZLMediaKit 编译产物（.a / .so）
├── src/
│   ├── main.cc             # 官方 Demo：单张图片推理
│   ├── rknn_yolo_stream.cpp   # 主程序：RTSP 多流推理
│   ├── postprocess.cc/h    # 后处理（NMS、解码）
│   ├── yolov8.h            # YOLO 数据结构定义
│   ├── conf/
│   │   ├── Config.cpp/h    # 旧版单流配置读取（兼容）
│   ├── rkmedia/
│   │   └── utils/
│   │       ├── mpp_decoder.cpp/h   # MPP 视频解码器封装
│   │       └── mpp_encoder.cpp/h   # MPP 视频编码器封装
│   ├── rknn/
│   │   ├── yolo_world/     # YOLO-World 模型推理与后处理
│   │   ├── yolov8_pose/    # YOLOv8-Pose 模型推理与后处理
│   │   ├── yolo26/         # YOLO26 模型推理与后处理
│   │   ├── yolov5/         # YOLOv5 模型推理与后处理
│   │   ├── clip_text/      # CLIP 文本模型（YOLO-World 配套）
│   │   └── tokenizer/      # CLIP Tokenizer
│   ├── rknpu1/             # RKNN NPU1 适配代码
│   ├── rknpu2/             # RKNN NPU2 适配代码
│   └── task/
│       ├── ThreadPool.cpp/h    # 线程池实现
│       └── yolo_model.cpp/h    # 多模型统一封装（YOLO11 / World / Pose / YOLO26 / YOLOv5）
├── submodules/
│   ├── mpp/                # Rockchip MPP 源码（硬件编解码）
│   └── ZLMediaKit/         # ZLMediaKit 源码（RTSP 拉流）
├── utils/                  # 通用工具库（图像、文件、绘制）
├── CMakeLists.txt
└── config.json             # 多流/多模型配置文件
```

---

## 依赖与编译

### 前置依赖

- Linux 交叉编译工具链（根据目标平台选择 aarch64 / armhf）
- [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2/tree/master/doc)（用于 ONNX → RKNN 模型转换）
- CMake >= 3.10
- OpenCV（已预置于 `3rdparty/opencv/`）

### MPP 编译

如需 RTSP 拉流解码，需编译 MPP 库：

```bash
cd submodules/mpp
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=./install
make -j$(nproc)
```

将生成的 `libutils.a` 复制到项目 `mpp_libs/` 目录下：

```bash
cp submodules/mpp/build/utils/libutils.a mpp_libs/
```

官方源码：https://github.com/rockchip-linux/mpp

### ZLMediaKit 编译

本项目使用 ZLMediaKit 作为流媒体客户端拉取 RTSP：

```bash
cd submodules/ZLMediaKit
mkdir build && cd build
cmake .. -DENABLE_API=ON
make -j$(nproc)
```

将编译好的 `libmk_api.so` 复制到 `mpp_libs/` 目录下：

```bash
cp submodules/ZLMediaKit/release/linux/Release/libmk_api.so mpp_libs/
```

官方源码：https://github.com/ZLMediaKit/ZLMediaKit

### CMake 构建

```bash
cd /root/rknn-yolo
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

编译成功后生成两个可执行文件：

- `rknn_yolo_demo` — 单张图片推理 Demo
- `rknn_yolo_stream` — RTSP 多流推理主程序

> **注意**：`CMakeLists.txt` 中默认设置 `LIB_ARCH=aarch64`、`TARGET_SOC=rk356X`，请根据实际芯片平台修改。

---

## 算法配置（algorithms.json）

RTSP 多流推理主程序 `rknn_yolo_stream` 在启动时会读取同目录下的 `algorithms.json`，用于配置四类业务算法：`no_helmet`（未戴安全帽）、`phone_in_zone`（作业区玩手机）、`person_intrusion`（人员禁入）、`absence`（人员离岗）。

### 完整示例

```json
{
    "global": {
        "upload_url": "http://36.7.84.146:28801/open/api/operate/upload",
        "level_id": 1,
        "log_dir": "/root/yaoan/rknn-yolo/logs",
        "alert_image_dir": "/root/yaoan/rknn-yolo/alerts",
        "upload_queue_size": 10,
        "upload_timeout_seconds": 10
    },
    "algorithms": [
        {
            "id": "no_helmet",
            "enabled": true,
            "algorithm_code": "0",
            "conf_threshold": 0.45,
            "cooldown_seconds": 60,
            "nvr_ip": "172.16.21.31",
            "channel": "1",
            "related_classes": {
                "person_model": "yolo26",
                "head_model": "yolov5",
                "person_class": "person",
                "head_class": "head",
                "helmet_class": "helmet"
            },
            "logic": "person_with_head_no_helmet"
        },
        {
            "id": "phone_in_zone",
            "enabled": true,
            "algorithm_code": "52",
            "conf_threshold": 0.45,
            "cooldown_seconds": 60,
            "nvr_ip": "172.16.21.31",
            "channel": "1",
            "fence_id": "whole_frame",
            "related_classes": {
                "person_model": "yolo26",
                "phone_model": "yolo11",
                "person_class": "person",
                "phone_class": "phone"
            },
            "logic": "phone_and_person_in_zone"
        },
        {
            "id": "person_intrusion",
            "enabled": true,
            "algorithm_code": "123",
            "conf_threshold": 0.45,
            "cooldown_seconds": 60,
            "nvr_ip": "172.16.21.31",
            "channel": "1",
            "fence_id": "whole_frame",
            "related_classes": {
                "person_model": "yolo26",
                "person_class": "person"
            },
            "logic": "person_in_zone"
        },
        {
            "id": "absence",
            "enabled": true,
            "algorithm_code": "44",
            "conf_threshold": 0.45,
            "cooldown_seconds": 60,
            "nvr_ip": "172.16.21.31",
            "channel": "1",
            "fence_id": "whole_frame",
            "absence_seconds": 10,
            "related_classes": {
                "person_model": "yolo26",
                "person_class": "person"
            },
            "logic": "no_person_in_zone_for_seconds"
        }
    ],
    "fences": [
        {
            "id": "whole_frame",
            "full_frame": true,
            "points": []
        },
        {
            "id": "work_zone",
            "full_frame": false,
            "points": [
                [100, 100],
                [400, 100],
                [400, 300],
                [100, 300]
            ]
        }
    ]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `global.upload_url` | String | 报警图片上传接口完整地址 |
| `global.level_id` | Int | 报警级别，默认 1 |
| `global.log_dir` | String | 算法日志存放目录 |
| `global.alert_image_dir` | String | 报警截图存放目录，按 `日期/算法名/` 组织 |
| `global.upload_queue_size` | Int | 上传队列长度，满时丢弃最旧任务 |
| `global.upload_timeout_seconds` | Int | 单次上传超时时间 |
| `algorithms` | Array | 四类算法配置 |
| `id` | String | 算法标识：`no_helmet`、`phone_in_zone`、`person_intrusion`、`absence` |
| `enabled` | Bool | 是否启用该算法 |
| `algorithm_code` | String | 平台接口的 `classIndex`，上传时作为报警类型 |
| `conf_threshold` | Float | 目标置信度阈值 |
| `cooldown_seconds` | Int | 两次报警之间的最小间隔 |
| `nvr_ip` | String | 上传参数中的设备/NVR IP |
| `channel` | String | 上传参数中的通道号 |
| `fence_id` | String | 关联的电子围栏 ID，`phone_in_zone` / `person_intrusion` / `absence` 必填 |
| `absence_seconds` | Int | **仅 `absence` 有效**：围栏内连续多少秒无人则触发离岗报警 |
| `related_classes` | Object | 算法需要的模型与类别映射 |
| `fences` | Array | 电子围栏列表 |
| `fences.id` | String | 围栏唯一标识 |
| `fences.full_frame` | Bool | `true` 表示全画面，此时 `points` 可空 |
| `fences.points` | Array | 多边形顶点，每个点为 `[x, y]`，坐标为原始画面像素坐标 |

### 电子围栏配置要点

- `full_frame: true` 表示算法在整个画面范围内判定。
- `full_frame: false` 时必须提供至少 3 个顶点，构成闭合多边形。
- 顶点坐标按视频原始分辨率填写，例如 1920×1080 的画面就按 `1920×1080` 标定。
- 报警截图会用黄色多边形自动画出非全画面的围栏区域。

### 四类算法逻辑简述

| 算法 | 触发条件 | 报警图中绘制的框 |
|------|---------|-----------------|
| `no_helmet` | 检测到 `person` 且匹配到 `head`，但没有匹配到 `helmet` | 该 `person` 框 + 对应 `head` 框（若有 `helmet` 也画出） |
| `phone_in_zone` | `person` 中心在围栏内，且与该 `person` 相交/包含的 `phone` | 所有触发报警的 `person` 框 + `phone` 框 |
| `person_intrusion` | `person` 中心在禁止进入的围栏内 | 所有闯入的 `person` 框 |
| `absence` | 围栏内连续 `absence_seconds` 秒未检测到 `person` | 不绘制检测框（因为无人），只保存当前画面 |

---

## 模型准备

### 模型导出（ONNX）

使用 airockchip 提供的 YOLO 训练与导出代码：

https://github.com/airockchip/ultralytics_yolov8

导出 ONNX 参考文档：  
https://github.com/airockchip/ultralytics_yolov8/blob/main/RKOPT_README.zh-CN.md

### 模型转换（RKNN）

参考rknn-model-zoo
```

将转换好的 `.rknn` 模型放入 `model/` 目录。

---

## 流配置（config.json）

多流/多模型配置通过 `config.json` 管理，支持每路流、每个模型独立参数：

```json
{
    "streams": [
        {
            "id": "A",
            "url": "rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101",
            "frameNum": 5,
            "saveDir": "/root/rknn-yolo/output/A",
            "models": [
                {
                    "type": "yolo11",
                    "mode": "/root/rknn-yolo/model/yolo11.rknn",
                    "classes": ["person", "car", ...],
                    "objectThreshold": 0.45,
                    "nmsThreshold": 0.25
                },
                {
                    "type": "yolo26",
                    "mode": "/root/yaoan/rknn-yolo/model/yolo26n-rk3588.rknn",
                    "classes": ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"],
                    "objectThreshold": 0.45,
                    "nmsThreshold": 0.45
                },
                {
                    "type": "yolov5",
                    "mode": "/root/rknn-yolo/model/yolov5s-rk3588.rknn",
                    "classes": ["person", "bicycle", "car", ...],
                    "objectThreshold": 0.45,
                    "nmsThreshold": 0.45
                }
            ]
        },
        {
            "id": "B",
            "url": "rtsp://admin:password@192.168.1.101:554/Streaming/Channels/101",
            "frameNum": 5,
            "saveDir": "/root/rknn-yolo/output/B",
            "models": [
                {
                    "type": "yolo11",
                    "mode": "/root/rknn-yolo/model/yolo11.rknn",
                    "classes": ["person", "car", ...],
                    "objectThreshold": 0.45,
                    "nmsThreshold": 0.25
                }
            ]
        }
    ]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `streams` | Array | 多路视频流配置 |
| `id` | String | 流唯一标识，用于输出目录命名 |
| `url` | String | RTSP 拉流地址 |
| `frameNum` | Int | 每多少帧执行一次推理（例如 5 表示每 5 帧处理 1 帧） |
| `saveDir` | String | 该流的检测结果保存目录 |
| `models` | Array | 该流挂载的模型列表，按顺序级联推理 |
| `type` | String | 模型类型：`yolo11`、`yolo_world`、`yolov8_pose`、`yolo26`、`yolov5` |
| `mode` | String | RKNN 模型文件绝对路径 |
| `textModel` | String | YOLO-World 配套的 CLIP 文本模型路径 |
| `textDescriptions` | Array | YOLO-World 的文本描述列表 |
| `classes` | Array | 该模型的类别名称列表（YOLO11 / Pose / YOLO26 / YOLOv5 必须配置） |
| `objectThreshold` | Float | 该模型的置信度阈值（覆盖全局默认值） |
| `nmsThreshold` | Float | 该模型的 NMS 阈值（覆盖全局默认值） |

---

## 运行

### 单图片推理 Demo

```bash
./build/rknn_yolo_demo model/yolo11.rknn model/bus.jpg
```

结果保存为 `out.png`。

### RTSP 多流推理

```bash
cd /root/rknn-yolo
./build/rknn_yolo_stream
```

程序会读取同目录下的 `config.json`，按配置拉起多路 RTSP 流并执行推理。

#### 输出示例

```
Loaded 2 stream(s) from config.
ThreadPool created with 4 threads.
Starting stream [A] URL: rtsp://admin:password@192.168.1.100:554/...
Starting stream [B] URL: rtsp://admin:password@192.168.1.101:554/...
[A] [TIMING] yolo11 infer: 12ms, objects: 3
[A] [TIMING] yolo26 infer: 8ms, objects: 2
[A] [TIMING] yolov5 infer: 10ms, objects: 3
[A] save_image: /root/rknn-yolo/output/A/A#1715073600#detected.png
```

#### 保存逻辑

当前默认逻辑：**只要检测到目标即保存截图**。  
如需修改保存条件或过滤类别，请编辑 `src/rknn_yolo_stream.cpp` 中的 `process_frame()` 函数。

### 单图片推理测试工具

除了 `rknn_yolo_demo`（仅支持 YOLO11）外，项目还提供了 `test_image` 工具，可快速验证 `yolo11 / yolov8_pose / yolo26 / yolov5` 模型在单张图片上的效果。

```bash
cd /root/yaoan/rknn-yolo/build
./test_image <model_path> <image_path> <model_type> <classes> [obj_thresh] [nms_thresh]
```

- `classes` 传逗号分隔的类别名，例如 `person,car`；也可传 `coco` 使用 COCO 80 默认类别。
- `obj_thresh` 默认 `0.25`，`nms_thresh` 默认 `0.45`。

示例：

```bash
# YOLO11 COCO 模型测试
./test_image /root/yaoan/rknn-yolo/model/yolo11.rknn /root/yaoan/rknn-yolo/model/bus.jpg yolo11 coco 0.45 0.25

# 自定义 phone 模型测试
./test_image /root/yaoan/rknn-yolo/model/phone-yolo11-1280.rknn /root/yaoan/rknn-yolo/model/head_phone.jpg yolo11 phone 0.15 0.45

# YOLOv5 测试
./test_image /root/yaoan/rknn-yolo/model/yolov5.rknn /root/yaoan/rknn-yolo/model/bus.jpg yolov5 coco 0.45 0.45

# YOLOv8-Pose 测试
./test_image /root/yaoan/rknn-yolo/model/yolov8_pose.rknn /root/yaoan/rknn-yolo/model/bus.jpg yolov8_pose person 0.45 0.45
```

结果会保存为当前目录下的 `out_test_image.png`。

---

## YOLOv5 说明

`YOLOv5` 是三头输出模型（P3/P4/P5），RKNN 输出为 3 个分支，分别对应 80×80、40×40、20×20 的 grid。后处理使用 anchor-based 解码，需通过 sigmoid 和 anchor 计算最终框坐标。

### 关键注意点

- **Anchor 固定**：后处理中写死了 YOLOv5 默认 anchor（P3: [10,13,16,30,33,23], P4: [30,61,62,45,59,119], P5: [116,90,156,198,373,326]），请确保导出的 RKNN 模型使用这些 anchor。
- **NMS 必须开启**：后处理中已集成标准 NMS，默认 `nmsThreshold=0.45`。
- **类别列表**：YOLOv5 使用 COCO 80 类，配置 `classes` 时必须按 COCO 顺序填写 80 个类别名。
- **输入格式**：模型输入为 RGB888，尺寸 640×640，通过 letterbox 缩放。

### 推荐阈值

| 参数 | 推荐值 |
|------|--------|
| `objectThreshold` | 0.45 |
| `nmsThreshold` | 0.45 |

---

## YOLO26 说明

`YOLO26` 是单头输出模型，RKNN 输出格式为 `[1, 84, 8400]`，channel-major，box 格式为 `xywh`。

### 关键注意点

- **NMS 必须开启**：虽然模型号称 "end-to-end"，但实际输出仍存在大量重叠框，需要在后处理中执行标准 NMS（默认 `nmsThreshold=0.45`）。
- **坐标映射**：后处理先将 `xywh` 转换到 `[0, 640]` 范围内并做 letterbox 反算，再映射回原图分辨率。
- **类别列表**：YOLO26 使用 COCO 80 类，配置 `classes` 时必须按 COCO 顺序填写 80 个类别名。

### 推荐阈值

| 参数 | 推荐值 |
|------|--------|
| `objectThreshold` | 0.45 |
| `nmsThreshold` | 0.45 |

---

## 目录结构

```
rknn-yolo/
├── 3rdparty/               # 第三方库（OpenCV、RKNN API、RGA、FFTW 等）
├── build/                  # 构建产物
├── model/                  # RKNN 模型、标签、测试图
│   ├── yolo11.rknn
│   ├── yolo26-rk3588.rknn
│   ├── yolov5s-rk3588.rknn
│   ├── yolo_world_v2s.rknn
│   ├── yolov8_pose.rknn
│   ├── clip_text.rknn
│   └── coco_80_labels_list.txt
├── mpp_libs/               # MPP 与 ZLMediaKit 编译产物
│   ├── libutils.a
│   └── libmk_api.so
├── src/                    # 全部源代码
│   ├── main.cc
│   ├── rknn_yolo_stream.cpp
│   ├── conf/
│   ├── rkmedia/
│   ├── rknn/               # 各模型后处理
│   ├── rknpu1/
│   ├── rknpu2/
│   └── task/
├── submodules/
│   ├── mpp/
│   └── ZLMediaKit/
├── tools/                  # 测试工具
│   └── test_image.cpp      # 单张图片多模型推理测试工具
├── utils/                  # imageutils、fileutils、imagedrawing
├── CMakeLists.txt
└── config.json
```

---

## 参考链接

- [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo)
- [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)
- [ultralytics_yolov8 (Rockchip 优化版)](https://github.com/airockchip/ultralytics_yolov8)
- [Rockchip MPP](https://github.com/rockchip-linux/mpp)
- [ZLMediaKit](https://github.com/ZLMediaKit/ZLMediaKit)

---

## License

本项目基于 rknn_model_zoo 修改，遵循相关开源协议。第三方库（OpenCV、MPP、ZLMediaKit、RKNN API 等）的许可证请参考各自官方仓库。
