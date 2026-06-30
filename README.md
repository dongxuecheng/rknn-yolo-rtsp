# RKNN YOLO 多流推理（rknn-yolo）

基于 [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo) 修改的 Rockchip NPU 目标检测、姿态估计与 RS485 报警推理框架。支持从 RTSP 视频流实时拉流、硬件解码、多模型级联推理、ROI 入侵检测，并通过 RS485 输出报警信号。

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
- [配置文件说明](#配置文件说明)
  - [ROI 与 RS485 报警](#roi-与-rs485-报警)
- [运行](#运行)
  - [RS485 报警](#rs485-报警)
  - [串口测试工具](#串口测试工具)
- [YOLO26N 说明](#yolo26n-说明)
- [目录结构](#目录结构)
- [参考链接](#参考链接)

---

## 简介

本项目在 rknn_model_zoo 的 YOLOv8 Demo 基础上进行扩展，新增以下能力：

- **RTSP 多路视频流并发处理**：通过 `ZLMediaKit` 拉取多路摄像头 RTSP 码流，使用 Rockchip `MPP` 进行硬件解码。
- **多模型级联推理**：单路视频流可同时挂载多个模型（如 YOLO11 + YOLO-World + YOLOv8-Pose + YOLO26N），级联执行推理。
- **ROI 入侵检测**：每路流可配置一个 ROI 区域，仅当 ROI 内检测到指定类别的目标时才触发报警。
- **RS485 报警输出**：检测到入侵时通过 `/dev/ttyS4` 发送 Modbus 指令开启报警；所有 ROI 均无人时自动发送关闭指令。
- **异步线程池**：解码后的帧通过线程池异步提交推理任务，避免阻塞解码线程。
- **结果保存**：检测到目标时自动绘制检测框并保存截图。
- **RGA 硬件加速**：使用 Rockchip RGA 进行 YUV420SP 到 RGB888 的高效格式转换。

---

## 特性

| 功能 | 说明 |
|------|------|
| **多路 RTSP** | 支持同时接入多路摄像头/视频流，通过 `config.json` 配置 |
| **多模型级联** | 每路流可配置多个模型，依次推理，独立阈值 |
| **模型类型** | 支持 `YOLO11`、`YOLO-World`、`YOLOv8-Pose`、`YOLO26N` |
| **ROI 检测** | 每路流支持一个矩形 ROI，可限定参与报警判定的模型白名单 |
| **RS485 报警** | 检测到 ROI 内 `person` 时自动发送 Modbus 开报警指令；全清时发送关报警指令 |
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
│   ├── yolov8_stream_img.cpp   # 主程序：RTSP 多流推理、ROI 报警、RS485 控制
│   ├── postprocess.cc/h    # 后处理（NMS、解码）
│   ├── yolov8.h            # YOLO 数据结构定义
│   ├── alarm/
│   │   ├── SerialAlarm.h   # RS485 报警单例类
│   │   └── SerialAlarm.cc  # 串口配置、Modbus 指令发送
│   ├── conf/
│   │   ├── Config.cpp/h    # 旧版单流配置读取（兼容）
│   ├── rkmedia/
│   │   └── utils/
│   │       ├── mpp_decoder.cpp/h   # MPP 视频解码器封装
│   │       └── mpp_encoder.cpp/h   # MPP 视频编码器封装
│   ├── rknn/
│   │   ├── yolo_world/     # YOLO-World 模型推理与后处理
│   │   ├── yolov8_pose/    # YOLOv8-Pose 模型推理与后处理
│   │   ├── yolo26/         # YOLO26N 模型推理与后处理
│   │   ├── clip_text/      # CLIP 文本模型（YOLO-World 配套）
│   │   └── tokenizer/      # CLIP Tokenizer
│   ├── rknpu1/             # RKNN NPU1 适配代码
│   ├── rknpu2/             # RKNN NPU2 适配代码
│   └── task/
│       ├── ThreadPool.cpp/h    # 线程池实现
│       └── yolo_model.cpp/h    # 多模型统一封装（YOLO11 / World / Pose / YOLO26N）
├── submodules/
│   ├── mpp/                # Rockchip MPP 源码（硬件编解码）
│   └── ZLMediaKit/         # ZLMediaKit 源码（RTSP 拉流）
├── tools/
│   └── test_alarm.cpp      # RS485 报警指令独立测试工具
├── utils/                  # 通用工具库（图像、文件、绘制）
├── CMakeLists.txt
└── config.json             # 多流/多模型/ROI/报警配置文件
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
- `yolov8_stream_img` — RTSP 多流推理主程序

> **注意**：`CMakeLists.txt` 中默认设置 `LIB_ARCH=aarch64`、`TARGET_SOC=rk356X`，请根据实际芯片平台修改。

---

## 模型准备

### 模型导出（ONNX）

使用 airockchip 提供的 YOLO 训练与导出代码：

https://github.com/airockchip/ultralytics_yolov8

导出 ONNX 参考文档：  
https://github.com/airockchip/ultralytics_yolov8/blob/main/RKOPT_README.zh-CN.md

### 模型转换（RKNN）

使用 `rknn-toolkit2` 将 ONNX 转换为 RKNN 模型：

```python
from rknn.api import RKNN

rknn = RKNN(verbose=True)
rknn.config(target_platform='rk3588')  # 根据实际平台修改
rknn.load_onnx(model='yolo11.onnx')
rknn.build(do_quantization=True, dataset='dataset.txt')
rknn.export_rknn('yolo11.rknn')
```

将转换好的 `.rknn` 模型放入 `model/` 目录。

---

## 配置文件说明

多流/多模型配置通过 `config.json` 管理，支持每路流、每个模型独立参数：

```json
{
    "streams": [
        {
            "id": "A",
            "url": "rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101",
            "frameNum": 5,
            "saveDir": "/root/rknn-yolo/output/A",
            "roi": {"x1": 100, "y1": 100, "x2": 540, "y2": 440},
            "roiModels": ["yolo11", "yolo26n"],
            "models": [
                {
                    "type": "yolo11",
                    "mode": "/root/rknn-yolo/model/yolo11.rknn",
                    "classes": ["person", "car", ...],
                    "objectThreshold": 0.45,
                    "nmsThreshold": 0.25
                },
                {
                    "type": "yolo26n",
                    "mode": "/root/rknn-yolo/model/yolo26n-rk3588.rknn",
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
            "roi": {"x1": 50, "y1": 50, "x2": 590, "y2": 430},
            "roiModels": ["yolo11"],
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
| `roi` | Object | ROI 区域，`x1/y1` 为左上角，`x2/y2` 为右下角 |
| `roiModels` | Array | 参与 ROI 报警判定的模型类型白名单，默认该流所有模型 |
| `models` | Array | 该流挂载的模型列表，按顺序级联推理 |
| `type` | String | 模型类型：`yolo11`、`yolo_world`、`yolov8_pose`、`yolo26n` |
| `mode` | String | RKNN 模型文件绝对路径 |
| `textModel` | String | YOLO-World 配套的 CLIP 文本模型路径 |
| `textDescriptions` | Array | YOLO-World 的文本描述列表 |
| `classes` | Array | 该模型的类别名称列表（YOLO11 / Pose / YOLO26N 必须配置） |
| `objectThreshold` | Float | 该模型的置信度阈值（覆盖全局默认值） |
| `nmsThreshold` | Float | 该模型的 NMS 阈值（覆盖全局默认值） |

### ROI 与 RS485 报警

- 当某个 `stream` 配置了 `roi` 时，`process_frame()` 会检测 ROI 内是否出现 `person` 类别。
- 只有 `roiModels` 中列出的模型类型会参与判定。
- 任意一路流 ROI 内检测到 `person`，`SerialAlarm` 会发送开启报警 Modbus 指令。
- 当所有流的 ROI 内均无 `person` 时，自动发送关闭报警指令。
- RS485 串口固定为 `/dev/ttyS4`，波特率 `9600`，数据位 `8`，无校验，停止位 `1`。

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
./build/yolov8_stream_img
```

程序会读取同目录下的 `config.json`，按配置拉起多路 RTSP 流并执行推理。

#### 输出示例

```
Loaded 2 stream(s) from config.
ThreadPool created with 4 threads.
Starting stream [A] URL: rtsp://admin:password@192.168.1.100:554/...
Starting stream [B] URL: rtsp://admin:password@192.168.1.101:554/...
[A] [TIMING] yolo11 infer: 12ms, objects: 3
[A] [TIMING] yolo26n infer: 8ms, objects: 2
[A] [ALARM] person_in_roi=1
[ALARM] ON (RS485)
[A] save_image: /root/rknn-yolo/output/A/A#1715073600#detected.png
```

#### 保存逻辑

当前默认逻辑：**该流配置的所有模型均检测到 `person` 类别时，才保存截图**。  
如需修改保存条件或报警类别，请编辑 `src/yolov8_stream_img.cpp` 中的 `process_frame()` 函数。

### RS485 报警

运行前请确保：

1. `/dev/ttyS4` 存在且当前用户有读写权限：
   ```bash
   ls -l /dev/ttyS4
   sudo chmod 666 /dev/ttyS4
   # 或永久加入 dialout 组
   sudo usermod -a -G dialout $USER
   ```
2. 串口在 `config.json` 中未禁用（目前硬编码为 `/dev/ttyS4`，初始化失败会打印警告，不会终止程序）。

### 串口测试工具

项目提供了一个不依赖主程序的独立 RS485 测试工具，方便快速验证硬件接线：

```bash
cd /root/rknn-yolo/tools
g++ -o test_alarm test_alarm.cpp
./test_alarm          # 默认 /dev/ttyS4
./test_alarm /dev/ttyUSB0   # 可指定其他串口
```

该工具会依次发送开启报警指令、等待 3 秒、发送关闭报警指令。如果设备正常，你应该能看到报警器先触发再释放。

---

## YOLO26N 说明

`YOLO26N` 是单头输出模型，RKNN 输出格式为 `[1, 84, 8400]`，channel-major，box 格式为 `xywh`。

### 关键注意点

- **NMS 必须开启**：虽然模型号称 "end-to-end"，但实际输出仍存在大量重叠框，需要在后处理中执行标准 NMS（默认 `nmsThreshold=0.45`）。
- **坐标映射**：后处理先将 `xywh` 转换到 `[0, 640]` 范围内并做 letterbox 反算，再映射回原图分辨率。
- **类别列表**：YOLO26N 使用 COCO 80 类，配置 `classes` 时必须按 COCO 顺序填写 80 个类别名。

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
│   ├── yolo26n-rk3588.rknn
│   ├── yolo_world_v2s.rknn
│   ├── yolov8_pose.rknn
│   ├── clip_text.rknn
│   └── coco_80_labels_list.txt
├── mpp_libs/               # MPP 与 ZLMediaKit 编译产物
│   ├── libutils.a
│   └── libmk_api.so
├── src/                    # 全部源代码
│   ├── alarm/              # RS485 报警模块
│   ├── main.cc
│   ├── yolov8_stream_img.cpp
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
│   └── test_alarm.cpp
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
