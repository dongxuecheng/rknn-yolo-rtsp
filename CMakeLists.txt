cmake_minimum_required(VERSION 3.10)

project(rknn_yolo_demo)
# 设置编译器
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置库架构
set(LIB_ARCH "aarch64")
set(TARGET_SOC "rk3588")

# 配置 OpenCV
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/opencv-linux-${LIB_ARCH}/share/OpenCV)
set(OpenCV_AIP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/opencv-linux-${LIB_ARCH}/include)
find_package(OpenCV REQUIRED)
message(STATUS "include path: ${OpenCV_AIP_INCLUDE}")
# 输出OpenCV信息
message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")

# 设置RKNN API路径
set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknpu2)
set(RKNN_API_INCLUDE_PATH ${RKNN_API_PATH}/include)
set(LIBRKNNRT ${RKNN_API_PATH}/${CMAKE_SYSTEM_NAME}/${LIB_ARCH}/librknnrt.so)

# 包含头文件
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${RKNN_API_INCLUDE_PATH}
    ${OpenCV_AIP_INCLUDE}
    ${CMAKE_CURRENT_LIST_DIR}/submodules/mpp/inc
    ${CMAKE_CURRENT_LIST_DIR}/submodules/mpp/osal/inc
    ${CMAKE_CURRENT_LIST_DIR}/submodules/mpp/utils
    ${CMAKE_CURRENT_LIST_DIR}/submodules/mpp/mpp/inc
    ${CMAKE_CURRENT_LIST_DIR}/submodules/mpp/mpp/base/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/submodules/ZLMediaKit/api/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rknn
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rknn/yolov8_pose
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rknn/yolo26
)

# 包含 3rdparty 和 utils 子目录
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utils utils.out)

# set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

# 获取 src 目录中的所有源文件
file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)

add_executable(${PROJECT_NAME}
    ${SRCS}  # 使用 src 目录中的所有源文件
	src/rknpu2/yolov8.cc
    src/conf/Config.cpp
)

target_link_libraries(${PROJECT_NAME}
    imageutils
    fileutils
    imagedrawing
    ${LIBRKNNRT}
    dl
    ${OpenCV_LIBS}
)


link_directories(
    mpp_libs
    /opt/app/slave/lib
)
add_executable(yolov8_stream_img
    src/yolov8_stream_img.cpp
    src/postprocess.cc
	src/rknpu2/yolov8.cc
    src/task/ThreadPool.cpp
    src/task/yolo_model.cpp
    src/rkmedia/utils/mpp_decoder.cpp
    src/rkmedia/utils/mpp_encoder.cpp
    src/conf/Config.cpp
    src/rknn/yolo_world/yolo_world.cc
    src/rknn/yolo_world/yolo_world_postprocess.cc
    src/rknn/yolov8_pose/yolov8_pose.cc
    src/rknn/yolov8_pose/yolov8_pose_postprocess.cc
    src/rknn/yolo26/yolo26.cc
    src/rknn/yolo26/yolo26_postprocess.cc
    src/alarm/SerialAlarm.cc
    src/rknn/clip_text/clip_text.cc
    src/rknn/tokenizer/clip_tokenizer.cpp
)

target_link_libraries(yolov8_stream_img
    imageutils
    fileutils
    imagedrawing
    ${LIBRKNNRT}
    dl
    ${OpenCV_LIBS}
    rockchip_mpp # 在mpp_libs下，包含了所有的mpp库
    utils # 在mpp_libs下，MPP工具库
    mk_api # 在mpp_libs下，ZLMediaKit API库
)


# 仅对 Linux 系统进行线程链接
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME}  ${OpenCV_LIBS} Threads::Threads)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
)

# 安装配置
install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/bus.jpg DESTINATION model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/coco_80_labels_list.txt DESTINATION model)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)
