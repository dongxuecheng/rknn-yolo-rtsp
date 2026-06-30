// Config.cpp
#include "Config.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>

using json = nlohmann::json;

Config &Config::getInstance()
{
    static Config instance; // 局部静态变量，保证全局只有一个实例
    return instance;
}

Config::Config()
{
    loadConfig();
}

void Config::loadConfig()
{
    std::ifstream file("config.json");
    if (!file.is_open())
    {
        std::cerr << "Could not open the config file!" << std::endl;
        return;
    }

    json config;
    file >> config;
    // 读取 classes 配置，若缺失则给定空列表
    if (config.contains("classes")){
        classes = config["classes"].get<std::vector<std::string>>();
    }else{
        std::cerr << "Warning: 'classes' key not found in config. Using default value." << std::endl;
        classes = {}; // 空的 classes 数组，或者你可以设置为默认值
    }

    // 读取 objectThreshold 配置，若缺失则给定默认值 0.75
    if (config.contains("objectThreshold")){
        objectThreshold = config["objectThreshold"].get<float>();
    }else{
        std::cerr << "Warning: 'objectThreshold' key not found in config. Using default value of 0.5." << std::endl;
        objectThreshold = 0.75f; // 默认值
    }

    // 读取 nmsThreshold 配置，若缺失则给定默认值 0.25
    if (config.contains("nmsThreshold")){
        nmsThreshold = config["nmsThreshold"].get<float>();
    }else{
        std::cerr << "Warning: 'nmsThreshold' key not found in config. Using default value of 0.25." << std::endl;
        nmsThreshold = 0.25f; // 默认值
    }

    // 加载 "mode" 配置
    if (config.contains("mode")){
        mode = config["mode"].get<std::string>();
    }

     // 加载 "imagePath" 配置
    if (config.contains("imagePath")) {
        imagePath = config["imagePath"].get<std::string>();
    }

    // 加载 "streamUrl" 配置
    if (config.contains("streamUrl")) {
        streamUrl = config["streamUrl"].get<std::string>(); 
    }

    // 加载 "tid" 配置
    if (config.contains("tid")) {
        tid = config["tid"].get<std::string>(); 
    }
    if (config.contains("frameNum")) {
        frameNum = config["frameNum"].get<int>();
    }

    // === YOLO-World 新增配置 ===
    if (config.contains("modelType")) {
        modelType_ = config["modelType"].get<std::string>();
    } else {
        modelType_ = "yolo11"; // 默认使用 yolo11
    }
    if (config.contains("textModel")) {
        textModel_ = config["textModel"].get<std::string>();
    }
    if (config.contains("textDescriptions")) {
        textDescriptions_ = config["textDescriptions"].get<std::vector<std::string>>();
    }

    // === 多模型级联配置 ===
    if (config.contains("extraModels") && config["extraModels"].is_array()) {
        for (const auto& item : config["extraModels"]) {
            ExtraModelConfig emc;
            if (item.contains("type")) emc.type = item["type"].get<std::string>();
            if (item.contains("mode")) emc.mode = item["mode"].get<std::string>();
            if (item.contains("textModel")) emc.textModel = item["textModel"].get<std::string>();
            if (item.contains("textDescriptions") && item["textDescriptions"].is_array()) {
                emc.textDescriptions = item["textDescriptions"].get<std::vector<std::string>>();
            }
            extraModels_.push_back(emc);
        }
    }

    file.close();  // 关闭文件流
}

const std::vector<std::string> &Config::getClasses() const
{
    return classes;
}

int Config::getClassCount() const
{
    return static_cast<int>(classes.size());
}

float Config::getObjectThreshold() const
{
    return objectThreshold;
}

float Config::getNmsThreshold() const
{
    return nmsThreshold;
}

const std::string &Config::getMode() const
{
    return mode;
}

const std::string &Config::getImagePath() const
{
    return imagePath;
}

const std::string &Config::getStreamUrl() const
{
    return streamUrl; 
}

const std::string &Config::getTid() const
{
    return tid;
}

int Config::getFrameNum() const
{
    return frameNum;
}

// === YOLO-World 新增实现 ===
const std::string &Config::getModelType() const
{
    return modelType_;
}

const std::string &Config::getTextModel() const
{
    return textModel_;
}

const std::vector<std::string> &Config::getTextDescriptions() const
{
    return textDescriptions_;
}

const std::vector<ExtraModelConfig>& Config::getExtraModels() const
{
    return extraModels_;
}
