// Config.h
#ifndef CONFIG_H
#define CONFIG_H

#include <vector>
#include <string>

struct ExtraModelConfig {
    std::string type;
    std::string mode;
    std::string textModel;
    std::vector<std::string> textDescriptions;
};

class Config {
public:
    static Config& getInstance();  // 获取配置实例
    const std::vector<std::string>& getClasses() const;  // 获取 classes 配置
    int getClassCount() const;  // 获取 classes 数量
    float getObjectThreshold() const;  // 获取 objectThreshold 配置
    float getNmsThreshold() const;  // 获取 nmsThreshold 配置
    const std::string& getMode() const;  // 获取 mode 配置
    const std::string& getImagePath() const;  // 获取图片存储路径配置
    const std::string& getStreamUrl() const;  // 获取视频流地址配置
    const std::string& getTid() const;  // 获取设备ID配置
    int getFrameNum() const;  // 每多少帧处理异常
    
    // YOLO-World 新增配置
    const std::string& getModelType() const;
    const std::string& getTextModel() const;
    const std::vector<std::string>& getTextDescriptions() const;
    
    // 多模型级联配置
    const std::vector<ExtraModelConfig>& getExtraModels() const;
    
private:
    Config();  // 构造函数私有化
    void loadConfig();  // 加载配置文件
    std::vector<std::string> classes;  // 存储配置数据
    float objectThreshold; // 存储 objectThreshold 配置
    float nmsThreshold;
    std::string mode;  // 存储 mode 配置
    std::string imagePath;  // 存储图片存储路径配置
    std::string streamUrl;  // 视频流地址配置
    std::string tid;  // 设备ID配置
    int frameNum;  // 每多少帧处理异常
    
    // YOLO-World 新增
    std::string modelType_;
    std::string textModel_;
    std::vector<std::string> textDescriptions_;
    
    // 附加模型列表（级联推理）
    std::vector<ExtraModelConfig> extraModels_;
};

#endif  // CONFIG_H
