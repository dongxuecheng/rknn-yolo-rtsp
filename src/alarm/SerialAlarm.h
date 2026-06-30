#ifndef SERIAL_ALARM_H
#define SERIAL_ALARM_H

#include <string>
#include <mutex>
#include <unordered_map>

class SerialAlarm {
public:
    static SerialAlarm& getInstance();

    // 初始化串口，默认 /dev/ttyS4，9600 8N1
    bool init(const char* device = "/dev/ttyS4");
    void deinit();

    // 每流每帧调用：true=该流ROI内有人，false=无人
    void report(const std::string& stream_id, bool has_person);

private:
    SerialAlarm() = default;
    ~SerialAlarm() { deinit(); }

    void update();        // 扫描所有流状态，决定是否切换报警
    void sendOpen();      // 发送开启报警 Modbus 指令
    void sendClose();     // 发送关闭报警 Modbus 指令

    std::mutex mtx_;
    std::unordered_map<std::string, bool> stream_status_;
    bool alarm_on_ = false;
    int fd_ = -1;
};

#endif // SERIAL_ALARM_H
