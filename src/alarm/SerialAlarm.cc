#include "SerialAlarm.h"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

SerialAlarm& SerialAlarm::getInstance() {
    static SerialAlarm instance;
    return instance;
}

bool SerialAlarm::init(const char* device) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }

    fd_ = open(device, O_RDWR | O_NOCTTY | O_SYNC);
    if (fd_ < 0) {
        perror("SerialAlarm: open serial port failed");
        return false;
    }

    struct termios tty;
    memset(&tty, 0, sizeof(tty));
    if (tcgetattr(fd_, &tty) != 0) {
        perror("SerialAlarm: tcgetattr failed");
        close(fd_);
        fd_ = -1;
        return false;
    }

    // 9600 baud
    cfsetospeed(&tty, B9600);
    cfsetispeed(&tty, B9600);

    // 8N1
    tty.c_cflag &= ~PARENB;          // no parity
    tty.c_cflag &= ~CSTOPB;          // 1 stop bit
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;              // 8 bits
    tty.c_cflag |= CREAD | CLOCAL;   // enable read, ignore modem control

    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); // raw mode
    tty.c_iflag &= ~(IXON | IXOFF | IXANY | ICRNL | INLCR | IGNCR);
    tty.c_oflag &= ~OPOST;

    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 1; // 100ms timeout

    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
        perror("SerialAlarm: tcsetattr failed");
        close(fd_);
        fd_ = -1;
        return false;
    }

    tcflush(fd_, TCIOFLUSH);
    printf("SerialAlarm: %s opened @ 9600 8N1\n", device);
    return true;
}

void SerialAlarm::deinit() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (fd_ >= 0) {
        tcflush(fd_, TCIOFLUSH);
        close(fd_);
        fd_ = -1;
    }
    stream_status_.clear();
    alarm_on_ = false;
}

void SerialAlarm::report(const std::string& stream_id, bool has_person) {
    std::lock_guard<std::mutex> lock(mtx_);
    stream_status_[stream_id] = has_person;
    update();
}

void SerialAlarm::update() {
    bool any_person = false;
    for (const auto& kv : stream_status_) {
        if (kv.second) {
            any_person = true;
            break;
        }
    }

    if (any_person && !alarm_on_) {
        sendOpen();
        alarm_on_ = true;
        printf("[ALARM] ON (RS485)\n");
    } else if (!any_person && alarm_on_) {
        sendClose();
        alarm_on_ = false;
        printf("[ALARM] OFF (RS485)\n");
    }
}

void SerialAlarm::sendOpen() {
    if (fd_ < 0) return;
    unsigned char cmd[] = {
        0x01, 0x10, 0x00, 0x10, 0x00, 0x05, 0x0A,
        0x00, 0x03, 0x00, 0x01, 0x01, 0x01, 0x00, 0x02, 0x02, 0x01,
        0x49, 0x69
    };
    int n = write(fd_, cmd, sizeof(cmd));
    if (n != (int)sizeof(cmd)) {
        perror("SerialAlarm: sendOpen write failed");
    }
    tcdrain(fd_);
}

void SerialAlarm::sendClose() {
    if (fd_ < 0) return;
    unsigned char cmd[] = {
        0x01, 0x10, 0x00, 0x10, 0x00, 0x05, 0x0A,
        0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x02, 0x02, 0x01,
        0x5D, 0x99
    };
    int n = write(fd_, cmd, sizeof(cmd));
    if (n != (int)sizeof(cmd)) {
        perror("SerialAlarm: sendClose write failed");
    }
    tcdrain(fd_);
}
