//
// RS485 报警指令独立测试工具
// 用法: ./test_alarm
// 直接往 /dev/ttyS4 发送开启/关闭报警的 Modbus 指令
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>

static const unsigned char CMD_OPEN[]  = {
    0x01, 0x10, 0x00, 0x10, 0x00, 0x05, 0x0A,
    0x00, 0x03, 0x00, 0x01, 0x01, 0x01, 0x00, 0x02, 0x02, 0x01,
    0x49, 0x69
};

static const unsigned char CMD_CLOSE[] = {
    0x01, 0x10, 0x00, 0x10, 0x00, 0x05, 0x0A,
    0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x02, 0x02, 0x01,
    0x5D, 0x99
};

static int open_serial(const char* device)
{
    int fd = open(device, O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0) {
        perror("open");
        return -1;
    }

    struct termios tty;
    memset(&tty, 0, sizeof(tty));
    if (tcgetattr(fd, &tty) != 0) {
        perror("tcgetattr");
        close(fd);
        return -1;
    }

    cfsetospeed(&tty, B9600);
    cfsetispeed(&tty, B9600);

    tty.c_cflag &= ~PARENB;          // 无校验
    tty.c_cflag &= ~CSTOPB;          // 1 停止位
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;              // 8 数据位
    tty.c_cflag |= CREAD | CLOCAL;   // 使能读取，忽略调制解调器控制线

    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); // 原始模式
    tty.c_iflag &= ~(IXON | IXOFF | IXANY | ICRNL | INLCR | IGNCR);
    tty.c_oflag &= ~OPOST;

    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 1; // 100ms 超时

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        perror("tcsetattr");
        close(fd);
        return -1;
    }

    tcflush(fd, TCIOFLUSH);
    printf("Serial %s opened @ 9600 8N1\n", device);
    return fd;
}

static void send_cmd(int fd, const char* name, const unsigned char* cmd, size_t len)
{
    printf("Sending %s (%zu bytes): ", name, len);
    for (size_t i = 0; i < len; ++i) {
        printf("%02X ", cmd[i]);
    }
    printf("\n");

    ssize_t n = write(fd, cmd, len);
    if (n != (ssize_t)len) {
        perror("write");
    } else {
        tcdrain(fd); // 等待发送完成
        printf("  -> %s sent successfully\n", name);
    }
}

int main(int argc, char** argv)
{
    const char* device = (argc > 1) ? argv[1] : "/dev/ttyS4";

    int fd = open_serial(device);
    if (fd < 0) {
        fprintf(stderr, "Failed to open serial port %s\n", device);
        return 1;
    }

    // 发送开启报警指令
    send_cmd(fd, "OPEN_ALARM", CMD_OPEN, sizeof(CMD_OPEN));

    printf("\nAlarm is ON. Waiting 3 seconds...\n\n");
    sleep(3);

    // 发送关闭报警指令
    send_cmd(fd, "CLOSE_ALARM", CMD_CLOSE, sizeof(CMD_CLOSE));

    printf("\nAlarm is OFF. Test complete.\n");

    close(fd);
    return 0;
}
