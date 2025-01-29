import logging
import os
import socket
import sys
import threading
import time
import numpy as np
import uhd
import yaml
import zmq
import queue
from datetime import datetime, timedelta

# 读取 YAML 配置文件
with open(os.path.join(os.path.dirname(__file__), "cal-settings.yml"), "r") as file:
    config = yaml.safe_load(file)

# 载入全局变量
RX_TX_SAME_CHANNEL = config["RX_TX_SAME_CHANNEL"]
CLOCK_TIMEOUT = config["CLOCK_TIMEOUT"]
INIT_DELAY = config["INIT_DELAY"]
RATE = config["RATE"]
LOOPBACK_RX_GAIN = config["LOOPBACK_RX_GAIN"]
REF_RX_GAIN = config["REF_RX_GAIN"]
FREQ = config["FREQ"]
CAPTURE_TIME = config["CAPTURE_TIME"]
server_ip = config["server_ip"]

# ZeroMQ 服务器用于同步多个 USRP
context = zmq.Context()
sync_socket = context.socket(zmq.SUB)
sync_socket.connect(f"tcp://{server_ip}:5557")
sync_socket.subscribe("")

# 日志系统
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console = logging.StreamHandler()
logger.addHandler(console)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] (%(threadName)-10s) %(message)s")
console.setFormatter(formatter)

# USRP 设备初始化
def setup_usrp(usrp):
    """配置 USRP 设备 RX 模式"""
    usrp.set_clock_source("external")
    usrp.set_time_source("external")
    usrp.set_rx_rate(RATE, 0)
    usrp.set_rx_gain(LOOPBACK_RX_GAIN, 0)  # 设置 RX 增益
    usrp.set_rx_freq(FREQ, 0)  # 设置接收频率
    usrp.set_rx_antenna("TX/RX", 0)  # 设定接收天线
    logger.info(f"USRP 设备初始化完成, 频率: {FREQ/1e6} MHz, 采样率: {RATE/1e6} MSps")

# 上行信道测量
def rx_ref(usrp, rx_streamer, quit_event, duration, result_queue, start_time=None):
    """测量上行信道（UL Channel Estimation），仅保存 IQ 数据"""
    logger.info("开始上行信道测量...")

    num_channels = rx_streamer.get_num_channels()
    max_samps_per_packet = rx_streamer.get_max_num_samps()
    buffer_length = int(duration * RATE * 2)  # 计算采样 buffer 大小
    iq_data = np.empty((num_channels, buffer_length), dtype=np.complex64)

    recv_buffer = np.zeros((num_channels, max_samps_per_packet), dtype=np.complex64)
    rx_md = uhd.types.RXMetadata()

    # 设定接收流
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = False
    stream_cmd.time_spec = start_time if start_time else uhd.types.TimeSpec(uhd.types.TimeSpec(usrp.get_time_now().get_real_secs() + INIT_DELAY))
    rx_streamer.issue_stream_cmd(stream_cmd)

    num_rx = 0
    try:
        while not quit_event.is_set():
            num_rx_i = rx_streamer.recv(recv_buffer, rx_md)
            if rx_md.error_code != uhd.types.RXMetadataErrorCode.none:
                logger.error(rx_md.error_code)
            else:
                if num_rx_i > 0:
                    samples = recv_buffer[:, :num_rx_i]
                    if num_rx + num_rx_i > buffer_length:
                        logger.error("超过 buffer 限制，数据丢失！")
                    else:
                        iq_data[:, num_rx:num_rx + num_rx_i] = samples
                        num_rx += num_rx_i
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("测量完成，停止 RX")
        rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))

        # 保存 IQ 数据
        file_name = f"dataset/usrp_csi_{socket.gethostname()}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.npz"
        np.savez_compressed(file_name, iq_samples=iq_data)
        logger.info(f"IQ 数据已保存: {file_name}")

        result_queue.put(file_name)

# 运行上行信道测量
def measure_pilot(usrp, rx_streamer, quit_event, result_queue, at_time=None):
    """执行上行信道测量"""
    logger.info("########### 开始上行信道估计 ###########")

    start_time = uhd.types.TimeSpec(at_time)
    logger.info(f"测量将在 {at_time} 秒后开始")

    usrp.set_rx_antenna("TX/RX", 1)  # 选择接收天线

    rx_thr = threading.Thread(
        target=rx_ref, args=(usrp, rx_streamer, quit_event, CAPTURE_TIME, result_queue, start_time)
    )
    rx_thr.start()

    time.sleep(CAPTURE_TIME + 1)  # 等待测量完成
    quit_event.set()
    rx_thr.join()

    usrp.set_rx_antenna("RX2", 1)  # 复位天线
    quit_event.clear()

# 主程序
def main():
    try:
        usrp = uhd.usrp.MultiUSRP("addr=192.168.10.2")  # 修改为你的 USRP IP
        setup_usrp(usrp)
        rx_streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))

        quit_event = threading.Event()
        result_queue = queue.Queue()

        start_time = get_current_time(usrp) + 5  # 5 秒后开始
        measure_pilot(usrp, rx_streamer, quit_event, result_queue, at_time=start_time)

        logger.info("测量完成，数据文件：" + result_queue.get())

    except Exception as e:
        logger.error(f"发生错误: {e}")
    finally:
        sys.exit(0)

if __name__ == "__main__":
    main()
