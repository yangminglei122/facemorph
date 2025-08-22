import sys
import os
from datetime import datetime

try:
    # 创建logs目录
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # PyInstaller打包环境
        logs_dir = os.path.join(os.path.dirname(sys.executable), "logs")
    else:
        # 开发环境
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
    
    os.makedirs(logs_dir, exist_ok=True)

    # 创建日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"app_{timestamp}.log")

    # 保存原始的stdout和stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # 打开日志文件
    log_handle = open(log_file, 'w', buffering=1, encoding='utf-8')

    # 创建同时写入文件和控制台的类
    class TeeWriter:
        def __init__(self, file_handle, console_handle):
            self.file_handle = file_handle
            self.console_handle = console_handle

        def write(self, text):
            try:
                self.file_handle.write(text)
                self.file_handle.flush()
            except:
                pass
            try:
                self.console_handle.write(text)
                self.console_handle.flush()
            except:
                pass

        def flush(self):
            try:
                self.file_handle.flush()
            except:
                pass
            try:
                self.console_handle.flush()
            except:
                pass

    # 重定向stdout和stderr到日志文件和控制台
    sys.stdout = TeeWriter(log_handle, original_stdout)
    sys.stderr = TeeWriter(log_handle, original_stderr)
    
    print(f"日志系统已初始化，日志文件: {log_file}")

except Exception as e:
    # 如果初始化日志系统失败，至少确保原始输出可用
    pass