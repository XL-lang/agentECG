import logging
import sys
import threading

# 创建一个模块级别的锁，用于保护初始化过程
_setup_lock = threading.Lock()

def get_logger(log_filename='app.log'):
    logger = logging.getLogger("my_project_logger")
    
    # 如果 logger 已经有 handler 了，说明已经初始化过，直接返回
    # 这样避免了多线程多次调用导致的重复 handler 问题
    if logger.handlers:
        return logger

    with _setup_lock:
        # 双重检查：获取锁之后再次检查，防止在等待锁的过程中已被其他线程初始化
        if logger.handlers:
            return logger
            
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-7s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 文件 Handler
        file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        # 控制台 Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger