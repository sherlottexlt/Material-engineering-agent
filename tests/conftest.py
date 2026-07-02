"""pytest 全局配置

抑制 loguru / logging 的 stdout 输出，避免 sandbox 缓冲区溢出导致进程被 kill。
仅在测试运行时生效，不影响正常服务运行。
"""
import logging

# 抑制标准库 logging
logging.disable(logging.CRITICAL)

# 抑制 loguru（项目大量使用 loguru.logger）
try:
    from loguru import logger
    logger.remove()  # 移除所有 handler，禁止任何输出
except ImportError:
    pass
