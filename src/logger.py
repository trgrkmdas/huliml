"""
Merkezi logging yapısı.
Tüm modüller bu logger'ı kullanarak log kayıtları oluşturur.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
from .config import get_config


class ColoredFormatter(logging.Formatter):
    """Renkli log formatı için formatter"""

    # ANSI renk kodları
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        # Log seviyesine göre renk ekle
        log_color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "MLProject",
    log_level: Optional[str] = None,
    log_to_file: bool = True,
    log_file_path: Optional[Path] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Logger'ı yapılandırır ve döndürür.

    Args:
        name: Logger adı
        log_level: Log seviyesi (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                   None ise config'den alınır
        log_to_file: Dosyaya log yazılsın mı?
        log_file_path: Log dosyası yolu (None ise otomatik oluşturulur)
        log_format: Log formatı (None ise varsayılan kullanılır)

    Returns:
        logging.Logger: Yapılandırılmış logger
    """
    config = get_config()

    # Logger oluştur
    logger = logging.getLogger(name)

    # Eğer zaten handler'lar varsa, tekrar ekleme
    if logger.handlers:
        return logger

    # Log seviyesini belirle
    if log_level is None:
        # Config'den al
        assert config.logging is not None  # Mypy için
        log_level = config.logging.log_level

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Log formatı
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Console handler (renkli)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = ColoredFormatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (renksiz)
    assert config.logging is not None  # Mypy için
    assert config.paths is not None  # Mypy için
    should_log_to_file = (
        log_to_file if log_to_file is not None else config.logging.log_to_file
    )
    if should_log_to_file:
        if log_file_path is None:
            # Config'den log dizinini al
            assert config.paths.project_root is not None  # Mypy için
            log_dir = config.paths.project_root / "logs"
            log_dir.mkdir(exist_ok=True)

            # Log dosyası adı: logs/MLProject_YYYY-MM-DD.log
            log_file_path = (
                log_dir / f"{name}_{datetime.now().strftime('%Y-%m-%d')}.log"
            )

        # Log dizinini oluştur
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


# Global logger instance
_logger: Optional[logging.Logger] = None


def get_logger(name: str = "MLProject") -> logging.Logger:
    """
    Global logger instance'ı döndürür.
    İlk çağrıda logger'ı yapılandırır.

    Args:
        name: Logger adı

    Returns:
        logging.Logger: Logger instance
    """
    global _logger

    if _logger is None:
        _logger = setup_logger(name)

    return _logger


# Modül seviyesinde logger fonksiyonları (kolay erişim için)
def debug(message: str, *args, **kwargs):
    """DEBUG seviyesinde log"""
    get_logger().debug(message, *args, **kwargs)


def info(message: str, *args, **kwargs):
    """INFO seviyesinde log"""
    get_logger().info(message, *args, **kwargs)


def warning(message: str, *args, **kwargs):
    """WARNING seviyesinde log"""
    get_logger().warning(message, *args, **kwargs)


def error(message: str, *args, **kwargs):
    """ERROR seviyesinde log"""
    get_logger().error(message, *args, **kwargs)


def critical(message: str, *args, **kwargs):
    """CRITICAL seviyesinde log"""
    get_logger().critical(message, *args, **kwargs)


def exception(message: str, *args, **kwargs):
    """ERROR seviyesinde log + exception traceback"""
    get_logger().exception(message, *args, **kwargs)
