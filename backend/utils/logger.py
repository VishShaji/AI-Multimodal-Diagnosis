import logging
import os
from typing import Optional

def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)