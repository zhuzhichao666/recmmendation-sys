"""
工具函数
"""

import os
import logging
import numpy as np
import random
from config import LOG_LEVEL, LOG_FORMAT


def set_seed(seed=42):
    """设置随机种子以保证可复现性"""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_logger(name):
    """获取日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def ensure_dir(dir_path):
    """确保目录存在"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    
    if norm_product == 0:
        return 0
    
    return dot_product / norm_product


def euclidean_distance(vec1, vec2):
    """计算欧几里得距离"""
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def pearson_correlation(vec1, vec2):
    """计算皮尔逊相关系数"""
    mean1 = np.mean(vec1)
    mean2 = np.mean(vec2)
    
    numerator = np.sum((vec1 - mean1) * (vec2 - mean2))
    denominator = np.sqrt(np.sum((vec1 - mean1) ** 2) * np.sum((vec2 - mean2) ** 2))
    
    if denominator == 0:
        return 0
    
    return numerator / denominator


def get_top_k(scores, k=10):
    """获取最高的k个索引和分数"""
    if len(scores) <= k:
        indices = np.argsort(-scores)
        return indices, scores[indices]
    
    indices = np.argpartition(-scores, k-1)[:k]
    indices = indices[np.argsort(-scores[indices])]
    return indices, scores[indices]
