"""
数据加载和预处理模块
"""

import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import DATA_DIR, MOVIELENS_URL, DATA_ZIP_FILE, RANDOM_SEED
from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)


class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        ensure_dir(data_dir)
    
    def download_movielens(self):
        """下载MovieLens 100K数据集"""
        if os.path.exists(os.path.join(self.data_dir, 'u.data')):
            logger.info("数据集已存在，跳过下载")
            return
        
        logger.info("开始下载MovieLens 100K数据集...")
        
        try:
            urllib.request.urlretrieve(MOVIELENS_URL, DATA_ZIP_FILE)
            logger.info("下载完成，正在解压...")
            
            with zipfile.ZipFile(DATA_ZIP_FILE, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            # 移动文件到data目录
            ml_dir = os.path.join(self.data_dir, 'ml-100k')
            for file in os.listdir(ml_dir):
                src = os.path.join(ml_dir, file)
                dst = os.path.join(self.data_dir, file)
                if os.path.isfile(src):
                    os.rename(src, dst)
            
            os.rmdir(ml_dir)
            os.remove(DATA_ZIP_FILE)
            
            logger.info("数据集准备完成")
            
        except Exception as e:
            logger.error(f"下载失败: {e}")
            raise
    
    def load_movielens(self, test_size=0.2, random_state=RANDOM_SEED):
        """
        加载MovieLens数据集
        
        参数:
            test_size: 测试集比例
            random_state: 随机种子
        
        返回:
            train_data: 训练数据 DataFrame
            test_data: 测试数据 DataFrame
        """
        # 下载数据集
        self.download_movielens()
        
        # 加载评分数据
        logger.info("加载评分数据...")
        ratings = pd.read_csv(
            os.path.join(self.data_dir, 'u.data'),
            sep='\t',
            header=None,
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python'
        )
        
        # 加载电影信息
        logger.info("加载电影信息...")
        movies = pd.read_csv(
            os.path.join(self.data_dir, 'u.item'),
            sep='|',
            header=None,
            names=['item_id', 'title', 'release_date', 'video_release_date',
                   'imdb_url', 'unknown', 'action', 'adventure', 'animation',
                   'childrens', 'comedy', 'crime', 'documentary', 'drama',
                   'fantasy', 'film_noir', 'horror', 'musical', 'mystery',
                   'romance', 'sci_fi', 'thriller', 'war', 'western'],
            engine='python',
            encoding='latin-1'
        )
        
        # 加载用户信息
        logger.info("加载用户信息...")
        users = pd.read_csv(
            os.path.join(self.data_dir, 'u.user'),
            sep='|',
            header=None,
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
            engine='python'
        )
        
        # 合并数据
        logger.info("合并数据...")
        data = ratings.merge(movies[['item_id', 'title']], on='item_id')
        data = data.merge(users[['user_id', 'age', 'gender']], on='user_id')
        
        # 数据信息
        logger.info(f"数据集信息:")
        logger.info(f"  - 用户数: {data['user_id'].nunique()}")
        logger.info(f"  - 电影数: {data['item_id'].nunique()}")
        logger.info(f"  - 评分数: {len(data)}")
        logger.info(f"  - 评分范围: {data['rating'].min()}-{data['rating'].max()}")
        logger.info(f"  - 稀疏度: {1 - len(data) / (data['user_id'].nunique() * data['item_id'].nunique()):.4f}")
        
        # 分割训练集和测试集
        logger.info(f"分割数据集 (test_size={test_size})...")
        train_data, test_data = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            stratify=data['user_id']
        )
        
        logger.info(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")
        
        return train_data, test_data
    
    def get_user_item_matrix(self, data):
        """
        获取用户-物品矩阵
        
        参数:
            data: 评分数据 DataFrame
        
        返回:
            user_item_matrix: 用户-物品矩阵 (np.ndarray)
            user_ids: 用户ID列表
            item_ids: 物品ID列表
        """
        user_ids = sorted(data['user_id'].unique())
        item_ids = sorted(data['item_id'].unique())
        
        user_idx_map = {uid: idx for idx, uid in enumerate(user_ids)}
        item_idx_map = {iid: idx for idx, iid in enumerate(item_ids)}
        
        matrix = np.zeros((len(user_ids), len(item_ids)))
        
        for _, row in data.iterrows():
            user_idx = user_idx_map[row['user_id']]
            item_idx = item_idx_map[row['item_id']]
            matrix[user_idx, item_idx] = row['rating']
        
        return matrix, user_ids, item_ids
    
    def get_item_features(self):
        """获取电影特征（类别）"""
        movies = pd.read_csv(
            os.path.join(self.data_dir, 'u.item'),
            sep='|',
            header=None,
            names=['item_id', 'title', 'release_date', 'video_release_date',
                   'imdb_url', 'unknown', 'action', 'adventure', 'animation',
                   'childrens', 'comedy', 'crime', 'documentary', 'drama',
                   'fantasy', 'film_noir', 'horror', 'musical', 'mystery',
                   'romance', 'sci_fi', 'thriller', 'war', 'western'],
            engine='python',
            encoding='latin-1'
        )
        
        genre_columns = ['unknown', 'action', 'adventure', 'animation',
                         'childrens', 'comedy', 'crime', 'documentary', 'drama',
                         'fantasy', 'film_noir', 'horror', 'musical', 'mystery',
                         'romance', 'sci_fi', 'thriller', 'war', 'western']
        
        return movies[['item_id', 'title']].copy(), movies[genre_columns].values
