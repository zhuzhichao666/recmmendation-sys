"""
推荐算法核心模块
"""

import numpy as np
from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.decomposition import TruncatedSVD
from src.utils import get_logger, cosine_similarity, get_top_k

logger = get_logger(__name__)


class BaseRecommender(ABC):
    """推荐算法基类"""
    
    def __init__(self):
        self.user_item_matrix = None
        self.user_ids = None
        self.item_ids = None
        self.user_id_map = None
        self.item_id_map = None
    
    @abstractmethod
    def fit(self, train_data):
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, user_id, item_id):
        """预测单个评分"""
        pass
    
    @abstractmethod
    def recommend(self, user_id, n=5, exclude_rated=True):
        """为用户推荐物品"""
        pass


class UserCF(BaseRecommender):
    """用户协同过滤推荐算法"""
    
    def __init__(self, n_neighbors=10, similarity_metric='cosine'):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.similarity_metric = similarity_metric
        self.user_similarities = None
    
    def fit(self, train_data):
        """训练模型"""
        logger.info(f"训练UserCF (neighbors={self.n_neighbors})...")
        
        from src.data_loader import DataLoader
        loader = DataLoader()
        
        self.user_item_matrix, self.user_ids, self.item_ids = loader.get_user_item_matrix(train_data)
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(self.item_ids)}
        
        # 计算用户相似度矩阵
        logger.info("计算用户相似度矩阵...")
        n_users = len(self.user_ids)
        self.user_similarities = np.zeros((n_users, n_users))
        
        for i in range(n_users):
            for j in range(i+1, n_users):
                sim = cosine_similarity(self.user_item_matrix[i], self.user_item_matrix[j])
                self.user_similarities[i, j] = sim
                self.user_similarities[j, i] = sim
        
        logger.info("UserCF训练完成")
    
    def predict(self, user_id, item_id):
        """预测评分"""
        if user_id not in self.user_id_map or item_id not in self.item_id_map:
            return 0
        
        user_idx = self.user_id_map[user_id]
        item_idx = self.item_id_map[item_id]
        
        # 找最相似的用户
        similarities = self.user_similarities[user_idx].copy()
        similarities[user_idx] = -1  # 排除自己
        
        top_indices = np.argsort(-similarities)[:self.n_neighbors]
        top_similarities = similarities[top_indices]
        top_similarities = top_similarities[top_similarities > 0]
        
        if len(top_similarities) == 0:
            return 0
        
        # 加权平均
        top_indices = top_indices[:len(top_similarities)]
        ratings = self.user_item_matrix[top_indices, item_idx]
        
        valid_mask = ratings > 0
        if np.sum(valid_mask) == 0:
            return 0
        
        valid_ratings = ratings[valid_mask]
        valid_similarities = top_similarities[valid_mask]
        
        prediction = np.sum(valid_ratings * valid_similarities) / np.sum(valid_similarities)
        return np.clip(prediction, 1, 5)
    
    def recommend(self, user_id, n=5, exclude_rated=True):
        """推荐物品"""
        if user_id not in self.user_id_map:
            logger.warning(f"用户 {user_id} 不在训练集中")
            return []
        
        user_idx = self.user_id_map[user_id]
        
        # 计算所有未评分物品的预测评分
        predictions = []
        for item_idx, item_id in enumerate(self.item_ids):
            if exclude_rated and self.user_item_matrix[user_idx, item_idx] > 0:
                continue
            
            pred_score = self.predict(user_id, item_id)
            if pred_score > 0:
                predictions.append((item_id, pred_score))
        
        # 排序并返回top-n
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]


class ItemCF(BaseRecommender):
    """物品协同过滤推荐算法"""
    
    def __init__(self, n_neighbors=10, similarity_metric='cosine'):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.similarity_metric = similarity_metric
        self.item_similarities = None
    
    def fit(self, train_data):
        """训练模型"""
        logger.info(f"训练ItemCF (neighbors={self.n_neighbors})...")
        
        from src.data_loader import DataLoader
        loader = DataLoader()
        
        self.user_item_matrix, self.user_ids, self.item_ids = loader.get_user_item_matrix(train_data)
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(self.item_ids)}
        
        # 计算物品相似度矩阵
        logger.info("计算物品相似度矩阵...")
        n_items = len(self.item_ids)
        self.item_similarities = np.zeros((n_items, n_items))
        
        for i in range(n_items):
            for j in range(i+1, n_items):
                sim = cosine_similarity(self.user_item_matrix[:, i], self.user_item_matrix[:, j])
                self.item_similarities[i, j] = sim
                self.item_similarities[j, i] = sim
        
        logger.info("ItemCF训练完成")
    
    def predict(self, user_id, item_id):
        """预测评分"""
        if user_id not in self.user_id_map or item_id not in self.item_id_map:
            return 0
        
        user_idx = self.user_id_map[user_id]
        item_idx = self.item_id_map[item_id]
        
        # 找最相似的物品
        similarities = self.item_similarities[item_idx].copy()
        similarities[item_idx] = -1  # 排除自己
        
        top_indices = np.argsort(-similarities)[:self.n_neighbors]
        top_similarities = similarities[top_indices]
        top_similarities = top_similarities[top_similarities > 0]
        
        if len(top_similarities) == 0:
            return 0
        
        # 加权平均
        top_indices = top_indices[:len(top_similarities)]
        ratings = self.user_item_matrix[user_idx, top_indices]
        
        valid_mask = ratings > 0
        if np.sum(valid_mask) == 0:
            return 0
        
        valid_ratings = ratings[valid_mask]
        valid_similarities = top_similarities[valid_mask]
        
        prediction = np.sum(valid_ratings * valid_similarities) / np.sum(valid_similarities)
        return np.clip(prediction, 1, 5)
    
    def recommend(self, user_id, n=5, exclude_rated=True):
        """推荐物品"""
        if user_id not in self.user_id_map:
            logger.warning(f"用户 {user_id} 不在训练集中")
            return []
        
        user_idx = self.user_id_map[user_id]
        
        predictions = []
        for item_idx, item_id in enumerate(self.item_ids):
            if exclude_rated and self.user_item_matrix[user_idx, item_idx] > 0:
                continue
            
            pred_score = self.predict(user_id, item_id)
            if pred_score > 0:
                predictions.append((item_id, pred_score))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]


class MatrixFactorization(BaseRecommender):
    """矩阵分解推荐算法 (SVD)"""
    
    def __init__(self, n_factors=50, learning_rate=0.01, n_epochs=100, reg_param=0.01):
        super().__init__()
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.reg_param = reg_param
        self.U = None
        self.V = None
    
    def fit(self, train_data):
        """训练模型"""
        logger.info(f"训练矩阵分解 (factors={self.n_factors}, epochs={self.n_epochs})...")
        
        from src.data_loader import DataLoader
        loader = DataLoader()
        
        self.user_item_matrix, self.user_ids, self.item_ids = loader.get_user_item_matrix(train_data)
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(self.item_ids)}
        
        n_users, n_items = self.user_item_matrix.shape
        
        # 随机初始化
        np.random.seed(42)
        self.U = np.random.normal(0, 0.01, (n_users, self.n_factors))
        self.V = np.random.normal(0, 0.01, (n_items, self.n_factors))
        
        # 训练
        for epoch in range(self.n_epochs):
            # 获取所有评分
            for user_idx in range(n_users):
                for item_idx in range(n_items):
                    if self.user_item_matrix[user_idx, item_idx] > 0:
                        rating = self.user_item_matrix[user_idx, item_idx]
                        pred = np.dot(self.U[user_idx], self.V[item_idx])
                        error = rating - pred
                        
                        # 梯度下降
                        self.U[user_idx] += self.learning_rate * (2 * error * self.V[item_idx] - self.reg_param * self.U[user_idx])
                        self.V[item_idx] += self.learning_rate * (2 * error * self.U[user_idx] - self.reg_param * self.V[item_idx])
            
            if (epoch + 1) % max(1, self.n_epochs // 5) == 0:
                logger.info(f"  Epoch {epoch+1}/{self.n_epochs} 完成")
        
        logger.info("矩阵分解训练完成")
    
    def predict(self, user_id, item_id):
        """预测评分"""
        if user_id not in self.user_id_map or item_id not in self.item_id_map:
            return 0
        
        user_idx = self.user_id_map[user_id]
        item_idx = self.item_id_map[item_id]
        
        prediction = np.dot(self.U[user_idx], self.V[item_idx])
        return np.clip(prediction, 1, 5)
    
    def recommend(self, user_id, n=5, exclude_rated=True):
        """推荐物品"""
        if user_id not in self.user_id_map:
            logger.warning(f"用户 {user_id} 不在训练集中")
            return []
        
        user_idx = self.user_id_map[user_id]
        
        predictions = []
        for item_idx, item_id in enumerate(self.item_ids):
            if exclude_rated and self.user_item_matrix[user_idx, item_idx] > 0:
                continue
            
            pred_score = self.predict(user_id, item_id)
            predictions.append((item_id, pred_score))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]


class HybridRecommender(BaseRecommender):
    """混合推荐算法"""
    
    def __init__(self, recommenders, weights):
        """
        参数:
            recommenders: 推荐器列表 (dict: {name: recommender})
            weights: 权重字典 (dict: {name: weight})
        """
        super().__init__()
        self.recommenders = recommenders
        self.weights = weights
    
    def fit(self, train_data):
        """训练所有推荐器"""
        logger.info("训练混合推荐器...")
        for name, rec in self.recommenders.items():
            logger.info(f"  训练 {name}...")
            rec.fit(train_data)
    
    def predict(self, user_id, item_id):
        """融合多个推荐器的预测"""
        predictions = []
        total_weight = 0
        
        for name, rec in self.recommenders.items():
            weight = self.weights.get(name, 1.0)
            pred = rec.predict(user_id, item_id)
            predictions.append(pred * weight)
            total_weight += weight
        
        if total_weight == 0:
            return 0
        
        return np.clip(np.sum(predictions) / total_weight, 1, 5)
    
    def recommend(self, user_id, n=5, exclude_rated=True):
        """融合多个推荐器的推荐"""
        recommendations = {}
        
        for name, rec in self.recommenders.items():
            weight = self.weights.get(name, 1.0)
            recs = rec.recommend(user_id, n=n*2, exclude_rated=exclude_rated)
            
            for item_id, score in recs:
                if item_id not in recommendations:
                    recommendations[item_id] = 0
                recommendations[item_id] += score * weight
        
        # 排序
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:n]
