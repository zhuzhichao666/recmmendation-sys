"""
推荐系统评估模块
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from src.utils import get_logger

logger = get_logger(__name__)


class Evaluator:
    """推荐系统评估器"""
    
    def __init__(self, test_data):
        self.test_data = test_data
    
    def rmse(self, recommender):
        """计算均方根误差"""
        errors = []
        
        for _, row in self.test_data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            true_rating = row['rating']
            pred_rating = recommender.predict(user_id, item_id)
            
            if pred_rating > 0:
                errors.append((true_rating - pred_rating) ** 2)
        
        if len(errors) == 0:
            return float('inf')
        
        return np.sqrt(np.mean(errors))
    
    def mae(self, recommender):
        """计算平均绝对误差"""
        errors = []
        
        for _, row in self.test_data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            true_rating = row['rating']
            pred_rating = recommender.predict(user_id, item_id)
            
            if pred_rating > 0:
                errors.append(np.abs(true_rating - pred_rating))
        
        if len(errors) == 0:
            return float('inf')
        
        return np.mean(errors)
    
    def precision(self, recommender, n=5, threshold=3):
        """
        计算精准度
        
        参数:
            recommender: 推荐器
            n: 推荐个数
            threshold: 好评阈值
        """
        precisions = []
        
        for user_id in self.test_data['user_id'].unique():
            user_test = self.test_data[self.test_data['user_id'] == user_id]
            good_items = set(user_test[user_test['rating'] >= threshold]['item_id'])
            
            if len(good_items) == 0:
                continue
            
            recommendations = recommender.recommend(user_id, n=n, exclude_rated=True)
            rec_items = set([item_id for item_id, _ in recommendations])
            
            hits = len(rec_items & good_items)
            precision = hits / n if n > 0 else 0
            precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0
    
    def recall(self, recommender, n=5, threshold=3):
        """
        计算召回率
        
        参数:
            recommender: 推荐器
            n: 推荐个数
            threshold: 好评阈值
        """
        recalls = []
        
        for user_id in self.test_data['user_id'].unique():
            user_test = self.test_data[self.test_data['user_id'] == user_id]
            good_items = set(user_test[user_test['rating'] >= threshold]['item_id'])
            
            if len(good_items) == 0:
                continue
            
            recommendations = recommender.recommend(user_id, n=n, exclude_rated=True)
            rec_items = set([item_id for item_id, _ in recommendations])
            
            hits = len(rec_items & good_items)
            recall = hits / len(good_items) if len(good_items) > 0 else 0
            recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0
    
    def f1(self, recommender, n=5, threshold=3):
        """计算F1分数"""
        precision = self.precision(recommender, n=n, threshold=threshold)
        recall = self.recall(recommender, n=n, threshold=threshold)
        
        if precision + recall == 0:
            return 0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def ndcg(self, recommender, n=5, threshold=3):
        """
        计算NDCG (Normalized Discounted Cumulative Gain)
        
        参数:
            recommender: 推荐器
            n: 推荐个数
            threshold: 好评阈值
        """
        ndcgs = []
        
        for user_id in self.test_data['user_id'].unique():
            user_test = self.test_data[self.test_data['user_id'] == user_id]
            good_items = set(user_test[user_test['rating'] >= threshold]['item_id'])
            
            if len(good_items) == 0:
                continue
            
            recommendations = recommender.recommend(user_id, n=n, exclude_rated=True)
            
            # 计算DCG
            dcg = 0
            for rank, (item_id, _) in enumerate(recommendations, 1):
                if item_id in good_items:
                    dcg += 1 / np.log2(rank + 1)
            
            # 计算理想DCG
            ideal_dcg = 0
            for rank in range(1, min(len(good_items), n) + 1):
                ideal_dcg += 1 / np.log2(rank + 1)
            
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
            ndcgs.append(ndcg)
        
        return np.mean(ndcgs) if ndcgs else 0
    
    def compare(self, recommenders, metrics=None, n=5):
        """
        比较多个推荐器
        
        参数:
            recommenders: 推荐器列表
            metrics: 评估指标列表
            n: 推荐个数
        
        返回:
            结果字典
        """
        if metrics is None:
            metrics = ['rmse', 'mae', 'precision', 'recall', 'f1', 'ndcg']
        
        results = {}
        
        for rec in recommenders:
            rec_name = rec.__class__.__name__
            logger.info(f"评估 {rec_name}...")
            
            results[rec_name] = {}
            
            for metric in metrics:
                if metric == 'rmse':
                    results[rec_name][metric] = self.rmse(rec)
                elif metric == 'mae':
                    results[rec_name][metric] = self.mae(rec)
                elif metric == 'precision':
                    results[rec_name][metric] = self.precision(rec, n=n)
                elif metric == 'recall':
                    results[rec_name][metric] = self.recall(rec, n=n)
                elif metric == 'f1':
                    results[rec_name][metric] = self.f1(rec, n=n)
                elif metric == 'ndcg':
                    results[rec_name][metric] = self.ndcg(rec, n=n)
        
        return results
    
    def plot_comparison(self, results, figsize=(12, 6)):
        """绘制比较图"""
        metrics = list(next(iter(results.values())).keys())
        
        # 分为两组: 预测指标和排序指标
        pred_metrics = ['rmse', 'mae']
        rank_metrics = [m for m in metrics if m not in pred_metrics]
        
        if pred_metrics:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # 绘制预测指标 (越小越好)
            pred_data = {rec: [results[rec].get(m, 0) for m in pred_metrics] 
                        for rec in results}
            self._plot_bars(axes[0], pred_data, pred_metrics, "预测指标 (越小越好)")
            
            # 绘制排序指标 (越大越好)
            rank_data = {rec: [results[rec].get(m, 0) for m in rank_metrics] 
                        for rec in results}
            self._plot_bars(axes[1], rank_data, rank_metrics, "排序指标 (越大越好)")
        else:
            fig, ax = plt.subplots(figsize=figsize)
            rank_data = {rec: [results[rec].get(m, 0) for m in rank_metrics] 
                        for rec in results}
            self._plot_bars(ax, rank_data, rank_metrics, "推荐指标")
        
        plt.tight_layout()
        plt.savefig('results/comparison.png', dpi=100, bbox_inches='tight')
        logger.info("比较图已保存到 results/comparison.png")
        plt.show()
    
    @staticmethod
    def _plot_bars(ax, data, metrics, title):
        """绘制柱状图"""
        x = np.arange(len(metrics))
        width = 0.2
        
        for i, (rec_name, values) in enumerate(data.items()):
            ax.bar(x + i * width, values, width, label=rec_name)
        
        ax.set_xlabel('指标')
        ax.set_ylabel('分数')
        ax.set_title(title)
        ax.set_xticks(x + width * (len(data) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def print_results(self, results):
        """打印结果"""
        print("\n" + "="*70)
        print("推荐系统评估结果")
        print("="*70)
        
        metrics = list(next(iter(results.values())).keys())
        
        # 打印表头
        header = "算法".ljust(20)
        for metric in metrics:
            header += f"{metric:>12}"
        print(header)
        print("-"*70)
        
        # 打印结果
        for rec_name, scores in results.items():
            row = rec_name.ljust(20)
            for metric in metrics:
                score = scores.get(metric, 0)
                row += f"{score:>12.4f}"
            print(row)
        
        print("="*70 + "\n")
