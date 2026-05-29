"""
推荐系统主程序
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from config import RANDOM_SEED
from src import (
    DataLoader, UserCF, ItemCF, MatrixFactorization, HybridRecommender,
    Evaluator, set_seed, get_logger
)

logger = get_logger(__name__)

# 确保results目录存在
os.makedirs('results', exist_ok=True)


def main():
    """主函数"""
    print("\n" + "="*70)
    print("   推荐系统 - MovieLens 100K 数据集演示")
    print("="*70 + "\n")
    
    # 设置随机种子
    set_seed(RANDOM_SEED)
    logger.info(f"设置随机种子: {RANDOM_SEED}")
    
    # ========== 1. 数据加载 ==========
    print("📊 [1/5] 数据加载...")
    print("-" * 70)
    
    loader = DataLoader()
    train_data, test_data = loader.load_movielens(test_size=0.2)
    
    print(f"✓ 训练集: {len(train_data)} 条记录")
    print(f"✓ 测试集: {len(test_data)} 条记录\n")
    
    # ========== 2. 模型训练 ==========
    print("🤖 [2/5] 模型训练...")
    print("-" * 70)
    
    # 创建推荐器
    user_cf = UserCF(n_neighbors=10)
    item_cf = ItemCF(n_neighbors=10)
    svd = MatrixFactorization(n_factors=50, n_epochs=50)
    
    # 训练
    print("\n▶ 用户协同过滤 (UserCF)...")
    user_cf.fit(train_data)
    print("✓ UserCF 训练完成")
    
    print("▶ 物品协同过滤 (ItemCF)...")
    item_cf.fit(train_data)
    print("✓ ItemCF 训练完成")
    
    print("▶ 矩阵分解 (SVD)...")
    svd.fit(train_data)
    print("✓ SVD 训练完成")
    
    # 创建混合推荐器
    print("▶ 混合推荐器...")
    hybrid = HybridRecommender(
        recommenders={
            'UserCF': user_cf,
            'ItemCF': item_cf,
            'SVD': svd,
        },
        weights={'UserCF': 0.3, 'ItemCF': 0.3, 'SVD': 0.4}
    )
    hybrid.fit(train_data)
    print("✓ 混合推荐器训练完成\n")
    
    # ========== 3. 模型评估 ==========
    print("📈 [3/5] 模型评估...")
    print("-" * 70)
    
    evaluator = Evaluator(test_data)
    
    recommenders = [user_cf, item_cf, svd, hybrid]
    results = evaluator.compare(
        recommenders,
        metrics=['rmse', 'mae', 'precision', 'recall', 'f1', 'ndcg'],
        n=5
    )
    
    print()
    evaluator.print_results(results)
    
    # ========== 4. 推荐示例 ==========
    print("🎬 [4/5] 推荐结果示例...")
    print("-" * 70)
    
    # 获取电影信息
    movies_info = pd.read_csv(
        os.path.join('data', 'u.item'),
        sep='|',
        header=None,
        names=['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + 
              ['genre'] * 19,
        usecols=['item_id', 'title'],
        engine='python',
        encoding='latin-1'
    )
    movies_dict = dict(zip(movies_info['item_id'], movies_info['title']))
    
    # 选择几个用户进行推荐
    sample_users = [1, 50, 100]
    
    for user_id in sample_users:
        print(f"\n👤 用户 {user_id} 的推荐结果:")
        print("  " + "-" * 66)
        
        recommendations = hybrid.recommend(user_id, n=5, exclude_rated=True)
        
        for rank, (item_id, score) in enumerate(recommendations, 1):
            title = movies_dict.get(item_id, f"电影 {item_id}")
            print(f"  {rank}. {title:50} (评分预测: {score:.2f})")
    
    print("\n")
    
    # ========== 5. 可视化结果 ==========
    print("📊 [5/5] 生成可视化结果...")
    print("-" * 70)
    
    try:
        evaluator.plot_comparison(results)
        print("✓ 已保存对比图表到 results/comparison.png\n")
    except Exception as e:
        logger.warning(f"生成图表失败: {e}")
    
    # ========== 性能分析 ==========
    print("⚡ 性能分析")
    print("="*70)
    
    # 找出最好的算法
    best_rec = min(results.items(), key=lambda x: x[1].get('rmse', float('inf')))
    print(f"✓ 最低RMSE: {best_rec[0]} ({best_rec[1].get('rmse', 0):.4f})")
    
    best_rec = max(results.items(), key=lambda x: x[1].get('recall', 0))
    print(f"✓ 最高召回率: {best_rec[0]} ({best_rec[1].get('recall', 0):.4f})")
    
    best_rec = max(results.items(), key=lambda x: x[1].get('f1', 0))
    print(f"✓ 最高F1分数: {best_rec[0]} ({best_rec[1].get('f1', 0):.4f})")
    
    print("\n" + "="*70)
    print("✨ 推荐系统演示完成！")
    print("="*70 + "\n")
    
    return results


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"程序异常: {e}", exc_info=True)
        sys.exit(1)
