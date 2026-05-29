"""
推荐系统配置文件
"""

# 数据配置
DATA_DIR = 'data'
MOVIELENS_URL = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
DATA_ZIP_FILE = 'data/ml-100k.zip'

# 模型配置
RANDOM_SEED = 42
TEST_SIZE = 0.2

# UserCF 配置
USER_CF_CONFIG = {
    'n_neighbors': 10,           # 相邻用户数
    'similarity_metric': 'cosine', # 相似度指标
}

# ItemCF 配置
ITEM_CF_CONFIG = {
    'n_neighbors': 10,           # 相邻物品数
    'similarity_metric': 'cosine',
}

# 矩阵分解配置
SVD_CONFIG = {
    'n_factors': 50,             # 隐向量维度
    'learning_rate': 0.01,       # 学习率
    'n_epochs': 100,             # 训练轮数
    'reg_param': 0.01,           # 正则化参数
    'verbose': True,
}

# 混合推荐配置
HYBRID_CONFIG = {
    'weights': {
        'user_cf': 0.3,
        'item_cf': 0.3,
        'svd': 0.4,
    },
    'n_recommendations': 5,
}

# 评估配置
EVAL_CONFIG = {
    'n_splits': 5,               # 交叉验证折数
    'n_recommendations': 5,      # 推荐个数
    'metrics': ['rmse', 'mae', 'precision', 'recall', 'f1', 'ndcg'],
}

# 可视化配置
PLOT_CONFIG = {
    'figsize': (12, 6),
    'style': 'seaborn-darkgrid',
    'dpi': 100,
}

# 日志配置
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
