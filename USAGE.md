# 使用指南

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行演示

```bash
python main.py
```

首次运行会自动下载MovieLens 100K数据集（约5MB）。

### 3. 查看结果

程序输出包括：
- ✓ 各算法性能对比
- ✓ 5个用户的推荐示例
- ✓ 对比图表（保存到 `results/comparison.png`）

## 📚 代码示例

### 基础使用

```python
from src.data_loader import DataLoader
from src.recommender import UserCF
from src.evaluation import Evaluator

# 1. 加载数据
loader = DataLoader()
train_data, test_data = loader.load_movielens()

# 2. 创建推荐器
recommender = UserCF(n_neighbors=10)
recommender.fit(train_data)

# 3. 获取推荐
recommendations = recommender.recommend(user_id=1, n=5)
for item_id, score in recommendations:
    print(f"电影 {item_id}: {score:.2f}")

# 4. 评估
evaluator = Evaluator(test_data)
rmse = evaluator.rmse(recommender)
recall = evaluator.recall(recommender, n=5)
print(f"RMSE: {rmse:.4f}, 召回率: {recall:.4f}")
```

### 比较多个算法

```python
from src.recommender import UserCF, ItemCF, MatrixFactorization

# 创建多个推荐器
user_cf = UserCF(n_neighbors=10)
item_cf = ItemCF(n_neighbors=10)
svd = MatrixFactorization(n_factors=50, n_epochs=100)

# 训练
for rec in [user_cf, item_cf, svd]:
    rec.fit(train_data)

# 比较
results = evaluator.compare(
    [user_cf, item_cf, svd],
    metrics=['rmse', 'mae', 'precision', 'recall'],
    n=5
)

evaluator.print_results(results)
evaluator.plot_comparison(results)
```

### 自定义推荐器

```python
from src.recommender import BaseRecommender
import numpy as np

class MyRecommender(BaseRecommender):
    def fit(self, train_data):
        # 你的训练逻辑
        pass
    
    def predict(self, user_id, item_id):
        # 预测评分 (1-5)
        return 3.5
    
    def recommend(self, user_id, n=5, exclude_rated=True):
        # 返回推荐列表 [(item_id, score), ...]
        return [(1, 4.5), (2, 4.3), (3, 4.1)]

# 使用
my_rec = MyRecommender()
my_rec.fit(train_data)
results = evaluator.compare([my_rec])
```

### 混合推荐

```python
from src.recommender import HybridRecommender

# 创建混合推荐器
hybrid = HybridRecommender(
    recommenders={
        'UserCF': user_cf,
        'ItemCF': item_cf,
        'SVD': svd,
    },
    weights={'UserCF': 0.3, 'ItemCF': 0.3, 'SVD': 0.4}
)

hybrid.fit(train_data)
recommendations = hybrid.recommend(user_id=1, n=5)
```

## 🔧 配置调整

编辑 `config.py` 修改配置：

```python
# UserCF 配置
USER_CF_CONFIG = {
    'n_neighbors': 20,  # 增加邻近用户数
    'similarity_metric': 'cosine',
}

# SVD 配置
SVD_CONFIG = {
    'n_factors': 100,  # 增加隐向量维度
    'learning_rate': 0.01,
    'n_epochs': 200,  # 增加训练轮数
}
```

## 📊 输出说明

### 评估指标

| 指标 | 说明 | 范围 | 越好 |
|------|------|------|------|
| RMSE | 均方根误差 | [0, ∞) | 越小 |
| MAE | 平均绝对误差 | [0, ∞) | 越小 |
| Precision | 精准度 | [0, 1] | 越大 |
| Recall | 召回率 | [0, 1] | 越大 |
| F1 | F1分数 | [0, 1] | 越大 |
| NDCG | 排序增益 | [0, 1] | 越大 |

### 推荐算法

| 算法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| UserCF | 考虑用户社交关系 | 计算复杂度高 | 用户相似性强 |
| ItemCF | 物品特征明确 | 新物品冷启动 | 物品间相似性强 |
| SVD | 捕捉隐特征 | 需要大量数据 | 通用场景 |
| 混合 | 综合多种优势 | 参数调优复杂 | 大规模系统 |

## 🐛 故障排除

### 问题1：下载数据失败

**原因**：网络连接问题

**解决**：
1. 检查网络连接
2. 手动下载：访问 https://grouplens.org/datasets/movielens/
3. 解压到 `data/` 目录

### 问题2：运行速度慢

**原因**：矩阵分解训练时间长

**解决**：
1. 减少 `SVD_CONFIG['n_epochs']`
2. 在 `main.py` 中注释掉SVD算法
3. 使用更小的数据集

### 问题3：内存不足

**原因**：计算相似度矩阵

**解决**：
1. 减少 `n_neighbors`
2. 使用稀疏矩阵
3. 增加系统内存

## 💡 改进建议

1. **增量学习**: 支持在线更新模型
2. **冷启动**: 实现内容推荐处理新用户/物品
3. **实时性**: 集成流处理框架
4. **可扩展性**: 使用分布式计算
5. **可解释性**: 添加推荐原因说明

---

有问题？查看 [README.md](README.md) 获取更多信息！
