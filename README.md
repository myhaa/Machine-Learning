# 参考

1. [amazing-feature-engineering](https://github.com/ashishpatel26/Amazing-Feature-Engineering)


# 一个完整的机器学习项目步骤

1. 项目概述
2. 数据获取
   * [data_input/load_data.py]()
3. 数据探索
   * [data_eda/eda.py]()
4. 数据清洗
   * [data_eda/missing_data.py](): 缺失数据处理
   * [data_eda/outlier.py](): 异常值处理
   * [data_eda/rare_value.py](): 离散变量处理
5. 特征工程
   * [feature_engineering/discretization.py](): 连续变量离散化
        1. equal binning, 
        2. kmeans binning etc.
   * [feature_engineering/encoding.py](): 离散变量编码 
        1. onehot, 
        2. ordinal etc
   * [feature_engineering/scaling.py](): 连续变量归一化
        1. z-score, 
        2. min-max, 
        3. robust
   * [feature_engineering/transformation.py](): 连续变量特征变换: 
        1. log, sqrt, 
        2. box-cox etc
6. 特征选择
   * [feature_selection/selection_method.py](): 特征选择方法：
        1. 检测常数变量
        2. 检测相关性
        3. 特征选择：
           1. chi2, 
           2. decision_tree, 
           3. rf, gbdt etc.
7. 模型构建
8. 模型调优
9. 给出解决方案
10. 部署、监控、维护
