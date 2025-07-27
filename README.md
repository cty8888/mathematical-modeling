# AQI空气质量预测系统

基于机器学习的空气质量指数(AQI)预测系统，支持多种污染物数据输入，提供精确的AQI预测和可视化分析。

## 项目概述

本项目是一个完整的AQI预测解决方案，包含数据处理、模型训练、预测应用和可视化分析。系统采用多种机器学习算法，对比分析后选择最佳模型进行AQI预测。

###  主要功能

- **AQI计算**：基于中国环保部标准的AQI计算器
- **机器学习预测**：支持线性回归、随机森林、LightGBM、CatBoost、神经网络等多种算法
- **智能模型选择**：自动对比模型性能并选择最佳模型
- **Web应用界面**：基于Streamlit的用户友好界面
- **数据可视化**：完整的模型性能分析和预测结果可视化
- **批量数据处理**：支持大量历史数据的处理和分析

##  项目结构

```
code/
├── aqi_calculator.py           # AQI计算器（独立使用）
├── aqi_model_training.py       # 机器学习模型训练脚本
├── aqi_predictor_app.py        # Streamlit Web应用
├── China_data/                 # 原始数据目录
│   └── 城市_20250101-20250719/ # 中国城市日数据(CSV格式)
├── China_data_CSV/             # 处理后的训练数据
│   ├── china_cities_aqi_daily.csv
│   └── target_cities_aqi_data.csv
├── model_outputs/              # 模型输出目录
│   ├── images/                 # 可视化图表
│   │   ├── base_models_comparison.png
│   │   ├── neural_network_comparison.png
│   │   └── final_model_analysis.png
│   └── models/                 # 训练好的模型
│       └── best_aqi_model.joblib
├── aqi_prediction_env/         # Python虚拟环境
└── requirements.txt            # 依赖包列表
```

##  快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone <repository-url>
cd code

# 创建虚拟环境（如果还没有）
python -m venv aqi_prediction_env

# 激活虚拟环境
# Windows:
aqi_prediction_env\Scripts\activate
# Linux/Mac:
source aqi_prediction_env/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 运行模型训练脚本
python aqi_model_training.py
```

训练过程将：
- 自动加载和预处理数据
- 训练多种机器学习模型
- 自动选择最佳性能模型
- 生成详细的性能分析报告
- 保存训练好的模型和可视化图表

### 3. 运行Web应用

```bash
# 启动Streamlit应用
streamlit run aqi_predictor_app.py
```

然后在浏览器中访问 `http://localhost:8501` 使用预测界面。

### 4. 使用AQI计算器（可选）

```bash
# 直接运行AQI计算器
python aqi_calculator.py
```

##  支持的污染物

| 污染物 | 单位 | 说明 |
|--------|------|------|
| PM2.5 | μg/m³ | 细颗粒物 |
| PM10 | μg/m³ | 可吸入颗粒物 |
| O3 | μg/m³ | 臭氧 |
| SO2 | μg/m³ | 二氧化硫 |
| NO2 | μg/m³ | 二氧化氮 |
| CO | mg/m³ | 一氧化碳 |

##  AQI等级标准

| AQI范围 | 等级 | 颜色 | 健康影响 |
|---------|------|------|----------|
| 0-50 | 优 | 🟢 绿色 | 空气质量令人满意，基本无空气污染 |
| 51-100 | 良 | 🟡 黄色 | 空气质量可接受，但某些污染物可能对极少数异常敏感人群健康有较弱影响 |
| 101-150 | 轻度污染 | 🟠 橙色 | 易感人群症状有轻度加剧，健康人群出现刺激症状 |
| 151-200 | 中度污染 | 🔴 红色 | 进一步加剧易感人群症状，可能对健康人群心脏、呼吸系统有影响 |
| 201-300 | 重度污染 | 🟣 紫色 | 心脏病和肺病患者症状显著加剧，运动耐受力降低，健康人群普遍出现症状 |
| >300 | 严重污染 | 🟤 褐红色 | 健康人群运动耐受力降低，有强烈症状，提前出现某些疾病 |

##  机器学习模型

本系统支持以下机器学习算法：

1. **线性回归 (Linear Regression)**
2. **随机森林 (Random Forest)**
3. **LightGBM**
4. **CatBoost**
5. **神经网络 (MLP Regressor)**
   - 基础神经网络
   - ReLU激活函数优化
   - 深层网络结构
   - 超参数优化版本

###  模型性能

系统会自动对比所有模型的性能指标：
- **R² 决定系数**：模型解释数据变异的能力
- **RMSE 均方根误差**：预测值与实际值的偏差
- **MAE 平均绝对误差**：预测误差的平均值
- **交叉验证分数**：模型泛化能力评估

##  数据处理

### 数据来源
- 中国主要城市2025年1-7月空气质量日数据
- 包含六种主要污染物的实时监测数据
- 数据已经过预处理和质量检验

### 数据预处理
- 缺失值处理
- 数据标准化
- 特征工程
- 数据验证和清理

##  Web应用界面

### 主要功能
- **实时预测**：输入污染物浓度，即时获得AQI预测
- **等级显示**：直观显示AQI等级和健康影响
- **首要污染物识别**：自动识别对AQI贡献最大的污染物
- **响应式设计**：适配不同屏幕尺寸

### 使用指南
1. 在输入框中输入各污染物浓度
2. 点击"预测AQI"按钮
3. 查看预测结果和健康建议
4. 了解首要污染物信息

##  可视化分析

系统自动生成三类分析图表：

### 1. 基础模型对比图 (`base_models_comparison.png`)
- 五种基础模型的R²、RMSE、MAE、交叉验证分数对比
- 帮助理解不同算法的性能差异

### 2. 神经网络对比图 (`neural_network_comparison.png`)
- 不同神经网络配置的性能对比
- 激活函数和网络结构优化效果分析

### 3. 最终模型分析图 (`final_model_analysis.png`)
- 预测值vs实际值散点图
- 残差分析
- 误差分布统计
- 模型性能详细分析

##  技术栈

- **Python 3.8+**
- **机器学习**：scikit-learn, LightGBM, CatBoost
- **数据处理**：pandas, numpy
- **可视化**：matplotlib, seaborn
- **Web界面**：Streamlit
- **模型保存**：joblib

##  依赖包

主要依赖包包括：
```
streamlit
pandas
numpy
scikit-learn
lightgbm
catboost
matplotlib
seaborn
joblib
tqdm
```

完整依赖列表请查看 `requirements.txt` 文件。

## 🔧 高级使用

### 自定义训练数据

1. 将新的训练数据放入 `China_data_CSV/` 目录
2. 确保数据格式包含必要的污染物列和AQI列
3. 重新运行训练脚本

### 模型调优

在 `aqi_model_training.py` 中可以调整：
- 模型参数
- 训练/测试集比例
- 交叉验证折数
- 特征选择策略

### 添加新模型

在 `train_models()` 方法中添加新的机器学习算法：
```python
models['新模型名称'] = YourNewModel()
```
