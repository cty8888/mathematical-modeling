import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import os
import time
from tqdm import tqdm
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
warnings.filterwarnings('ignore')
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class AQIPredictor:
    def __init__(self):
        self.scaler = StandardScaler()  # 数据标准化器
        self.model = None  # 存储训练模型
        self.feature_names = None #存储训练特征列名
        self.output_dir = "model_outputs"  # 输出文件夹名称
        self.images_dir = os.path.join(self.output_dir, "images")  # 图片文件夹
        self.models_dir = os.path.join(self.output_dir, "models")  # 模型文件夹
        
    def create_output_directory(self):
        """创建输出文件夹"""
        # 创建主文件夹
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"已创建输出文件夹: {self.output_dir}")
        
        # 创建图片子文件夹
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
            print(f"已创建图片文件夹: {self.images_dir}")
        
        # 创建模型子文件夹
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            print(f"已创建模型文件夹: {self.models_dir}")
        
        if os.path.exists(self.output_dir):
            print(f"所有输出文件夹已准备就绪")
    
    def load_data(self):
        print("正在加载数据...")
        # 加载中国城市AQI数据
        self.combined_data = pd.read_csv('China_data_CSV/target_cities_aqi_data.csv')
        
        print(f"数据形状: {self.combined_data.shape}")
        print(f"数据列名: {list(self.combined_data.columns)}")
        
        return self.combined_data
    
    def prepare_data(self):
        """数据预处理和特征工程"""
        print("\n" + "="*60)
        print("数据预处理")
        print("="*60)
        
        # 新数据格式的特征列
        pollutant_features = [
            'PM2.5',
            'PM10', 
            'O3',
            'SO2',
            'NO2',
            'CO'
        ]
        
        # AQI作为目标变量
        aqi_target = 'AQI'
        
        print(f"特征列: {pollutant_features}")
        print(f"目标列: {aqi_target}")
        
        # 检查所需列是否存在
        missing_features = [col for col in pollutant_features if col not in self.combined_data.columns]
        if missing_features:
            print(f"警告: 缺少特征列: {missing_features}")
            # 只使用存在的特征列
            pollutant_features = [col for col in pollutant_features if col in self.combined_data.columns]
        
        if aqi_target not in self.combined_data.columns:
            print(f"错误: 目标列 {aqi_target} 不存在")
            raise ValueError(f"目标列 {aqi_target} 不存在于数据中")
        
        # 准备训练数据
        print("正在准备训练数据...")
        self.X = self.combined_data[pollutant_features].apply(pd.to_numeric, errors='coerce')
        self.y = self.combined_data[aqi_target].apply(pd.to_numeric, errors='coerce')
        
        # 删除缺失值
        print("正在清理缺失值...")
        valid_rows = self.X.notna().all(axis=1) & self.y.notna()
        self.X = self.X[valid_rows]
        self.y = self.y[valid_rows]
        
        print(f"最终训练数据形状: X={self.X.shape}, y={self.y.shape}")
        
        self.feature_names = list(self.X.columns)
        
        # 数据标准化
        print("正在进行数据标准化...")
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        return self.X_scaled, self.y
    

    
    def train_models(self):
        """训练多个模型并比较性能"""
        print("\n" + "="*60)
        print("模型训练")
        print("="*60)
        # 分割训练集和测试集 (8:2)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42
        )
        print(f"训练集大小: {X_train.shape[0]} 样本")
        print(f"测试集大小: {X_test.shape[0]} 样本")
        models = {
            '线性回归': LinearRegression(),
            '随机森林': RandomForestRegressor(n_estimators=100, random_state=42),
            'LightGBM': LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1),
            'CatBoost': CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1, random_seed=42, verbose=False),
            '神经网络_基础': MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', alpha=0.001, learning_rate_init=0.001, max_iter=500, random_state=42)
        }

        self.model_results = {}
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        # 使用tqdm显示训练进度
        model_items = list(models.items())
        for i, (name, model) in enumerate(tqdm(model_items, desc="训练模型", unit="模型")):
            print(f"\n[{i+1}/{len(model_items)}] 训练 {name} 模型...")
            start_time = time.time()
            
            # 训练模型
            print("  正在训练...")
            model.fit(X_train, y_train)
            
            # 预测
            print("  正在预测...")
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # 评估指标
            train_mse = mean_squared_error(y_train, y_pred_train) #训练集均方误差
            test_mse = mean_squared_error(y_test, y_pred_test) #测试集均方误差
            train_r2 = r2_score(y_train, y_pred_train) #训练集R²
            test_r2 = r2_score(y_test, y_pred_test) #测试集R²
            test_mae = mean_absolute_error(y_test, y_pred_test) #测试集平均绝对误差
            
            # 交叉验证
            print("  正在进行交叉验证...")
            cv_scores = cross_val_score(model, self.X_scaled, self.y, cv=5, scoring='r2')
            
            # 保存结果
            self.model_results[name] = {
                'model': model,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred_test,
                'y_test': y_test
            }
            
            training_time = time.time() - start_time
            print(f"  训练集 R²: {train_r2:.4f}")
            print(f"  测试集 R²: {test_r2:.4f}")
            print(f"  测试集 MAE: {test_mae:.4f}")
            print(f"  交叉验证 R² (5折): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  训练耗时: {training_time:.2f}秒")
        
        # 选择基础最佳模型
        best_model_name = max(self.model_results.keys(), 
                            key=lambda x: self.model_results[x]['test_r2'])
        self.best_model = self.model_results[best_model_name]['model']
        
        print(f"\n基础最佳模型: {best_model_name}")
        
        return self.model_results
    
    def should_optimize_neural_network(self, top_n=3, score_threshold=0.02):
        """
        判断是否需要优化神经网络
        
        Parameters:
        - top_n: 如果神经网络在前N名，则值得优化
        - score_threshold: 如果神经网络与最佳模型的R²差距小于此值，则值得优化
        
        Returns:
        - bool: 是否需要优化神经网络
        - str: 判断原因
        """
        # 获取所有基础模型的R²分数
        model_scores = {name: result['test_r2'] for name, result in self.model_results.items()}
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 找出神经网络相关模型
        neural_models = [name for name in model_scores.keys() if '神经网络' in name]
        if not neural_models:
            return False, "没有找到神经网络基础模型"
        
        best_neural = max(neural_models, key=lambda x: model_scores[x])
        best_neural_score = model_scores[best_neural]
        best_overall_score = sorted_models[0][1]
        best_overall_model = sorted_models[0][0]
        
        # 判断1: 神经网络是否在前N名
        top_models = [model[0] for model in sorted_models[:top_n]]
        if best_neural in top_models:
            rank = top_models.index(best_neural) + 1
            return True, f"神经网络 '{best_neural}' 排名第{rank}，值得优化"
        
        # 判断2: 神经网络与最佳模型的差距是否很小
        score_gap = best_overall_score - best_neural_score
        if score_gap <= score_threshold:
            return True, f"神经网络 '{best_neural}' (R²={best_neural_score:.4f}) 与最佳模型 '{best_overall_model}' (R²={best_overall_score:.4f}) 差距仅{score_gap:.4f}，值得优化"
        
        # 不值得优化
        return False, f"神经网络 '{best_neural}' (R²={best_neural_score:.4f}) 与最佳模型 '{best_overall_model}' (R²={best_overall_score:.4f}) 差距{score_gap:.4f}过大，跳过优化"

    def optimize_neural_network(self):
        """在基础模型选定后，优化神经网络的激活函数和结构"""
        print("\n" + "="*60)
        print("神经网络激活函数优化")
        print("="*60)
        
        # 导入MLPRegressor
        from sklearn.neural_network import MLPRegressor
        
        # 测试ReLU激活函数
        neural_variants = {}
        neural_variants['神经网络_RELU'] = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42
        )
        print("✓ 测试RELU激活函数")
        
        # 测试不同网络结构
        structure_variants = {
            '神经网络_深层': MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=800,
                random_state=42
            ),
            '神经网络_优化': MLPRegressor(
                hidden_layer_sizes=(150, 100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate_init=0.002,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
        }
        
        neural_variants.update(structure_variants)
        print("✓ 测试深层网络结构")
        print("✓ 测试优化超参数")
        
        # 训练神经网络变体
        neural_items = list(neural_variants.items())
        for i, (name, model) in enumerate(tqdm(neural_items, desc="优化神经网络", unit="变体")):
            print(f"\n[{i+1}/{len(neural_items)}] 训练 {name} 模型...")
            start_time = time.time()
            
            # 训练模型
            print("  正在训练...")
            model.fit(self.X_train, self.y_train)
            
            # 预测
            print("  正在预测...")
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # 评估指标
            train_mse = mean_squared_error(self.y_train, y_pred_train)
            test_mse = mean_squared_error(self.y_test, y_pred_test)
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            
            # 交叉验证
            print("  正在进行交叉验证...")
            cv_scores = cross_val_score(model, self.X_scaled, self.y, cv=5, scoring='r2')
            
            # 添加到结果中
            self.model_results[name] = {
                'model': model,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred_test,
                'y_test': self.y_test
            }
            
            training_time = time.time() - start_time
            print(f"  训练集 R²: {train_r2:.4f}")
            print(f"  测试集 R²: {test_r2:.4f}")
            print(f"  测试集 MAE: {test_mae:.4f}")
            print(f"  交叉验证 R² (5折): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  训练耗时: {training_time:.2f}秒")
        
        # 重新选择最佳模型
        best_model_name = max(self.model_results.keys(), 
                            key=lambda x: self.model_results[x]['test_r2'])
        self.best_model = self.model_results[best_model_name]['model']
        
        print(f"\n优化后最佳模型: {best_model_name}")
        
        return neural_variants
    
    def evaluate_model(self):
        """最佳模型的测试集回归分析和可视化"""
        print("\n" + "="*60)
        print("模型性能比较总结")
        print("="*60)
        
        # 模型比较总结
        print(f"{'模型名称':<20} {'R²':<8} {'RMSE':<8} {'MAE':<8} {'CV均值':<8}")
        print("-" * 65)
        
        for name, result in self.model_results.items():
            rmse = np.sqrt(result['test_mse'])
            print(f"{name:<20} {result['test_r2']:<8.4f} {rmse:<8.2f} {result['test_mae']:<8.2f} {result['cv_mean']:<8.4f}")
        
        # 找出最佳模型
        best_model_name = max(self.model_results.keys(), 
                            key=lambda x: self.model_results[x]['test_r2'])
        print(f"\n最终最佳模型: {best_model_name} (基于测试集R²分数)")
        
        # 最佳模型详细分析
        print(f"\n{'='*60}")
        print(f"{best_model_name} - 详细测试集回归分析")
        print(f"{'='*60}")
        
        best_result = self.model_results[best_model_name]
        y_test = best_result['y_test']
        y_pred = best_result['predictions']
        
        # 基本回归指标
        r2 = best_result['test_r2']
        mse = best_result['test_mse']
        mae = best_result['test_mae']
        rmse = np.sqrt(mse)
        
        print(f"回归评估指标:")
        print(f"  R² 决定系数: {r2:.4f}")
        print(f"  均方误差 (MSE): {mse:.4f}")
        print(f"  均方根误差 (RMSE): {rmse:.4f}")
        print(f"  平均绝对误差 (MAE): {mae:.4f}")
        
        # 计算预测精度
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        print(f"  平均绝对百分比误差 (MAPE): {mape:.2f}%")
        
        # 残差统计
        residuals = y_test - y_pred
        print(f"\n残差统计:")
        print(f"  残差均值: {residuals.mean():.4f}")
        print(f"  残差标准差: {residuals.std():.4f}")
        print(f"  残差最小值: {residuals.min():.4f}")
        print(f"  残差最大值: {residuals.max():.4f}")
        
        # 预测范围分析
        print(f"\n预测值统计:")
        print(f"  实际值范围: [{y_test.min():.2f}, {y_test.max():.2f}]")
        print(f"  预测值范围: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
        print(f"  实际值均值: {y_test.mean():.2f}")
        print(f"  预测值均值: {y_pred.mean():.2f}")
        
        # 计算预测准确性（在不同误差范围内的样本比例）
        abs_errors = np.abs(y_test - y_pred)
        within_5 = np.sum(abs_errors <= 5) / len(abs_errors) * 100
        within_10 = np.sum(abs_errors <= 10) / len(abs_errors) * 100
        within_20 = np.sum(abs_errors <= 20) / len(abs_errors) * 100
        
        print(f"\n预测准确性分布:")
        print(f"  误差 ≤ 5 AQI点: {within_5:.1f}% 的样本")
        print(f"  误差 ≤ 10 AQI点: {within_10:.1f}% 的样本")
        print(f"  误差 ≤ 20 AQI点: {within_20:.1f}% 的样本")
        
        # 交叉验证结果
        print(f"\n交叉验证 (5折):")
        print(f"  平均 R²: {best_result['cv_mean']:.4f}")
        print(f"  标准差: {best_result['cv_std']:.4f}")
        print(f"  置信区间 (95%): [{best_result['cv_mean'] - 1.96*best_result['cv_std']:.4f}, {best_result['cv_mean'] + 1.96*best_result['cv_std']:.4f}]")
        
        # 特征重要性分析（如果是树模型）
        if hasattr(self.best_model, 'feature_importances_'):
            print(f"\n{best_model_name} - 特征重要性:")
            importances = self.best_model.feature_importances_
            feature_importance = pd.DataFrame({
                '特征': self.feature_names,
                '重要性': importances
            }).sort_values('重要性', ascending=False)
            
            for _, row in feature_importance.iterrows():
                print(f"  {row['特征']}: {row['重要性']:.4f}")
        

        
        # 最终总结
        print(f"\n{'='*60}")
        print("最佳模型性能总结")
        print(f"{'='*60}")
        print(f"最佳模型: {best_model_name}")
        print(f"测试集 R² 分数: {best_result['test_r2']:.4f}")
        print(f"测试集 RMSE: {rmse:.2f}")
        print(f"测试集 MAE: {best_result['test_mae']:.2f}")
        print(f"预测精度 (误差≤10 AQI点): {within_10:.1f}% 的测试样本")

    def create_base_models_comparison(self):

        base_models = ['线性回归', '随机森林', '神经网络_基础', 'LightGBM', 'CatBoost']
        # 提取数据
        r2_scores = [self.model_results[name]['test_r2'] for name in base_models if name in self.model_results]
        rmse_scores = [np.sqrt(self.model_results[name]['test_mse']) for name in base_models if name in self.model_results]
        mae_scores = [self.model_results[name]['test_mae'] for name in base_models if name in self.model_results]
        cv_scores = [self.model_results[name]['cv_mean'] for name in base_models if name in self.model_results]
        # 过滤存在的模型名称
        existing_models = [name for name in base_models if name in self.model_results]
        plt.figure(figsize=(16, 12))
        
        # 1. R²分数对比
        plt.subplot(2, 2, 1)
        bars1 = plt.bar(existing_models, r2_scores, color='skyblue', alpha=0.8)
        plt.title('$R^2$ 分数对比', fontsize=16, fontweight='bold')
        plt.ylabel('$R^2$ Score', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        # 调整y轴范围
        y_min = max(0, min(r2_scores) - 0.05)
        y_max = min(1, max(r2_scores) + 0.05)
        plt.ylim(y_min, y_max)
        # 添加数值标签，统一放在图表上方固定位置
        label_y = y_max - (y_max - y_min) * 0.15  # 距离顶部15%的位置
        for i, (bar, score) in enumerate(zip(bars1, r2_scores)):
            plt.text(bar.get_x() + bar.get_width()/2, label_y, 
                    f'{score:.4f}', ha='center', va='center', fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        plt.grid(True, alpha=0.3, axis='y')
        
        # 2. RMSE对比
        plt.subplot(2, 2, 2)
        bars2 = plt.bar(existing_models, rmse_scores, color='lightcoral', alpha=0.8)
        plt.title('RMSE 对比', fontsize=16, fontweight='bold')
        plt.ylabel('RMSE', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        # 调整y轴范围
        y_max_rmse = max(rmse_scores) * 1.1
        plt.ylim(0, y_max_rmse)
        # 添加数值标签，统一放在图表上方固定位置
        label_y_rmse = y_max_rmse * 0.9  # 距离顶部10%的位置
        for bar, score in zip(bars2, rmse_scores):
            plt.text(bar.get_x() + bar.get_width()/2, label_y_rmse, 
                    f'{score:.2f}', ha='center', va='center', fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        plt.grid(True, alpha=0.3, axis='y')
        
        # 3. MAE对比
        plt.subplot(2, 2, 3)
        bars3 = plt.bar(existing_models, mae_scores, color='lightgreen', alpha=0.8)
        plt.title('MAE 对比', fontsize=16, fontweight='bold')
        plt.ylabel('MAE', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        # 调整y轴范围
        y_max_mae = max(mae_scores) * 1.1
        plt.ylim(0, y_max_mae)
        # 添加数值标签，统一放在图表上方固定位置
        label_y_mae = y_max_mae * 0.9  # 距离顶部10%的位置
        for bar, score in zip(bars3, mae_scores):
            plt.text(bar.get_x() + bar.get_width()/2, label_y_mae, 
                    f'{score:.2f}', ha='center', va='center', fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        plt.grid(True, alpha=0.3, axis='y')
        
        # 4. 交叉验证均值对比
        plt.subplot(2, 2, 4)
        bars4 = plt.bar(existing_models, cv_scores, color='gold', alpha=0.8)
        plt.title('交叉验证 $R^2$ 对比', fontsize=16, fontweight='bold')
        plt.ylabel('CV $R^2$ Mean', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        # 调整y轴范围
        y_min_cv = max(0, min(cv_scores) - 0.05)
        y_max_cv = min(1, max(cv_scores) + 0.05)
        plt.ylim(y_min_cv, y_max_cv)
        # 添加数值标签，统一放在图表上方固定位置
        label_y_cv = y_max_cv - (y_max_cv - y_min_cv) * 0.15  # 距离顶部15%的位置
        for bar, score in zip(bars4, cv_scores):
            plt.text(bar.get_x() + bar.get_width()/2, label_y_cv, 
                    f'{score:.4f}', ha='center', va='center', fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(self.images_dir, 'base_models_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"基础模型对比图已保存为 '{os.path.join(self.images_dir, 'base_models_comparison.png')}'")

    def create_neural_network_activation_comparison(self):
        """创建神经网络激活函数相关参数对比图"""
        # 神经网络相关模型
        neural_models = ['神经网络_基础', '神经网络_RELU', '神经网络_深层', '神经网络_优化']
        
        # 提取存在的神经网络模型
        existing_neural = [name for name in neural_models if name in self.model_results]
        
        if len(existing_neural) < 2:
            print("神经网络变体不足，跳过激活函数对比图")
            return
        
        # 提取数据
        r2_scores = [self.model_results[name]['test_r2'] for name in existing_neural]
        mae_scores = [self.model_results[name]['test_mae'] for name in existing_neural]
        cv_scores = [self.model_results[name]['cv_mean'] for name in existing_neural]
        
        plt.figure(figsize=(18, 6))
        
        # 1. R²分数对比
        plt.subplot(1, 3, 1)
        bars1 = plt.bar(existing_neural, r2_scores, color='lightblue', alpha=0.8)
        plt.title('神经网络变体 $R^2$ 分数对比', fontsize=16, fontweight='bold')
        plt.ylabel('$R^2$ Score', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        # 调整y轴范围
        y_min_r2 = min(r2_scores) - 0.002
        y_max_r2 = min(1.0, max(r2_scores) + 0.003)
        plt.ylim(y_min_r2, y_max_r2)
        # 添加数值标签，统一放在图表上方固定位置
        label_y_r2 = y_max_r2 - (y_max_r2 - y_min_r2) * 0.15  # 距离顶部15%的位置
        for bar, score in zip(bars1, r2_scores):
            plt.text(bar.get_x() + bar.get_width()/2, label_y_r2, 
                    f'{score:.4f}', ha='center', va='center', fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        plt.grid(True, alpha=0.3, axis='y')
        
        # 2. MAE对比
        plt.subplot(1, 3, 2)
        bars2 = plt.bar(existing_neural, mae_scores, color='lightcoral', alpha=0.8)
        plt.title('神经网络变体 MAE 对比', fontsize=16, fontweight='bold')
        plt.ylabel('MAE', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        # 调整y轴范围
        y_max_mae = max(mae_scores) * 1.1
        plt.ylim(0, y_max_mae)
        # 添加数值标签，统一放在图表上方固定位置
        label_y_mae = y_max_mae * 0.9  # 距离顶部10%的位置
        for bar, score in zip(bars2, mae_scores):
            plt.text(bar.get_x() + bar.get_width()/2, label_y_mae, 
                    f'{score:.2f}', ha='center', va='center', fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        plt.grid(True, alpha=0.3, axis='y')
        
        # 3. 交叉验证对比
        plt.subplot(1, 3, 3)
        bars3 = plt.bar(existing_neural, cv_scores, color='lightgreen', alpha=0.8)
        plt.title('神经网络变体 交叉验证 $R^2$ 对比', fontsize=16, fontweight='bold')
        plt.ylabel('CV $R^2$ Mean', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        # 调整y轴范围
        y_min_cv = min(cv_scores) - 0.003
        y_max_cv = min(1.0, max(cv_scores) + 0.004)
        plt.ylim(y_min_cv, y_max_cv)
        # 添加数值标签，统一放在图表上方固定位置
        label_y_cv = y_max_cv - (y_max_cv - y_min_cv) * 0.15  # 距离顶部15%的位置
        for bar, score in zip(bars3, cv_scores):
            plt.text(bar.get_x() + bar.get_width()/2, label_y_cv, 
                    f'{score:.4f}', ha='center', va='center', fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(self.images_dir, 'neural_network_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"神经网络变体对比图已保存为 '{os.path.join(self.images_dir, 'neural_network_comparison.png')}'")
    
    def create_final_model_analysis(self):
        # 获取最佳模型结果
        best_model_name = max(self.model_results.keys(), 
                            key=lambda x: self.model_results[x]['test_r2'])
        best_result = self.model_results[best_model_name]
        y_test = best_result['y_test']
        y_pred = best_result['predictions']
        residuals = y_test - y_pred
        abs_errors = np.abs(residuals)
        r2 = best_result['test_r2']
        
        plt.figure(figsize=(18, 14))
        
        # 1. 预测值 vs 实际值散点图
        plt.subplot(2, 2, 1)
        plt.scatter(y_test, y_pred, alpha=0.7, color='blue', s=50)
        # 添加完美预测线
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='完美预测线')
        plt.xlabel('实际AQI值', fontsize=14, fontweight='bold')
        plt.ylabel('预测AQI值', fontsize=14, fontweight='bold')
        plt.title(f'{best_model_name} - 预测值vs实际值', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        # 添加R²分数到图中
        plt.text(0.05, 0.95, f'$R^2$ = {r2:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top', fontsize=13, fontweight='bold')
        
        # 2. 残差分析散点图
        plt.subplot(2, 2, 2)
        plt.scatter(y_pred, residuals, alpha=0.7, color='green', s=50)
        plt.axhline(y=0, color='r', linestyle='--', lw=3)
        plt.xlabel('预测AQI值', fontsize=14, fontweight='bold')
        plt.ylabel('残差 (实际值 - 预测值)', fontsize=14, fontweight='bold')
        plt.title('残差分析', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        # 添加残差统计信息
        plt.text(0.05, 0.95, f'残差均值: {residuals.mean():.3f}\n残差标准差: {residuals.std():.3f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                verticalalignment='top', fontsize=12, fontweight='bold')
        
        # 3. 残差分布直方图
        plt.subplot(2, 2, 3)
        n, bins, patches = plt.hist(residuals, bins=25, alpha=0.7, color='orange', edgecolor='black', linewidth=1)
        plt.xlabel('残差', fontsize=14, fontweight='bold')
        plt.ylabel('频次', fontsize=14, fontweight='bold')
        plt.title('残差分布', fontsize=16, fontweight='bold')
        plt.axvline(x=0, color='r', linestyle='--', lw=3, label='零残差线')
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(fontsize=12)
        
        # 4. 预测误差分布直方图
        plt.subplot(2, 2, 4)
        n, bins, patches = plt.hist(abs_errors, bins=25, alpha=0.7, color='purple', edgecolor='black', linewidth=1)
        plt.xlabel('绝对误差', fontsize=14, fontweight='bold')
        plt.ylabel('频次', fontsize=14, fontweight='bold')
        plt.title('预测误差分布', fontsize=16, fontweight='bold')
        plt.axvline(x=5, color='green', linestyle='--', alpha=0.8, lw=2, label='5 AQI点')
        plt.axvline(x=10, color='orange', linestyle='--', alpha=0.8, lw=2, label='10 AQI点')
        plt.axvline(x=20, color='red', linestyle='--', alpha=0.8, lw=2, label='20 AQI点')
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(fontsize=11)
        
        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(self.images_dir, 'final_model_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"最终模型分析图已保存为 '{os.path.join(self.images_dir, 'final_model_analysis.png')}'")

    def save_model(self, filename='best_aqi_model.joblib'):
        """保存最佳模型"""
        import joblib
        
        model_package = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        model_path = os.path.join(self.models_dir, filename)
        joblib.dump(model_package, model_path)
        print(f"\n最佳模型已保存为: {model_path}")
    
    def predict_aqi(self, pollutant_values):
        """使用训练好的模型预测AQI"""
        if self.best_model is None:
            print("请先训练模型!")
            return None
        
        # 数据预处理
        if isinstance(pollutant_values, dict):
            # 如果输入是字典，转换为DataFrame
            input_df = pd.DataFrame([pollutant_values])
            input_df = input_df[self.feature_names]  # 确保特征顺序正确
        else:
            input_df = pd.DataFrame([pollutant_values], columns=self.feature_names)
        
        # 标准化
        input_scaled = self.scaler.transform(input_df)
        
        # 预测
        prediction = self.best_model.predict(input_scaled)
        
        return prediction[0]

def main():
    """主函数"""
    print("开始AQI预测模型训练...")
    total_start_time = time.time()
    
    # 创建预测器实例
    predictor = AQIPredictor()
    
    try:
        # 1. 加载数据
        predictor.load_data()
        
        # 2. 数据预处理
        X, y = predictor.prepare_data()
        
        # 3. 基础模型训练
        base_results = predictor.train_models()
        
        # 4. 智能判断是否优化神经网络
        should_optimize, reason = predictor.should_optimize_neural_network()
        print(reason)
        
        if should_optimize:
            # 5. 神经网络激活函数优化
            neural_results = predictor.optimize_neural_network()
        else:
            print("神经网络优化跳过，因为不值得优化。")
            neural_results = {} # 确保neural_results为空字典
        
        # 6. 评估最佳模型
        predictor.evaluate_model()
        
        # 7. 创建输出文件夹
        predictor.create_output_directory() # 创建输出文件夹
        
        # 8. 保存模型
        predictor.save_model()
        
        # 9. 创建可视化图表
        print("\n正在生成可视化图表...")
        print("  [1/3] 生成基础模型对比图...")
        predictor.create_base_models_comparison()           # 图1: 五个基础模型的4个参数
        
        if should_optimize:
            print("  [2/3] 生成神经网络对比图...")
            predictor.create_neural_network_activation_comparison()  # 图2: 神经网络激活函数相关参数
        else:
            print("  [2/3] 跳过神经网络对比图 (未进行优化)")
            
        print("  [3/3] 生成最终模型分析图...")
        predictor.create_final_model_analysis()             # 图3: 最终模型的测试集回归分析

        total_time = time.time() - total_start_time
        print("\n" + "="*60)
        print(" 训练完成!")
        print("="*60)
        print(f"  总训练时间: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
        print(f"训练样本数: {len(predictor.X):,}")
        print(f" 最佳模型: {max(predictor.model_results.keys(), key=lambda x: predictor.model_results[x]['test_r2'])}")
        print(f"最佳R²分数: {max(result['test_r2'] for result in predictor.model_results.values()):.4f}")
        
        
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 