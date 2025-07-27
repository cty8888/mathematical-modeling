import pandas as pd
import numpy as np
import os
from glob import glob
from aqi_calculator import AQICalculator

def process_china_air_quality_data():
    """处理中国城市空气质量数据"""
    
    # 创建输出文件夹
    output_dir = "China_data_CSV"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出文件夹: {output_dir}")
    
    # 创建AQI计算器
    calculator = AQICalculator()
    
    # 获取所有CSV文件
    data_folder = "China_data/城市_20250101-20250719"
    csv_files = glob(os.path.join(data_folder, "*.csv"))
    csv_files.sort()
    
    print(f"找到 {len(csv_files)} 个数据文件")
    
    # 存储所有处理后的数据
    all_results = []
    
    for i, file_path in enumerate(csv_files):
        try:
            print(f"处理文件 {i+1}/{len(csv_files)}: {os.path.basename(file_path)}")
            
            # 读取单个文件
            df = pd.read_csv(file_path)
            
            # 提取日期
            date = df['date'].iloc[0]
            
            # 处理单个文件的数据
            daily_results = process_single_file(df, date, calculator)
            
            if daily_results:
                all_results.extend(daily_results)
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue
    
    # 转换为DataFrame并保存
    if all_results:
        result_df = pd.DataFrame(all_results)
        output_file = os.path.join(output_dir, "china_cities_aqi_daily.csv")
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n处理完成! 结果保存至: {output_file}")
        print(f"总共处理了 {len(result_df)} 条记录")
        print(f"包含 {result_df['date'].nunique()} 天的数据")
        print(f"包含 {result_df['city'].nunique()} 个城市")
    else:
        print("没有处理出有效数据")

def process_single_file(df, date, calculator):
    """处理单个文件的数据"""
    results = []
    
    # 获取所有城市列（除了date, hour, type）
    city_columns = [col for col in df.columns if col not in ['date', 'hour', 'type']]
    
    # 需要的污染物类型映射
    pollutant_mapping = {
        'PM2.5': 'PM2.5',
        'PM10': 'PM10', 
        'O3': 'O3',
        'SO2': 'SO2',
        'NO2': 'NO2',
        'CO': 'CO'
    }
    
    # 检查文件中存在的污染物类型
    available_types = df['type'].unique()
    print(f"  可用污染物类型: {list(available_types)}")
    
    # 为每个城市计算24小时平均值
    for city in city_columns:
        try:
            # 提取该城市的所有数据
            city_data = {}
            
            # 获取各污染物的24小时数据
            for pollutant_name, column_key in pollutant_mapping.items():
                if column_key in available_types:
                    # 获取该污染物24小时的数据
                    pollutant_rows = df[df['type'] == column_key]
                    if not pollutant_rows.empty:
                        # 获取该城市该污染物的24小时数据
                        hourly_values = []
                        for _, row in pollutant_rows.iterrows():
                            value = row[city]
                            # 处理缺失值和非数值
                            if pd.notna(value) and str(value).strip() != '':
                                try:
                                    hourly_values.append(float(value))
                                except (ValueError, TypeError):
                                    continue
                        
                        # 如果有效数据点少于18个（75%），跳过该城市该天的数据
                        if len(hourly_values) >= 18:  # 至少75%的数据完整
                            # 计算24小时平均值
                            daily_avg = np.mean(hourly_values)
                            # CO保留一位小数，其他污染物转为整数
                            if pollutant_name == 'CO':
                                city_data[pollutant_name] = round(daily_avg, 1)
                            else:
                                city_data[pollutant_name] = round(daily_avg)
            
            # 如果该城市数据不完整，跳过
            required_pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2']  # 必需的污染物
            if not all(p in city_data for p in required_pollutants):
                continue
            
            # 准备AQI计算的数据格式
            aqi_input = {
                'PM2.5_24h': city_data.get('PM2.5'),
                'PM10_24h': city_data.get('PM10'),
                'SO2_24h': city_data.get('SO2'),
                'NO2_24h': city_data.get('NO2'),
            }
            
            # 添加可选污染物
            if 'O3' in city_data:
                aqi_input['O3_8h'] = city_data['O3']
            if 'CO' in city_data:
                aqi_input['CO_24h'] = city_data['CO']
            
            # 计算AQI
            aqi = calculator.calculate_aqi(aqi_input)
            
            if aqi is not None:
                # 构建结果记录
                result = {
                    'date': date,
                    'city': city,
                    'AQI': aqi,
                    'PM2.5': city_data.get('PM2.5', None),
                    'PM10': city_data.get('PM10', None),
                    'O3': city_data.get('O3', None),
                    'SO2': city_data.get('SO2', None),
                    'NO2': city_data.get('NO2', None),
                    'CO': city_data.get('CO', None)
                }
                results.append(result)
                
        except Exception as e:
            print(f"  处理城市 {city} 时出错: {e}")
            continue
    
    print(f"  成功处理了 {len(results)} 个城市的数据")
    return results


if __name__ == "__main__":
    print("中国城市空气质量数据处理程序")
    print("="*50)
    

    
    # 处理数据
    process_china_air_quality_data() 