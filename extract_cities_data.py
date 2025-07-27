import pandas as pd
import os

def extract_cities_data():
    """
    从china_cities_aqi_daily.csv文件中提取指定城市的空气质量数据
    """
    # 指定要提取的城市列表
    target_cities = [
        '上海', '南京', '苏州', '连云港', '徐州', '扬州', '无锡', '常州', 
        '镇江', '泰州', '淮安', '盐城', '宿迁', '杭州', '宁波', '温州', 
        '绍兴', '湖州', '嘉兴', '台州', '金华', '舟山', '衢州', '丽水'
    ]
    
    # 数据文件路径
    input_file = 'China_data_CSV/china_cities_aqi_daily.csv'
    
    print("开始提取数据...")
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f" 错误：文件 {input_file} 不存在！")
        return
    
    try:
        # 读取CSV文件
        print(f"正在读取文件：{input_file}")
        df = pd.read_csv(input_file, encoding='utf-8')
        
        print(f"原始数据：{len(df)} 行，{len(df.columns)} 列")
        print(f"数据列名：{list(df.columns)}")
        
        # 显示数据中所有城市
        all_cities = df['city'].unique()
        print(f"数据中包含的城市总数：{len(all_cities)}")
        
        # 检查目标城市是否在数据中
        available_cities = [city for city in target_cities if city in all_cities]
        missing_cities = [city for city in target_cities if city not in all_cities]
        
        print(f"\n找到的目标城市（{len(available_cities)}个）：{available_cities}")
        if missing_cities:
            print(f"缺失的城市（{len(missing_cities)}个）：{missing_cities}")
        
        if not available_cities:
            print(" 错误：文件中没有找到任何目标城市的数据！")
            return
        
        # 筛选目标城市的数据
        print(f"\n正在筛选目标城市的数据...")
        filtered_data = df[df['city'].isin(available_cities)].copy()
        
        # 按城市和日期排序
        filtered_data = filtered_data.sort_values(['city', 'date'])
        
        # 确保输出目录存在
        output_dir = 'China_data_CSV'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录：{output_dir}")
        
        # 保存筛选后的数据到China_data_CSV目录
        output_file = 'China_data_CSV/target_cities_aqi_data.csv'
        filtered_data.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n✓ 数据提取完成！")
        print(f"  提取记录数：{len(filtered_data)}")
        print(f"  提取城市数：{len(available_cities)}")
        print(f"  输出文件：{output_file}")
        
        # 显示数据统计信息
        print(f"\n数据概览：")
        if 'date' in filtered_data.columns:
            print(f"  日期范围：{filtered_data['date'].min()} 到 {filtered_data['date'].max()}")
        
        # 按城市统计记录数
        city_counts = filtered_data['city'].value_counts()
        print(f"\n各城市记录数：")
        for city, count in city_counts.items():
            print(f"  {city}: {count}条记录")
        
        
        print(f"\n 数据提取完成！合并文件已保存为：{output_file}")
        print(f" 文件位置：{os.path.abspath(output_file)}")
        
    except Exception as e:
        print(f" 处理文件时出错：{str(e)}")

if __name__ == "__main__":
    extract_cities_data() 