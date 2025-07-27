import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

class AQIPredictorApp:
    def __init__(self, model_path='model_outputs/models/best_aqi_model.joblib'):
        if os.path.exists(model_path):
            try:
                # 加载训练好的模型
                model_package = joblib.load(model_path)
                self.model = model_package['model']
                self.scaler = model_package['scaler']
                self.feature_names = model_package['feature_names']
                self.model_loaded = True
            except Exception as e:
                self.model_loaded = False
                self.error_message = f"模型加载失败: {str(e)}"
        else:
            self.model_loaded = False
            self.error_message = f"模型文件 {model_path} 未找到"
    
    def predict_single(self, pollutant_data):
        """预测单个样本的AQI"""
        if not self.model_loaded:
            return None
        
        try:
            # 创建输入DataFrame
            input_df = pd.DataFrame([pollutant_data])
            
            # 确保特征顺序正确
            input_df = input_df.reindex(columns=self.feature_names, fill_value=0)
            
            # 标准化
            input_scaled = self.scaler.transform(input_df)
            
            # 预测
            prediction = self.model.predict(input_scaled)[0]
            
            return prediction
            
        except Exception as e:
            st.error(f"预测过程中出现错误: {str(e)}")
            return None
    
    def get_aqi_level(self, aqi_value):
        """根据AQI值返回空气质量等级"""
        if aqi_value <= 50:
            return ("优", "空气质量令人满意，基本无空气污染", "#00e400")
        elif aqi_value <= 100:
            return ("良", "空气质量可接受，但某些污染物可能对极少数异常敏感人群健康有较弱影响", "#ffff00")
        elif aqi_value <= 150:
            return ("轻度污染", "易感人群症状有轻度加剧，健康人群出现刺激症状", "#ff7e00")
        elif aqi_value <= 200:
            return ("中度污染", "进一步加剧易感人群症状，可能对健康人群心脏、呼吸系统有影响", "#ff0000")
        elif aqi_value <= 300:
            return ("重度污染", "心脏病和肺病患者症状显著加剧，运动耐受力降低，健康人群普遍出现症状", "#8f3f97")
        else:
            return ("严重污染", "健康人群运动耐受力降低，有强烈症状，提前出现某些疾病", "#7e0023")
    
    def get_primary_pollutant(self, pollutant_data):
        """根据污染物浓度计算首要污染物"""
        pollutant_standards = {
            'PM2.5': [(0, 35, 0, 50), (35, 75, 50, 100), (75, 115, 100, 150), 
                     (115, 150, 150, 200), (150, 250, 200, 300), (250, 500, 300, 500)],
            'PM10': [(0, 50, 0, 50), (50, 150, 50, 100), (150, 250, 100, 150), 
                    (250, 350, 150, 200), (350, 420, 200, 300), (420, 600, 300, 500)],
            'O3': [(0, 100, 0, 50), (100, 160, 50, 100), (160, 215, 100, 150), 
                  (215, 265, 150, 200), (265, 800, 200, 300)],
            'SO2': [(0, 50, 0, 50), (50, 150, 50, 100), (150, 475, 100, 150), 
                   (475, 800, 150, 200), (800, 1600, 200, 300)],
            'NO2': [(0, 40, 0, 50), (40, 80, 50, 100), (80, 180, 100, 150), 
                   (180, 280, 150, 200), (280, 565, 200, 300)],
            'CO': [(0, 2, 0, 50), (2, 4, 50, 100), (4, 14, 100, 150), 
                  (14, 24, 150, 200), (24, 36, 200, 300)]
        }
        
        pollutant_aqis = {}
        
        for pollutant, concentration in pollutant_data.items():
            if pollutant in pollutant_standards:
                standards = pollutant_standards[pollutant]
                aqi = 0
                
                for c_low, c_high, aqi_low, aqi_high in standards:
                    if c_low <= concentration <= c_high:
                        # 线性插值计算AQI
                        aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (concentration - c_low) + aqi_low
                        break
                    elif concentration > c_high:
                        aqi = aqi_high  # 超出范围使用最高值
                
                pollutant_aqis[pollutant] = aqi
        
        if pollutant_aqis:
            primary_pollutant = max(pollutant_aqis, key=pollutant_aqis.get)
            return primary_pollutant, pollutant_aqis[primary_pollutant]
        
        return None, 0

def main():
    # 页面配置 - 使用宽布局
    st.set_page_config(
        page_title="AQI预测系统",
        page_icon="🌬️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # 自定义CSS样式 - 优化页面布局
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        max-width: none;
    }
    .stForm {
        border: none;
    }
    .stNumberInput > div > div {
        height: 2.5rem;
    }
    .stButton > button {
        height: 2.5rem;
        margin-top: 0.5rem;
    }
    h1 {
        font-size: 2rem !important;
        margin-bottom: 1rem !important;
    }
    h4 {
        font-size: 1.1rem !important;
        margin-bottom: 0.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 简洁的标题
    st.markdown("# 🌬️ AQI空气质量预测系统")
    st.markdown("---")
    
    # 初始化应用
    if 'app' not in st.session_state:
        st.session_state.app = AQIPredictorApp()
    
    app = st.session_state.app
    
    # 检查模型是否加载成功
    if not app.model_loaded:
        st.error("❌ " + app.error_message)
        st.info("💡 请先运行 aqi_model_training.py 训练模型")
        if st.button("🔄 重新加载模型"):
            st.session_state.app = AQIPredictorApp()
            st.rerun()
        return
    
    st.success("✅ 模型加载成功！")
    
    # 创建四列布局，更宽松的间距：AQI指标 | 输入区域 | 预测结果 | 空白
    col1, col2, col3, col4 = st.columns([1.2, 2.5, 2.2, 0.1])
    
    # 左侧：AQI指标参考（固定显示）
    with col1:
        st.markdown("#### 📊 AQI指标")
        st.markdown("""
        <div style="
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 12px;
            margin: 5px 0;
            font-size: 0.9em;
        ">
            <div style="margin-bottom: 6px;">
                <strong style="color: #00e400;">🟢 0-50 优</strong><br>
                <small style="color: #666;">空气质量令人满意</small>
            </div>
            <div style="margin-bottom: 6px;">
                <strong style="color: #ffb000;">🟡 51-100 良</strong><br>
                <small style="color: #666;">空气质量可接受</small>
            </div>
            <div style="margin-bottom: 6px;">
                <strong style="color: #ff7e00;">🟠 101-150 轻度污染</strong><br>
                <small style="color: #666;">敏感人群有影响</small>
            </div>
            <div style="margin-bottom: 6px;">
                <strong style="color: #ff0000;">🔴 151-200 中度污染</strong><br>
                <small style="color: #666;">健康人群有影响</small>
            </div>
            <div style="margin-bottom: 6px;">
                <strong style="color: #8f3f97;">🟣 201-300 重度污染</strong><br>
                <small style="color: #666;">所有人群有影响</small>
            </div>
            <div>
                <strong style="color: #7e0023;">🟤 >300 严重污染</strong><br>
                <small style="color: #666;">所有人群严重影响</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 中间：输入区域
    with col2:
        st.markdown("#### 🎯 输入污染物浓度")
        
        # 创建输入表单 - 使用更紧凑的布局
        with st.form("pollutant_form"):
            # 使用三列布局来组织输入字段，让它们更紧凑
            input_col1, input_col2, input_col3 = st.columns(3)
            
            with input_col1:
                pm25 = st.number_input(
                    "PM2.5 (μg/m³)", 
                    min_value=0.0, 
                    max_value=500.0, 
                    value=35.0, 
                    step=1.0,
                    help="细颗粒物"
                )
                
                so2 = st.number_input(
                    "SO2 (μg/m³)", 
                    min_value=0.0, 
                    max_value=200.0, 
                    value=10.0, 
                    step=1.0,
                    help="二氧化硫"
                )
            
            with input_col2:
                pm10 = st.number_input(
                    "PM10 (μg/m³)", 
                    min_value=0.0, 
                    max_value=600.0, 
                    value=50.0, 
                    step=1.0,
                    help="可吸入颗粒物"
                )
                
                no2 = st.number_input(
                    "NO2 (μg/m³)", 
                    min_value=0.0, 
                    max_value=200.0, 
                    value=40.0, 
                    step=1.0,
                    help="二氧化氮"
                )
            
            with input_col3:
                o3 = st.number_input(
                    "O3 (μg/m³)", 
                    min_value=0.0, 
                    max_value=400.0, 
                    value=120.0, 
                    step=1.0,
                    help="臭氧"
                )
                
                co = st.number_input(
                    "CO (mg/m³)", 
                    min_value=0.0, 
                    max_value=10.0, 
                    value=0.8, 
                    step=0.1,
                    help="一氧化碳"
                )
            
            # 预测按钮 - 居中显示
            st.markdown("<br>", unsafe_allow_html=True)
            predict_button = st.form_submit_button("🔮 预测AQI", type="primary", use_container_width=True)
    
    # 右侧：预测结果
    with col3:
        st.markdown("#### 📈 预测结果")
        
        # 预测逻辑
        if predict_button:
            # 准备数据
            pollutant_data = {
                'PM2.5': pm25,
                'PM10': pm10,
                'O3': o3,
                'SO2': so2,
                'NO2': no2,
                'CO': co
            }
            
            # 进行预测
            predicted_aqi = app.predict_single(pollutant_data)
            
            if predicted_aqi is not None:
                level, description, color = app.get_aqi_level(predicted_aqi)
                
                # 显示主要结果 - 更紧凑的设计
                st.markdown(f"""
                <div style="
                    background-color: {color}20;
                    border-left: 4px solid {color};
                    padding: 15px;
                    border-radius: 6px;
                    margin: 8px 0;
                ">
                    <h2 style="color: {color}; margin: 0; font-size: 1.8em;">AQI: {predicted_aqi:.3f}</h2>
                    <h3 style="color: {color}; margin: 3px 0; font-size: 1.2em;">{level}</h3>
                    <p style="margin: 8px 0; color: #333; font-size: 0.9em;">{description}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 如果不是优等级，显示首要污染物 - 更紧凑的设计
                if predicted_aqi > 50:
                    primary_pollutant, primary_aqi = app.get_primary_pollutant(pollutant_data)
                    if primary_pollutant:
                        # 污染物中文名称映射
                        pollutant_names = {
                            'PM2.5': '细颗粒物',
                            'PM10': '可吸入颗粒物', 
                            'O3': '臭氧',
                            'SO2': '二氧化硫',
                            'NO2': '二氧化氮',
                            'CO': '一氧化碳'
                        }
                        
                        primary_name = pollutant_names.get(primary_pollutant, primary_pollutant)
                        
                        st.markdown(f"""
                        <div style="
                            background-color: #fff3cd;
                            border: 1px solid #ffeaa7;
                            border-radius: 6px;
                            padding: 12px;
                            margin: 8px 0;
                        ">
                            <h4 style="color: #856404; margin: 0; font-size: 1em;">⚠️ 首要污染物</h4>
                            <p style="margin: 3px 0; color: #856404; font-weight: bold; font-size: 0.95em;">{primary_name}</p>
                            <small style="color: #6c757d;">对AQI贡献最大</small>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error("预测失败，请检查输入数据")
        else:
            st.markdown("""
            <div style="
                background-color: #e7f3ff;
                border: 1px solid #b3d7ff;
                border-radius: 6px;
                padding: 15px;
                margin: 8px 0;
                text-align: center;
            ">
                <p style="margin: 0; color: #0066cc;">👆 请输入污染物数据并点击预测按钮</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 底部信息 - 简洁版本
    st.markdown("""
    <div style='
        text-align: center; 
        color: #888; 
        font-size: 0.85em; 
        margin-top: 20px; 
        padding: 10px;
        border-top: 1px solid #eee;
    '>
        🔬 基于机器学习的AQI预测系统 | 💡 仅供参考，实际数据请以官方为准
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 