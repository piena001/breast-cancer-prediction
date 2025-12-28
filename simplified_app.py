import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

# 页面配置
st.set_page_config(
    page_title="乳腺癌风险预测系统",
    page_icon="🩺",
    layout="wide"
)

def main():
    st.title("🩺 乳腺癌风险预测系统")
    
    try:
        # 简化版：只显示基本信息和数据概览
        st.markdown("### 系统已启动，正在加载数据...")
        
        # 加载数据
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        
        st.success("✅ 数据加载成功！")
        
        # 显示数据概览
        st.markdown("### 数据集概览")
        st.write(f"数据集包含 {len(df)} 个样本，{len(df.columns) - 1} 个特征")
        st.write(f"良性样本：{len(df[df['target'] == 1])} 个")
        st.write(f"恶性样本：{len(df[df['target'] == 0])} 个")
        
        # 显示前5行数据
        st.write("### 数据前5行")
        st.write(df.head())
        
    except Exception as e:
        st.error(f"系统出现错误：{e}")
        import traceback
        st.text(traceback.format_exc())

if __name__ == "__main__":
    main()