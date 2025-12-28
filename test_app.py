import streamlit as st

# 简单的测试脚本
st.title("测试页面")
st.write("如果能看到这个内容，说明Streamlit基本功能正常")

# 检查版本信息
import sys
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_breast_cancer

st.write(f"Python版本: {sys.version}")
st.write(f"Streamlit版本: {st.__version__}")
st.write(f"Matplotlib版本: {matplotlib.__version__}")
st.write(f"Pandas版本: {pd.__version__}")
st.write(f"NumPy版本: {np.__version__}")
st.write(f"Seaborn版本: {sns.__version__}")

# 测试数据加载
try:
    data = load_breast_cancer()
    st.success("成功加载乳腺癌数据集")
except Exception as e:
    st.error(f"加载数据失败: {e}")