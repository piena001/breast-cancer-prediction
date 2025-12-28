import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import traceback

# 页面基本配置
st.set_page_config(
    page_title="乳腺癌风险预测系统",
    page_icon="🩺",
    layout="wide"
)

# 解决 Matplotlib 中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    try:
        st.title("🩺 乳腺癌风险预测系统 (ML Web App)")
        st.write("调试模式：检查应用启动流程...")
        
        # 测试数据加载
        try:
            st.write("1. 尝试加载数据...")
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            st.write("✅ 数据加载成功")
            st.write(f"数据形状: {df.shape}")
        except Exception as e:
            st.error(f"❌ 数据加载失败: {e}")
            st.text(traceback.format_exc())
            return
        
        # 测试侧边栏
        try:
            st.write("2. 尝试构建侧边栏...")
            st.sidebar.title("⚙️ 系统设置")
            split_size = st.sidebar.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
            model_list = ["Logistic Regression", "SVM", "KNN", "Decision Tree"]
            selected_model = st.sidebar.selectbox("选择算法", model_list)
            st.write("✅ 侧边栏构建成功")
        except Exception as e:
            st.error(f"❌ 侧边栏构建失败: {e}")
            st.text(traceback.format_exc())
            return
        
        # 测试模型训练
        try:
            st.write("3. 尝试模型训练...")
            X = df.drop('target', axis=1)
            y = df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            clf = LogisticRegression(random_state=42)
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"✅ 模型训练成功，准确率: {accuracy:.2%}")
        except Exception as e:
            st.error(f"❌ 模型训练失败: {e}")
            st.text(traceback.format_exc())
            return
        
        # 测试绘图功能
        try:
            st.write("4. 尝试绘制图表...")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='mean radius', y='mean texture', hue='target', ax=ax)
            st.pyplot(fig)
            st.write("✅ 图表绘制成功")
        except Exception as e:
            st.error(f"❌ 图表绘制失败: {e}")
            st.text(traceback.format_exc())
            return
        
        st.success("🎉 所有功能测试通过！应用可以正常运行")
        
    except Exception as e:
        st.error(f"应用启动失败: {e}")
        st.text(traceback.format_exc())

if __name__ == "__main__":
    main()