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

st.set_page_config(
    page_title="乳腺癌风险预测系统",
    page_icon="🩺",
    layout="wide"
)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data.target_names

def sidebar_layout(df):
    st.sidebar.title("⚙️ 系统设置")
    
    st.sidebar.subheader("📊 模型选择")
    model_list = ["Logistic Regression", "Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)", "Decision Tree"]
    selected_model = st.sidebar.selectbox("选择算法", model_list)

    model_params = {}
    if selected_model == "K-Nearest Neighbors (KNN)":
        k_value = st.sidebar.slider("K 值", 1, 20, 5)
        model_params['k'] = k_value
    elif selected_model == "Decision Tree":
        max_depth = st.sidebar.slider("最大深度", 1, 20, 5)
        model_params['max_depth'] = max_depth
    elif selected_model == "Support Vector Machine (SVM)":
        C_value = st.sidebar.slider("正则化系数 (C)", 0.01, 10.0, 1.0)
        model_params['C'] = C_value
    
    st.sidebar.subheader("📈 测试集比例")
    split_size = st.sidebar.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)

    st.sidebar.subheader("🩺 患者特征输入")
    st.sidebar.info("调整下方滑块输入患者指标")
    
    user_input = {}
    feature_columns = df.columns[:-1]
    top_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
    
    for col in top_features:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        user_input[col] = st.sidebar.slider(f"{col}", min_val, max_val, mean_val)
    
    return split_size, selected_model, model_params, user_input, feature_columns

def plot_feature_importance(model, feature_names, model_name, X_val, y_val):
    importances = None
    
    if model_name == "Decision Tree":
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
    elif model_name == "Logistic Regression" or model_name == "Support Vector Machine (SVM)":
        if hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
            
    if importances is None:
        from sklearn.inspection import permutation_importance
        result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42)
        importances = result.importances_mean

    feature_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_imp = feature_imp.sort_values(by='Importance', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_imp, palette='viridis', ax=ax)
    plt.title(f'特征重要性分析 ({model_name})')
    plt.xlabel('重要性得分')
    plt.ylabel('医学特征')
    
    return fig

def main():
    st.title("🩺 乳腺癌风险预测系统")
    
    with st.expander("📖 系统简介", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("""
            **💡 系统说明**
            
            本系统基于机器学习技术，使用威斯康星乳腺癌数据集进行训练，可帮助医生和研究人员快速评估乳腺癌风险。
            """)
        with col2:
            st.success("""
            **🎯 使用方法**
            
            1. 在左侧选择模型和参数
            2. 输入患者特征数据
            3. 点击"开始预测"按钮
            4. 查看预测结果和分析报告
            """)
        with col3:
            st.warning("""
            **📊 数据集信息**
            
            - 样本数：569例
            - 特征数：30个医学指标
            - 类别：恶性(0) / 良性(1)
            - 来源：Sklearn数据集
            """)

    df, target_names = load_data()
    test_size, model_name, params, user_input_dict, all_features = sidebar_layout(df)

    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = None
    if model_name == "Logistic Regression":
        clf = LogisticRegression(random_state=42)
    elif model_name == "Support Vector Machine (SVM)":
        clf = SVC(C=params['C'], probability=True, random_state=42)
    elif model_name == "K-Nearest Neighbors (KNN)":
        clf = KNeighborsClassifier(n_neighbors=params['k'])
    elif model_name == "Decision Tree":
        clf = DecisionTreeClassifier(max_depth=params['max_depth'], random_state=42)

    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    st.divider()
    st.header("📊 模型性能指标")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("当前模型", model_name)
    with metric_col2:
        st.metric("准确率", f"{accuracy:.2%}")
    with metric_col3:
        st.metric("AUC 值", f"{roc_auc:.2f}")
    with metric_col4:
        st.metric("测试集比例", f"{test_size:.0%}")

    st.divider()
    st.header("🔮 预测结果")
    
    input_data = []
    feature_means = df.drop('target', axis=1).mean()
    
    for feature in all_features:
        if feature in user_input_dict:
            input_data.append(user_input_dict[feature])
        else:
            input_data.append(feature_means[feature])
            
    input_vector = np.array(input_data).reshape(1, -1)
    input_vector_scaled = scaler.transform(input_vector)

    if st.button("� 开始预测", use_container_width=True):
        prediction = clf.predict(input_vector_scaled)[0]
        prediction_proba = clf.predict_proba(input_vector_scaled)[0]
        
        col_pred1, col_pred2 = st.columns(2)
        
        with col_pred1:
            if prediction == 0:
                st.error(f"⚠️ 高风险 (恶性)")
                st.metric("恶性概率", f"{prediction_proba[0]:.2%}")
                st.progress(int(prediction_proba[0] * 100))
            else:
                st.success(f"✅ 低风险 (良性)")
                st.metric("良性概率", f"{prediction_proba[1]:.2%}")
                st.progress(int(prediction_proba[1] * 100))
        
        with col_pred2:
            st.info("""
            **预测详情**
            
            - 模型：{0}
            - 测试集准确率：{1:.2%}
            - AUC 值：{2:.2f}
            """.format(model_name, accuracy, roc_auc))
            
            st.markdown("""
            > **重要提示**：本预测结果仅基于机器学习模型实验，不能作为真实临床诊断依据。请遵医嘱。
            """)

    st.divider()
    st.header("📈 详细分析报告")

    tab1, tab2, tab3 = st.tabs(["混淆矩阵", "ROC 曲线", "特征重要性"])

    with tab1:
        st.subheader("混淆矩阵分析")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                    xticklabels=['恶性 (0)', '良性 (1)'], 
                    yticklabels=['恶性 (0)', '良性 (1)'])
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        st.pyplot(fig_cm)
        
        st.info("""
        **混淆矩阵解读**：
        - 左上角：正确预测为恶性的数量
        - 右上角：错误预测为良性的数量（漏诊）
        - 左下角：错误预测为恶性的数量（误诊）
        - 右下角：正确预测为良性的数量
        """)

    with tab2:
        st.subheader("ROC 曲线分析")
        fig_roc, ax_roc = plt.subplots()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率 (False Positive Rate)')
        plt.ylabel('真阳性率 (True Positive Rate)')
        plt.title('ROC 曲线')
        plt.legend(loc="lower right")
        st.pyplot(fig_roc)
        
        st.info("""
        **ROC 曲线解读**：
        - 曲线越靠近左上角，模型性能越好
        - AUC 值范围：0.5（随机猜测）~ 1.0（完美分类）
        - AUC > 0.9：优秀；0.8-0.9：良好；0.7-0.8：一般；< 0.7：较差
        """)

    with tab3:
        st.subheader("特征重要性分析")
        st.markdown("该图展示了模型在判断'良性/恶性'时，认为哪些医学特征最为关键。")
        fig_imp = plot_feature_importance(clf, df.columns[:-1], model_name, X_test_scaled, y_test)
        st.pyplot(fig_imp)
        
        st.info("""
        **特征重要性解读**：
        - **条形越长**：代表该特征对预测结果的影响越大
        - **医学意义**：例如，如果 `mean concave points`（平均凹点数）排在第一位，说明模型认为这个指标是判断癌症最核心的依据
        - 这有助于医生理解模型的决策过程，提高可解释性
        """)

if __name__ == "__main__":
    main()
