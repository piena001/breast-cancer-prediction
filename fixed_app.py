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

# 页面基本配置
st.set_page_config(
    page_title="Breast Cancer Prediction System",
    page_icon="🩺",
    layout="wide"
)

# 只保留负号修复，移除中文字体设置以避免字体依赖问题
plt.rcParams['axes.unicode_minus'] = False

@st.cache_data
def load_data():
    """加载乳腺癌数据集"""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data.target_names

def sidebar_layout(df):
    st.sidebar.title("⚙️ System Settings")
    
    # 数据集划分
    split_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)

    # 模型选择
    model_list = ["Logistic Regression", "SVM", "KNN", "Decision Tree"]
    selected_model = st.sidebar.selectbox("Select Algorithm", model_list)

    # 模型超参数配置
    model_params = {}
    if selected_model == "KNN":
        k_value = st.sidebar.slider("K Value (n_neighbors)", 1, 20, 5)
        model_params['k'] = k_value
    elif selected_model == "Decision Tree":
        max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
        model_params['max_depth'] = max_depth
    elif selected_model == "SVM":
        C_value = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
        model_params['C'] = C_value
    
    # 患者特征输入
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
    """
    Plot feature importance bar chart
    """
    importances = None
    
    # 1. Decision Tree (has built-in feature_importances_)
    if model_name == "Decision Tree":
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_

    # 2. Logistic Regression / Linear SVM (use coef_ coefficients)
    elif model_name == "Logistic Regression" or model_name == "SVM":
        if hasattr(model, 'coef_'):
            # Take absolute value, as negative coefficients also indicate strong correlation (negative correlation)
            importances = np.abs(model.coef_[0])
            
    # 3. KNN or non-linear SVM (no built-in importance, use Permutation Importance for universal calculation)
    #    Note: This requires sklearn.inspection
    if importances is None:
        from sklearn.inspection import permutation_importance
        # This is a universal method that judges importance by seeing how much accuracy drops when feature order is shuffled
        # X_val here needs to be standardized data
        result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42)
        importances = result.importances_mean

    # --- Plotting logic ---
    # Create DataFrame
    feature_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    # Sort by importance in descending order
    feature_imp = feature_imp.sort_values(by='Importance', ascending=False).head(10) # Only show top 10

    # Use Seaborn for plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_imp, palette='viridis', ax=ax)
    
    plt.title(f'Feature Importance Analysis ({model_name})')
    plt.xlabel('Importance Score')
    plt.ylabel('Medical Feature')
    
    return fig

def main():
    try:
        st.title("🩺 Breast Cancer Prediction System")
        st.markdown("Machine Learning Web Application for Breast Cancer Risk Prediction")
        
        # 加载数据
        df, target_names = load_data()
        
        # 调用侧边栏布局
        test_size, model_name, params, user_input_dict, all_features = sidebar_layout(df)

        # 数据预处理
        X = df.drop('target', axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 模型构建
        clf = None
        if model_name == "Logistic Regression":
            clf = LogisticRegression(random_state=42)
        elif model_name == "SVM":
            clf = SVC(C=params.get('C', 1.0), probability=True, random_state=42)
        elif model_name == "KNN":
            clf = KNeighborsClassifier(n_neighbors=params.get('k', 5))
        elif model_name == "Decision Tree":
            clf = DecisionTreeClassifier(max_depth=params.get('max_depth', 5), random_state=42)

        # 模型训练
        clf.fit(X_train_scaled, y_train)
        
        # 模型评估
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)

        # 界面展示：模型评估部分
        st.header("📊 1. Model Evaluation Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Model Accuracy", value=f"{accuracy:.2%}")
            st.write(f"Current Model: {model_name}")
            st.write(f"Test Size: {test_size}")

        with col2:
            st.info("💡 Dataset Info: 0 = Malignant, 1 = Benign")

        # 可视化 - 使用更简单的图表避免中文问题
        st.subheader("Visualization")
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.markdown("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                        xticklabels=['Malignant (0)', 'Benign (1)'], 
                        yticklabels=['Malignant (0)', 'Benign (1)'])
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            st.pyplot(fig_cm)

        with viz_col2:
            st.markdown("ROC Curve")
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            fig_roc, ax_roc = plt.subplots()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            st.pyplot(fig_roc)

        st.write("---")
        st.subheader("🔍 3. Model Interpretability (Feature Importance)")
        st.markdown("This chart shows which medical features the model considers most critical when determining 'Benign/Malignant'.")

        # Call the function we just wrote
        # Note: X_test_scaled (standardized test data) is passed here for KNN/SVM universal calculation
        fig_imp = plot_feature_importance(clf, df.columns[:-1], model_name, X_test_scaled, y_test)
        
        st.pyplot(fig_imp)
        
        # Add interpretation text
        st.info("""
        **Chart Interpretation:**
        - **Longer bars**: Indicate that the feature has a greater impact on the prediction results.
        - **Medical significance**: For example, if `mean concave points` ranks first, it means the model considers this indicator the core basis for cancer diagnosis.
        """)

        # 界面展示：预测功能部分
        st.divider()
        st.header("🔮 2. Online Prediction")
        
        # 构建用户输入向量
        input_data = []
        feature_means = df.drop('target', axis=1).mean()
        
        for feature in all_features:
            if feature in user_input_dict:
                input_data.append(user_input_dict[feature])
            else:
                input_data.append(feature_means[feature])
                
        input_vector = np.array(input_data).reshape(1, -1)
        
        # 标准化
        input_vector_scaled = scaler.transform(input_vector)

        if st.button("Predict"):
            prediction = clf.predict(input_vector_scaled)[0]
            prediction_proba = clf.predict_proba(input_vector_scaled)[0]
            
            st.subheader("Prediction Result:")
            
            # 结果解析
            if prediction == 0:
                st.error(f"⚠️ High Risk (Malignant)")
                st.write(f"Probability of Malignant: {prediction_proba[0]:.2%}")
                st.progress(int(prediction_proba[0] * 100))
            else:
                st.success(f"✅ Low Risk (Benign)")
                st.write(f"Probability of Benign: {prediction_proba[1]:.2%}")
                st.progress(int(prediction_proba[1] * 100))
                
            st.markdown("Note: This is for educational purposes only. Not a substitute for professional medical advice.")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        import traceback
        st.text(traceback.format_exc())

if __name__ == "__main__":
    main()