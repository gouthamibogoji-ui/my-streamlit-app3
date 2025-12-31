import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve
)
from matplotlib.colors import ListedColormap

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(page_title="Logistic Regression App", layout="wide")
st.title("📊 Customer Purchase Prediction-Logistic Regression")

# ----------------------------------
# Upload Dataset
# ----------------------------------
st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Logistic dataset1 CSV", type=["csv"]
)

if uploaded_file is not None:

    dataset = pd.read_csv(uploaded_file)
    st.subheader("📁 Dataset Preview")
    st.dataframe(dataset.head())

    # ----------------------------------
    # Feature & Target Selection
    # ----------------------------------
    st.sidebar.subheader("Feature Selection")

    feature_cols = st.sidebar.multiselect(
        "Select EXACTLY 2 Feature Columns",
        dataset.columns,
        default=list(dataset.columns[2:4])
    )

    target_col = st.sidebar.selectbox(
        "Select Target Column",
        dataset.columns,
        index=len(dataset.columns) - 1
    )

    if len(feature_cols) != 2:
        st.warning("⚠️ Please select EXACTLY 2 feature columns.")
        st.stop()

    # ----------------------------------
    # Prepare Features
    # ----------------------------------
    X_df = dataset[feature_cols]
    y = dataset[target_col].values

    # Detect categorical & numerical columns
    cat_cols = X_df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X_df.select_dtypes(exclude=["object"]).columns.tolist()

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), cat_cols),
            ("num", StandardScaler(), num_cols)
        ]
    )

    X = preprocessor.fit_transform(X_df)

    # ----------------------------------
    # Train-Test Split
    # ----------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    # ----------------------------------
    # Train Model
    # ----------------------------------
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # ----------------------------------
    # Model Metrics
    # ----------------------------------
    st.subheader("📌 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Confusion Matrix**")
        st.write(confusion_matrix(y_test, y_pred))

    with col2:
        st.write("**Accuracy**")
        st.write(accuracy_score(y_test, y_pred))

    st.write("**Classification Report**")
    st.text(classification_report(y_test, y_pred))

    st.write("**Train Accuracy:**", classifier.score(X_train, y_train))
    st.write("**Test Accuracy:**", classifier.score(X_test, y_test))

    # ----------------------------------
    # ROC Curve
    # ----------------------------------
    st.subheader("📈 ROC Curve")

    y_pred_prob = classifier.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # ----------------------------------
    # Decision Boundary (ONLY if numeric)
    # ----------------------------------
    if len(cat_cols) == 0:
        st.subheader("🧠 Decision Boundary (Training Set)")

        X1, X2 = np.meshgrid(
            np.arange(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 0.01),
            np.arange(X_train[:, 1].min() - 1, X_train[:, 1].max() + 1, 0.01)
        )

        fig2, ax2 = plt.subplots()
        ax2.contourf(
            X1,
            X2,
            classifier.predict(
                np.array([X1.ravel(), X2.ravel()]).T
            ).reshape(X1.shape),
            alpha=0.75,
            cmap=ListedColormap(("red", "green"))
        )

        for i in np.unique(y_train):
            ax2.scatter(
                X_train[y_train == i, 0],
                X_train[y_train == i, 1],
                label=i
            )

        ax2.set_title("Decision Boundary (Training Set)")
        ax2.set_xlabel(feature_cols[0])
        ax2.set_ylabel(feature_cols[1])
        ax2.legend()
        st.pyplot(fig2)

    else:
        st.info("ℹ️ Decision boundary is shown only when BOTH features are numerical.")
        
    # ----------------------------------
    # Decision Boundary (Test Set)
    # ----------------------------------
    st.subheader("🧪 Decision Boundary (Test Set)")

    X1_test, X2_test = np.meshgrid(
        np.arange(X_test[:, 0].min() - 1, X_test[:, 0].max() + 1, 0.01),
        np.arange(X_test[:, 1].min() - 1, X_test[:, 1].max() + 1, 0.01)
    )

    fig3, ax3 = plt.subplots()
    ax3.contourf(
        X1_test,
        X2_test,
        classifier.predict(
            np.array([X1_test.ravel(), X2_test.ravel()]).T
        ).reshape(X1_test.shape),
        alpha=0.75,
        cmap=ListedColormap(("red", "green"))
    )

    for i in np.unique(y_test):
        ax3.scatter(
            X_test[y_test == i, 0],
            X_test[y_test == i, 1],
            label=i
        )

    ax3.set_title("Decision Boundary (Test Set)")
    ax3.set_xlabel(feature_cols[0])
    ax3.set_ylabel(feature_cols[1])
    ax3.legend()
    st.pyplot(fig3)


    # ----------------------------------
    # Future Prediction
    # ----------------------------------
    st.subheader("🔮 Future Prediction")

    future_file = st.file_uploader(
        "Upload Future Dataset2 CSV", type=["csv"], key="future"
    )

    if future_file is not None:
        future_df = pd.read_csv(future_file)
        st.write("Future Data Preview")
        st.dataframe(future_df.head())

        future_X = preprocessor.transform(future_df[feature_cols])
        future_df["Prediction"] = classifier.predict(future_X)

        st.success("✅ Prediction Completed")
        st.dataframe(future_df)

        csv = future_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions",
            csv,
            "logistic_predictions.csv",
            "text/csv"
        )

else:
    st.info("⬅️ Upload a dataset from the sidebar to begin.")



