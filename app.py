
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, matthews_corrcoef, roc_auc_score,confusion_matrix, classification_report


st.title("ML Classification Model Comparison for Breast Cancer Prediction")

st.subheader("Test Dataset")

with open("test_data.csv", "rb") as f:
    st.download_button(
        label="Download sample test CSV",
        data=f,
        file_name="test_data.csv",
        mime="text/csv"
    )

file = st.file_uploader("Upload Test CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.write("Preview of dataset")
    st.dataframe(df.head())

    if "target" not in df.columns:
        st.error("Dataset must contain 'target' column")
        st.stop()

    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = joblib.load("model/scaler.pkl")
    X = scaler.transform(X)

    model_name = st.selectbox(
        "Select Model",
        [
            "logistic",
            "decision_tree",
            "knn",
            "naive_bayes",
            "random_forest",
            "xgboost"
        ]
    )

    model = joblib.load(f"model/{model_name}.pkl")

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    st.subheader("Evaluation Metrics")

    metrics_dict = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC", "MCC"],
    "Value": [
        accuracy_score(y, preds),
        precision_score(y, preds),
        recall_score(y, preds),
        f1_score(y, preds),
        roc_auc_score(y, probs),
        matthews_corrcoef(y, preds)
    ]
}
    
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df["Value"] = metrics_df["Value"].round(3)

    st.subheader("Metrics Table")
    st.dataframe(metrics_df, width="stretch")


    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, preds)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )

    st.dataframe(cm_df, width="stretch")

    st.subheader("Classification Report")
    report = classification_report(y, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.round(3)


    st.dataframe(report_df, width="stretch")


else:
    st.info("Please upload dataset.")
