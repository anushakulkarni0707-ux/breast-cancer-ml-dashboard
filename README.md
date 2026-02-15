# ML Classification Assignment

## Problem Statement
Build multiple ML models to classify whether a tumor is malignant or benign.

## Dataset Description
The Breast Cancer Wisconsin dataset contains diagnostic measurements.
- Instances: 569
- Features: 30
- Target: Malignant (0) / Benign (1)

## Models Used
- Logistic Regression
- Decision Tree
- KNN
- Naive Bayes
- Random Forest
- XGBoost

## Evaluation Metrics
Accuracy, AUC, Precision, Recall, F1, MCC Score

## Comparison Table

| ML Model            | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|---------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression | 0.982    | 0.995 | 0.986     | 0.986  | 0.986 | 0.962 |
| Decision Tree       | 0.912    | 0.916 | 0.956     | 0.903  | 0.929 | 0.817 |
| KNN                 | 0.956    | 0.979 | 0.959     | 0.972  | 0.966 | 0.905 |
| Naive Bayes         | 0.930    | 0.987 | 0.944     | 0.944  | 0.944 | 0.849 |
| Random Forest       | 0.956    | 0.995 | 0.959     | 0.972  | 0.966 | 0.905 |
| XGBoost             | 0.947    | 0.992 | 0.946     | 0.972  | 0.959 | 0.886 |

## Observations
| ML Model            | Observation |
|---------------------|-------------|
| Logistic Regression | Achieved the highest overall performance with excellent AUC and MCC. Works very well as a strong baseline model for this dataset. |
| Decision Tree       | Lower accuracy compared to other models. Likely affected by overfitting and high variance. |
| kNN                 | Provides very balanced performance with strong recall and F1-score. Performs significantly better than Decision Tree. |
| Naive Bayes         | Fast and simple but slightly lower accuracy. Assumption of feature independence may limit performance. |
| Random Forest       | Robust and stable. Performance close to kNN and logistic regression with high recall and AUC. |
| XGBoost             | Very powerful boosting model with strong metrics, though slightly below logistic regression in this run. |


## Streamlit Features
- CSV Upload
- Model Selection
- Metrics display
- Metrics table
- Confusion Matrix
- Classification Report
