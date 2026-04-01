# 👥 Customer Segmentation & Churn Prediction

Built for Rapido Data Science Internship application.

## 📌 Problem Statement
Segment ride-hailing users into behavioral groups using RFM Analysis
and predict which users are likely to churn using Machine Learning.

## 📊 Dataset
- 50,000 simulated ride-hailing users
- Features: ride history, spending, complaints, ratings, activity

## 🔍 Key Findings from EDA
- days_since_last_ride is the strongest churn predictor
- Users with 3+ complaints churn at significantly higher rates
- Low avg rating given correlates strongly with churn

## 👥 Customer Segments
| Segment | Description | Retention Strategy |
|---------|-------------|-------------------|
| Champions | High frequency, high spend | Loyalty rewards |
| At Risk | Previously active, dropping off | Win-back campaigns |
| Loyal | Consistent, medium spend | Subscription plans |
| New Users | Recently joined | Onboarding offers |
| Hibernating | Long inactive | Aggressive discounts |

## 🤖 Model Performance
| Metric | Score |
|--------|-------|
| Algorithm | LightGBM Classifier |
| AUC-ROC | 0.87+ |
| Top Churn Driver | days_since_last_ride |

## 🛠️ Tech Stack
Python · LightGBM · K-Means · SHAP · Pandas · Scikit-learn · Matplotlib · Seaborn

## 📁 Files
- `customer_segmentation_churn.ipynb` — Full notebook
- `lgbm_churn_model.pkl` — Saved trained model
- `eda_churn.png` — EDA visualizations
- `churn_model_results.png` — Confusion matrix + SHAP
- `segments_pca.png` — Customer segments PCA plot
- `elbow_curve.png` — K-Means elbow curve
- `churn_report.json` — Detailed summary report
