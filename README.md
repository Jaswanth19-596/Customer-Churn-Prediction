# Customer Churn Prediction

This project focuses on predicting customer churn for a telecommunications company using machine learning techniques. It includes thorough data exploration, visualization, feature engineering, and model evaluation to understand the factors influencing customer churn and to develop predictive models.

## Dataset

- **Source**: Telco Customer Churn dataset
- **Features**:
  - **Customer Info**: ID, Gender, Senior Citizen, Partner, Dependents
  - **Services**: Internet, Phone, Streaming, Security, etc.
  - **Contract Info**: Type of contract, paperless billing, payment method
  - **Usage/Charges**: Tenure, Monthly Charges, Total Charges
  - **Target**: `Churn` (Yes/No)

## Exploratory Data Analysis (EDA)

- Verified data types and null/missing values
- Identified column types: binary, categorical, numerical
- Visualized churn relationships across demographic and service features
- Detected imbalances in churn distribution

## Feature Engineering

- Converted object types to appropriate numeric representations
- Addressed issues like:
  - `TotalCharges` being stored as string
  - High-cardinality columns
- Encoded categorical variables using Label Encoding and One-Hot Encoding as needed

## Class Imbalance Handling

- Visualized imbalance (approx. 26% churn rate)
- Applied techniques like:
  - Undersampling
  - Oversampling (SMOTE, RandomOverSampler)
  - Ensemble-balanced techniques (e.g., Balanced Random Forest)

## Model Training & Evaluation

- Trained multiple classifiers:
  - Logistic Regression
  - K-Nearest Neighbors
  - Decision Tree
  - Support Vector Machine
  - Random Forest
- Used:
  - Cross-validation (StratifiedKFold)
  - Confusion matrix, classification report
  - ROC-AUC and Precision-Recall curves
- Performed GridSearchCV and RandomizedSearchCV for hyperparameter tuning

## Insights & Results

- Random Forest & Logistic Regression yielded competitive AUC scores
- Oversampling (SMOTE) improved recall for minority (churned) class
- Feature importance revealed `Contract`, `Tenure`, and `MonthlyCharges` as strong predictors

## Visualization

- Used `matplotlib`, `seaborn`, and `plotly.express` for:
  - Churn vs. contract type, payment method, paperless billing
  - KDE, box, violin, bar plots for numerical comparisons
  - Subplots to compare across multiple categories

## Dependencies

- Python 3.x
- pandas, numpy, scikit-learn
- seaborn, matplotlib, plotly
- imbalanced-learn

## How to Run

1. Clone this repo
2. Ensure required packages are installed via pip
3. Run `Customer_Churn_Prediction.ipynb` in a Jupyter Notebook environment

---

> **Note**: The notebook demonstrates a real-world ML workflow for a classification problem involving imbalanced classes. It is suited for beginner to intermediate data scientists looking to understand end-to-end pipelines.
