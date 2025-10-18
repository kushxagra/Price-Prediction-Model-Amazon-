🛒 Amazon Price Prediction Model

Predicting product prices on Amazon using data-driven regression models.
This project focuses on analyzing product features such as category, ratings, reviews, and brand information to build an accurate price prediction model using machine learning.

📌 Overview

E-commerce platforms like Amazon contain vast product listings with dynamic pricing based on multiple factors — brand reputation, ratings, category, and demand trends.
This project aims to predict the price of Amazon products using structured product metadata and text-based features, enabling insights into pricing patterns and assisting in competitive pricing strategies.

🎯 Objectives

Clean and preprocess Amazon product data

Perform feature engineering for categorical, textual, and numerical features

Build and evaluate multiple regression models

Optimize the best-performing model using hyperparameter tuning

Interpret feature importance to understand price drivers

🧠 ML Pipeline
1. Data Collection

Dataset scraped or obtained from Amazon product listings via public datasets or APIs.

Typical fields include:

product_name, category, brand, rating, review_count, description, price

2. Data Preprocessing

Handled missing values and outliers

Encoded categorical variables (category, brand) using One-Hot or Label Encoding

Cleaned textual data (product name and description) using NLP preprocessing (tokenization, stopword removal)

Scaled numerical features (rating, review_count) using StandardScaler

3. Feature Engineering

Extracted useful features such as:

Average rating buckets

Review-to-rating ratio

Word count from descriptions

Category-level price averages

Performed dimensionality reduction (optional, using PCA or feature selection)

4. Model Building

Models tested:

Linear Regression

Random Forest Regressor

XGBoost Regressor

HistGradientBoostingRegressor (final model – best performance)

5. Evaluation Metrics

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² Score

6. Model Optimization

GridSearchCV and cross-validation for hyperparameter tuning

Feature importance analysis for interpretability

🧩 Tech Stack
Component	Tools/Frameworks
Language	Python 3.10+
Data Handling	pandas, numpy
Visualization	matplotlib, seaborn
ML Models	scikit-learn, xgboost
Deployment (optional)	Flask / Streamlit
Environment	Jupyter Notebook / VS Code
