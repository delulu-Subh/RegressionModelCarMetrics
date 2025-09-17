# RegressionModelCarMetrics
🚗 Used Cars Price Prediction Using Linear Regression

📌 Project Overview

This project predicts the selling price of used cars based on numerical and categorical features such as year of manufacture, kilometers driven, fuel type, seller type, transmission, and ownership status using Linear Regression.

Dataset: used_cars_dataset.csv

Total Rows: 4340

Numerical Features: year, km_driven

Categorical Features: name, fuel, seller_type, transmission, owner

Target Variable: selling_price

🧩 Workflow

Data Preprocessing

Selected relevant numerical and categorical columns.

One-hot encoded categorical features to convert them into numerical format.

Scaled numerical features using StandardScaler for better model performance.

Train-Test Split

Split dataset into 80% training and 20% testing data using train_test_split.

Model Training

Trained a Linear Regression model on preprocessed data.

Extracted model coefficients and intercept.

Prediction

Generated predictions for the test dataset.

Compared Actual vs Predicted selling prices.

Evaluation Metrics

Mean Absolute Error (MAE): 9.03 × 10⁹

Mean Squared Error (MSE): 2.95 × 10²¹

Root Mean Squared Error (RMSE): 5.43 × 10¹⁰

R² Score: -7.95 × 10⁹

⚠️ The negative R² indicates that the model is performing very poorly on this high-dimensional dataset.

Visualization

Scatter plot of Actual vs Predicted selling prices.

📈 Sample Code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
dataset = pd.read_csv('used_cars_dataset.csv')

# Select features
numerical_features = ['year','km_driven']
categorical_features = ['name','fuel','seller_type','transmission','owner']

dataset = dataset[numerical_features + categorical_features + ['selling_price']]

# One-hot encode categorical variables
encoder = OneHotEncoder(drop='first', sparse=False)
categorical_encoded = encoder.fit_transform(dataset[categorical_features])
categorical_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(categorical_features))
dataset = pd.concat([dataset[numerical_features], categorical_df, dataset['selling_price']], axis=1)

# Split data
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale numerical features
scaler = StandardScaler().fit(dataset[numerical_features])
X[numerical_features] = scaler.transform(dataset[numerical_features])

# Train Linear Regression model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predictions
y_pred = regressor.predict(x_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# Plot Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='darkcyan', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.title('Actual vs Predicted Selling Prices')
plt.grid(True)
plt.show()

📊 Example Results
Metric	Value
MAE	9.03 × 10⁹
MSE	2.95 × 10²¹
RMSE	5.43 × 10¹⁰
R² Score	-7.95 × 10⁹
🚀 Insights

Linear Regression struggles with very high-dimensional categorical features (1500+ columns after encoding).

Feature engineering or dimensionality reduction is recommended before regression.

Advanced models like Random Forest, XGBoost, or Gradient Boosting may give much better performance.

🔗 Files

used_cars_dataset.csv — Original dataset

UsedCarsPricePrediction.ipynb — Full notebook with preprocessing, training, and evaluation
