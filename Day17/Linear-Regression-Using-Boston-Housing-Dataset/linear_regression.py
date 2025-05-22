# =====================================================
# 🏠 Linear Regression on Boston Housing Dataset
# =====================================================
# This script performs basic data cleaning and applies
# linear regression to predict house prices (MEDV)
# using the Boston housing dataset.
# =====================================================

# 📦 Importing Required Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# -----------------------------------------------------
# 📁 Step 1: Load the Dataset
# -----------------------------------------------------
# Ensure 'HousingData.csv' is in your working directory.
df = pd.read_csv('HousingData.csv')

# -----------------------------------------------------
# 🔍 Step 2: Check for Missing Values
# -----------------------------------------------------
print("🧹 Missing values before cleaning:\n", df.isna().sum())

# -----------------------------------------------------
# 🧼 Step 3: Clean the Data
# -----------------------------------------------------
# Drop any rows containing NaN values
df = df.dropna()

print("\n✅ Missing values after cleaning:\n", df.isna().sum())

# -----------------------------------------------------
# 🧠 Step 4: Define Features and Target
# -----------------------------------------------------
# X = input features (first 13 columns)
# y = target variable ('MEDV' - Median house value)
X = df.iloc[:, 0:13].values
y = df["MEDV"].values

# -----------------------------------------------------
# ✂️ Step 5: Split the Data (Training & Testing)
# -----------------------------------------------------
# 70% for training, 30% for testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)

# -----------------------------------------------------
# 🏗️ Step 6: Initialize and Train the Model
# -----------------------------------------------------
model = LinearRegression()
model.fit(x_train, y_train)

# -----------------------------------------------------
# 📈 Step 7: Make Predictions on Test Set
# -----------------------------------------------------
y_pred = model.predict(x_test)

# -----------------------------------------------------
# 🧮 Step 8: Evaluate the Model (RMSE)
# -----------------------------------------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# -----------------------------------------------------
# 📊 Output: Display RMSE
# -----------------------------------------------------
print("\n📉 Root Mean Squared Error (RMSE):", rmse)

