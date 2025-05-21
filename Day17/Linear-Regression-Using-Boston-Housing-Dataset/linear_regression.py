import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load CSV 
bos1 = pd.read_csv('HousingData.csv')

# Show missing values before cleaning
print("Before cleaning:\n", bos1.isna().sum())

# Drop missing rows
bos1 = bos1.dropna()

# Check again
print("\nAfter cleaning:\n", bos1.isna().sum())

# Define features and target
X = np.array(bos1.iloc[:, 0:13])     # All columns except the last
Y = np.array(bos1["MEDV"])           # Target column

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=5)

# Initialize and train model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Predict on test set
pred_lr = lr.predict(x_test)

# Calculate RMSE manually (for older scikit-learn versions)
mse = mean_squared_error(y_test, pred_lr)
rmse = np.sqrt(mse)

print("Error for Linear Regression (RMSE):", rmse)
