# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split         # For splitting data into training and testing sets
from sklearn.preprocessing import normalize                  # For normalizing the features
from sklearn.metrics import mean_squared_error               # For calculating MSE (model evaluation)
from sklearn.tree import DecisionTreeRegressor               # For building a Decision Tree Regression model
import warnings

# Ignore unnecessary warnings to keep output clean
warnings.filterwarnings('ignore')

# Step 1: Load the dataset from a URL
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
raw_data = pd.read_csv(url)         # Reading CSV data into a DataFrame
print("Sample Data:")
print(raw_data.head())              # Display first 5 rows

# Step 2: Separate the target variable (label) 'tip_amount'
y = raw_data[['tip_amount']].values.astype('float32')  # Extract tip_amount column and convert to float

# Step 3: Remove the label column from the features
proc_data = raw_data.drop(['tip_amount'], axis=1)       # Drop the 'tip_amount' column from the feature set

# Step 4: Convert the remaining data to a NumPy array (feature matrix)
X = proc_data.values

# Step 5: Normalize the features so they are on the same scale
# This helps the model perform better by treating all features equally
X = normalize(X, axis=1, norm='l1', copy=False)

# Step 6: Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 7: Create and train the Decision Tree Regressor
# - criterion = 'squared_error' means it tries to minimize the mean squared error
# - max_depth = 8 controls how deep the tree can go (to avoid overfitting)
dt_reg = DecisionTreeRegressor(criterion='squared_error', max_depth=8, random_state=35)
dt_reg.fit(X_train, y_train)       # Train the model using the training data

# Step 8: Make predictions using the trained model on the test data
y_pred = dt_reg.predict(X_test)
print("Predicted Tip Amounts:")
print(y_pred)

# Step 9: Evaluate the model using Mean Squared Error (MSE)
mse_score = mean_squared_error(y_test, y_pred)
print('MSE Score : {0:.3f}'.format(mse_score))   # Lower is better

# Step 10: Evaluate model performance using R² Score (coefficient of determination)
r2_score = dt_reg.score(X_test, y_test)
print('R^2 Score : {0:.3f}'.format(r2_score))    # Closer to 1.0 means better performance
