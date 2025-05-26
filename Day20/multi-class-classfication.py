# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score

# Ignore warnings to keep output clean
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data = pd.read_csv(file_path)
print(data.head())  # View first 5 rows

# Plot distribution of the target variable (obesity levels)
sns.countplot(y='NObeyesdad', data=data)
plt.title('Distribution of Obesity Levels')
plt.show()

# Check for missing values
print(data.isnull().sum())

# View dataset info and statistical summary
print(data.info())
print(data.describe())

# ==================== Preprocessing ====================

# Select continuous (numerical float) columns to standardize
continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()

# Apply standard scaling to normalize the float features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[continuous_columns])

# Convert the scaled features back to a DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))

# Replace original continuous features with scaled ones
scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

# Identify categorical columns to be encoded (excluding the target)
categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('NObeyesdad')

# Apply One-Hot Encoding to convert categorical variables into dummy variables
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])

# Convert encoded features to DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# Combine encoded and scaled data into one final dataset
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)

# Convert target labels (obesity classes) to numeric codes
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes

# Split features and target
X = prepped_data.drop('NObeyesdad', axis=1)
y = prepped_data['NObeyesdad']

# Split the dataset into training and testing sets (80/20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==================== Logistic Regression - One-vs-All ====================
model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)  # OvA is default for binary logistic regression
model_ova.fit(X_train, y_train)
y_pred_ova = model_ova.predict(X_test)

# Evaluate accuracy of One-vs-All
print("One-vs-All (OvA) Strategy")
print(f"Accuracy: {np.round(100 * accuracy_score(y_test, y_pred_ova), 2)}%")

# ==================== Logistic Regression - One-vs-One ====================
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train, y_train)
y_pred_ovo = model_ovo.predict(X_test)

# Evaluate accuracy of One-vs-One
print("One-vs-One (OvO) Strategy")
print(f"Accuracy: {np.round(100 * accuracy_score(y_test, y_pred_ovo), 2)}%")

# ==================== Test Size Experiment ====================
# Try different test sizes to see how accuracy is affected
for test_size in [0.1, 0.3]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    model_ova.fit(X_train, y_train)
    y_pred = model_ova.predict(X_test)
    print(f"\nTest Size: {test_size}")
    print("Accuracy:", accuracy_score(y_test, y_pred))

# ==================== Feature Importance Visualization ====================
# Feature importance = average absolute value of coefficients
feature_importance = np.mean(np.abs(model_ova.coef_), axis=0)
plt.barh(X.columns, feature_importance)
plt.title("Feature Importance (One-vs-All)")
plt.xlabel("Importance")
plt.show()

# Feature importance for One-vs-One model
# Get coefficients of all underlying binary classifiers
coefs = np.array([est.coef_[0] for est in model_ovo.estimators_])
feature_importance_ovo = np.mean(np.abs(coefs), axis=0)

plt.barh(X.columns, feature_importance_ovo)
plt.title("Feature Importance (One-vs-One)")
plt.xlabel("Importance")
plt.show()

# ==================== Pipeline Function ====================
def obesity_risk_pipeline(data_path, test_size=0.2):
    """
    Automates obesity classification pipeline:
    - Loads data
    - Standardizes numerical features
    - Encodes categorical features
    - Trains and evaluates Logistic Regression model
    """
    data = pd.read_csv(data_path)

    continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[continuous_columns])
    scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))
    scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

    categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
    categorical_columns.remove('NObeyesdad')

    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(scaled_data[categorical_columns])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

    prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)
    prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes

    X = prepped_data.drop('NObeyesdad', axis=1)
    y = prepped_data['NObeyesdad']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

# Call the pipeline
obesity_risk_pipeline(file_path, test_size=0.2)
