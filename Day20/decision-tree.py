# Importing necessary libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder               # For converting categorical data to numbers
from sklearn.model_selection import train_test_split         # For splitting data into training and test sets
from sklearn.tree import DecisionTreeClassifier, plot_tree   # For building and visualizing the decision tree
from sklearn import metrics                                  # For checking model accuracy
import warnings

# Ignore any warnings to keep output clean
warnings.filterwarnings('ignore')

# Step 1: Load the dataset from the given URL
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
my_data = pd.read_csv(path)
print("First 5 rows of the data:")
print(my_data.head())              # Display top 5 rows

# Check basic information about the dataset (data types, nulls, etc.)
print("\nData Info:")
print(my_data.info())

# Step 2: Convert categorical columns ('Sex', 'BP', 'Cholesterol') to numeric values using LabelEncoder
label_encoder = LabelEncoder()
my_data['Sex'] = label_encoder.fit_transform(my_data['Sex'])                # Male/Female → 1/0
my_data['BP'] = label_encoder.fit_transform(my_data['BP'])                  # LOW/NORMAL/HIGH → 1/2/0 (depends on LabelEncoder)
my_data['Cholesterol'] = label_encoder.fit_transform(my_data['Cholesterol'])

# Display transformed data
print("\nEncoded Data:")
print(my_data.head())

# Step 3: Check for missing values
print("\nMissing Values:")
print(my_data.isnull().sum())           # Should all be 0 if data is clean

# Step 4: Map Drug names to numbers (optional, used just for correlation calculation)
custom_map = {'drugA': 0, 'drugB': 1, 'drugC': 2, 'drugX': 3, 'drugY': 4}
my_data['Drug_num'] = my_data['Drug'].map(custom_map)

# Step 5: Calculate correlation of each feature with the target
corr_with_drug = my_data.drop('Drug', axis=1).corr()['Drug_num']
print("\nCorrelation with Drug class (numerical):")
print(corr_with_drug)

# Optional: Step 6 - Plot the number of samples for each drug type
# (Uncomment below to show the bar chart)
# category_counts = my_data['Drug'].value_counts()
# plt.bar(category_counts.index, category_counts.values, color='blue')
# plt.xlabel('Drug')
# plt.ylabel('Count')
# plt.title('Category Distribution')
# plt.xticks(rotation=45)
# plt.show()

# Step 7: Prepare data for model training
y = my_data['Drug']                                        # Target variable (original labels, not numbers)
X = my_data.drop(['Drug', 'Drug_num'], axis=1)             # Features (drop both label columns)

# Step 8: Split data into training (70%) and testing (30%) sets
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=32)

# Step 9: Create and train the Decision Tree Classifier
# - Using "entropy" to decide splits (Information Gain)
# - max_depth = 4 limits how deep the tree can grow
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugTree.fit(X_trainset, y_trainset)

# Step 10: Make predictions on test data
tree_predictions = drugTree.predict(X_testset)

# Step 11: Evaluate model accuracy
accuracy = metrics.accuracy_score(y_testset, tree_predictions)
print("\nDecision Tree Accuracy:", accuracy)

# Step 12: Visualize the decision tree
plt.figure(figsize=(12, 8))  # Bigger plot for better visibility
plot_tree(drugTree, feature_names=X.columns, class_names=drugTree.classes_, filled=True)
plt.title("Decision Tree Visualization")
plt.show()

