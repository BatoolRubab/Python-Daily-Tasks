from sklearn.linear_model import LogisticRegression
import numpy as np

# Training Data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y=np.array([0,0,0,0,1,1,1,1])
# Train the model
model = LogisticRegression()
model.fit(X,y)

# Step 4: Predict
hours = [4.5]
prediction = model.predict([hours])
probability = model.predict_proba([hours])

# Step 5: Show result
print(f"\n📘 Studied {hours} hours")
print(f"✅ Prediction: {'Pass' if prediction[0] == 1 else 'Fail'}")
print(f'🔢 Probability to Pass: {probability[0][1]:.2f}')


