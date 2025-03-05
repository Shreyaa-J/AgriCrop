# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle
from sklearn.preprocessing import StandardScaler

# Load the crop recommendation dataset
data = pd.read_csv('Crop_recommendation.csv')  # Ensure this file is in the same directory

# Separate features and target variable
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    'Bayes Net': GaussianNB()
}

# Train and save each model
"""for name, model in models.items():
    model.fit(X_train, y_train)
    with open(f'{name.replace(" ", "_").lower()}_model.pkl', 'wb') as file:
        pickle.dump(model, file)
"""
accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    accuracies[name] = accuracy
    # Save model
    with open(f"{name.replace(' ', '_').lower()}_model.pkl", 'wb') as file:
        pickle.dump(model, file)

# Save accuracies
with open("model_accuracies.pkl", "wb") as file:
    pickle.dump(accuracies, file)
print("Models have been trained and saved.")
