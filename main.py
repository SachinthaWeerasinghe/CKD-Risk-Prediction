from flask import Flask, request, jsonify

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

app = Flask(__name__)

data = pd.read_csv('./CKD-Dataset.csv')
# Split the data into features (X) and labels (y)
X = data.drop('Risk Status', axis=1)  # Replace 'Risk Status' with the actual column name for the target variable
y = data['Risk Status']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Adjust the test_size and random_state as desired

# Example of one-hot encoding using pandas get_dummies
cat_columns = ['Gender', 'Distict', 'Local Autority', 'Family History of CKD', 'Blood Pressure', 'Diabetes', 'Medications for diabetes/blood pressure', 'Alchol consumption', 'Tobbaco Consumption', 'Water resource', 'Usage of Artificial beverages', 'Antibiotic Consumption', 'Fertilizer']
X_train_encoded = pd.get_dummies(X_train, columns=cat_columns)
X_test_encoded = pd.get_dummies(X_test, columns=cat_columns)

# Create the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_encoded, y_train)

# Re-encode the categorical variables in the test data
X_test_encoded = pd.get_dummies(X_test, columns=cat_columns)

# Get missing columns in the test data
missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)

# Add missing columns to the test data with values of 0
for col in missing_cols:
    X_test_encoded[col] = 0

# Ensure the order of columns in the test data matches the order in the training data
X_test_encoded = X_test_encoded[X_train_encoded.columns]

# Get missing columns in the test data
missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)

# Add missing columns to the test data with values of 0
for col in missing_cols:
    X_test_encoded[col] = 0

# Ensure the order of columns in the test data matches the order in the training data
X_test_encoded = X_test_encoded[X_train_encoded.columns]

from sklearn.model_selection import cross_val_score

# Combine the features and target variables
data = pd.concat([X_train_encoded, y_train], axis=1)

# Perform cross-validation
cv_scores = cross_val_score(model, X_train_encoded, y_train, cv=5)  # Adjust the number of folds (cv) as desired


@app.route("/user", methods=["POST"])
def user():
    data = request.get_json()
    # Create a DataFrame from user inputs
    user_input_df = pd.DataFrame(data, index=[0])

    # Perform one-hot encoding on the user input DataFrame
    user_input_encoded = pd.get_dummies(user_input_df, columns=cat_columns)

    # Check for missing columns
    missing_cols = set(X_train_encoded.columns) - set(user_input_encoded.columns)

    # Add missing columns with values of 0
    for col in missing_cols:
        user_input_encoded[col] = 0

    # Reorder the columns to match the training data
    user_input_encoded = user_input_encoded[X_train_encoded.columns]

    # Make predictions on the user input data
    user_predictions = model.predict(user_input_encoded)

    return user_predictions[0], 200


if __name__ == "__main__":
    app.run(debug=True)