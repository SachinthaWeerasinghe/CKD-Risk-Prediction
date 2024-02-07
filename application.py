from flask import Flask, request

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

df = pd.read_csv('./CKD-Dataset.csv')
# Split the data into features (X) and labels (y)
X = df.drop('Risk Status', axis=1)  # Replace 'Risk Status' with the actual column name for the target variable
y = df['Risk Status']

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the model on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

@app.route("/health")
def health():
    return "UP"

@app.route("/user", methods=["POST"])
def user():
    data = request.get_json()
    # Create a DataFrame from user inputs
    user_input_df = pd.DataFrame(data)

    # Perform one-hot encoding on the user input data
    user_input_encoded = pd.get_dummies(user_input_df)

    # Realign the user input data columns with the training data columns
    user_input_encoded = user_input_encoded.reindex(columns=X.columns, fill_value=0)

    # Predict the risk status for the user input data
    predicted_risk_status = clf.predict(user_input_encoded)

    return predicted_risk_status, 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)