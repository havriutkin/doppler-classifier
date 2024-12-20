import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import Preprocessing

# Prepare the dataset
preprocessing = Preprocessing("doppler_data.json")
data = preprocessing.prepare_dataset(preprocessing.load_data())
data = preprocessing.balance_dataset(data)

# Separate features (X) and labels (y) and split the dataset
X = data.drop(columns=["label"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVC model with rbf kernel
svc_model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svc_model.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = svc_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the evaluation results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)


# Train the model with the polynomial kernel
svc_model = SVC(kernel="poly", C=1.0, degree=2, gamma="scale", random_state=42)
svc_model.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = svc_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the evaluation results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)


# Train logistic regression model
from sklearn.linear_model import LogisticRegression

logreg_model = LogisticRegression(random_state=42)
logreg_model.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = logreg_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the evaluation results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)
