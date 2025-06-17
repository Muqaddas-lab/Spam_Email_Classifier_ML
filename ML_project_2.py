# 📦 Libraries Import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load CSV File
data = pd.read_csv("emails.csv") 

#  Preview data
print(data.head())

# Label column (already 0 = ham, 1 = spam)
data['label_num'] = data['Prediction']

# Define X and y
X = data.drop(columns=["Email No.", "Prediction"])  # features
y = data["Prediction"]  # labels

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# 📊 Evaluation
print("\n🔍 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n✅ Accuracy Score:", accuracy_score(y_test, y_pred))

#Hyperparameter Tuning
param_grid = {'alpha': [0.1, 0.5, 1.0]}
grid = GridSearchCV(MultinomialNB(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("\n💡 Best Parameters:", grid.best_params_)
best_model = grid.best_estimator_

#  Final Evaluation
y_pred_best = best_model.predict(X_test)
print("\n🧪 Accuracy After Tuning:", accuracy_score(y_test, y_pred_best))
