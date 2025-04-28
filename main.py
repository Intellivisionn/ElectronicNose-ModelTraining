import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import dataLoader
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from enums import Label
from transformer import transform
import numpy as np

data = dataLoader.loadData()

X_train = [list(data_point.values())[:-1] for data_point in data]
y_train = [data_point["label"] for data_point in data]

transformed_test = transform('Data\\kokot_lavender2_20250424_173844.json', Label.LAVENDER.value)
X_test = [list(data_point.values())[:-1] for data_point in transformed_test]
y_test = [data_point["label"] for data_point in transformed_test]


#Split into train and test sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# df = pd.DataFrame(X_train, columns=[feature for feature in list(data[0].keys())[:-1]])
# corr_matrix = df.corr()

# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns)
# plt.title("Correlation Matrix")
# plt.show()

#Initialize model
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, verbose=1)

#Train the model
gb_clf.fit(X_train, y_train)

probas = gb_clf.predict_proba(X_test)

#Predict on test data
y_pred = gb_clf.predict(X_test)

# Get confidence for each prediction
confidences = [proba[pred] for proba, pred in zip(probas, y_pred)]

confidences_by_element = {}

for i in range(12):
    confidences_by_element[Label(i).name] = []

for prob in probas:
    for i, p in enumerate(prob):
        confidences_by_element[Label(i).name].append(p * 100)  # scale to %

# Plotting
plt.figure(figsize=(10, 6))
for label, values in confidences_by_element.items():
    plt.plot(values, label=f'{label} confidence')

plt.xlabel("Sample Index")
plt.ylabel("Confidence (%)")
plt.title("Per-Class Confidence Scores")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Average confidence
average_confidence = sum(confidences) / len(confidences)

unique_elements, counts = np.unique(y_pred, return_counts=True)

# Print the number and its count
for element, count in zip(unique_elements, counts):
    print(f"{Label(element).name} appears {count} times.")

# from sklearn.model_selection import cross_val_score

# scores = cross_val_score(gb_clf, X_train, y_train, cv=5)  # 5-fold cross-validation
# print(f"Cross-validation scores: {scores}")
# print(f"Mean score: {scores.mean()} +/- {scores.std()}")


#Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}\nAverage confidence: {average_confidence}")

#Print classification report
print(classification_report(y_test, y_pred))
