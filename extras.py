
# df = pd.DataFrame(X_train, columns=[feature for feature in list(trainData[0].keys())[:-1]])
# corr_matrix = df.corr()

# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns)
# plt.title("Correlation Matrix")
# plt.show()

# from sklearn.model_selection import cross_val_score

# scores = cross_val_score(gb_clf, X_train, y_train, cv=5)  # 5-fold cross-validation
# print(f"Cross-validation scores: {scores}")
# print(f"Mean score: {scores.mean()} +/- {scores.std()}")

probas = gb_clf.predict_proba(X_test)

# Get confidence for each prediction
confidences = [proba[pred] for proba, pred in zip(probas, y_pred)]

confidences_by_element = {}

for i in range(5):
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
