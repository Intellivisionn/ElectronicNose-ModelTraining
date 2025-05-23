import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold

from ensembler import Ensembler
from enums import Label
import dataLoader
import setup

def trainModel(ensembler):
    trainData = dataLoader.loadTrainData()

    X_train = [reading[:-1] for reading in trainData]
    y_train = [reading[-1] for reading in trainData]

    ensembler.fit(X_train, y_train)

def testModel(ensembler):
    testData = dataLoader.loadTestData()

    X_test = [reading[:-1] for reading in testData]
    y_test = [reading[-1] for reading in testData]

    y_pred, confs = ensembler.predict(X_test)

    ensembler.modelTest(X_test, y_test)

    print("Predictions:")
    for i, pred in enumerate(y_pred):
        if pred == y_test[i]:
            print(f"\033[92mSample {i}: {Label(y_test[i]).name} (predicted {Label(pred).name} with {int(confs[i][pred] * 100)}% confidence)\033[0m")
        else:
            print(f"\033[91mSample {i}: {Label(y_test[i]).name} (predicted {Label(pred).name} with {int(confs[i][pred] * 100)}% confidence, {Label(y_test[i]).name} had {int(confs[i][y_test[i]] * 100)}%)\033[0m")

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

def crossValidate(ensembler, k_folds=0):
    allData = dataLoader.loadAllData()
    print(f"Total samples: {len(allData)}")

    X = np.array([reading[:-1] for reading in allData], dtype=np.float32)
    y = np.array([reading[-1] for reading in allData])

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold = 1
    scores = []

    for train_index, val_index in kf.split(X):
        print(f"\nFold {fold}:")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        ensembler.fit(X_train, y_train)
        y_pred, _ = ensembler.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        print(f"Fold {fold} Accuracy: {acc:.2f}")
        scores.append(acc)
        fold += 1

    print(f"\nAverage Accuracy across {k_folds} folds: {np.mean(scores):.2f}")

if __name__ == "__main__":
    method = int(input("Enter 1 for training or 2 for testing or 3 for cross-validation: "))
    ensembler = Ensembler(setup.MODELS)

    if method == 1:
        trainModel(ensembler)
    elif method == 2:
        testModel(ensembler)
    elif method == 3:
        k_folds = int(input("Enter number of folds for cross-validation: "))
        crossValidate(ensembler, k_folds)