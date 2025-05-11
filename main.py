import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from enums import Label
import numpy as np
import joblib

import dataLoader

def create_time_window_features(data, window_size):
    result = []
    for i in range(window_size - 1, len(data)):
        subres = []
        for datapoint in data[i - window_size + 1:i + 1]:
            subres.extend(datapoint)
        result.append(subres)

    return result

def trainModel():
    trainData = dataLoader.loadTrainData()

    X_train = [data_point[:-1] for data_point in trainData]
    y_train = [data_point[-1] for data_point in trainData[4:]]
    # gradients = np.gradient(X_train, axis=0)
    # X_train = np.concatenate((X_train, gradients), axis=1)
    X_train = create_time_window_features(X_train, 5)

    #Initialize model
    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, verbose=1)

    #Train the model
    gb_clf.fit(X_train, y_train)

    joblib.dump(gb_clf, 'model.pkl')

def testModel():
    gb_clf = joblib.load('model.pkl')

    testData = dataLoader.loadTestData()

    X_test = [data_point[:-1] for data_point in testData]
    y_test = [data_point[-1] for data_point in testData[4:]]
    # gradients = np.gradient(X_train, axis=0)
    # X_train = np.concatenate((X_train, gradients), axis=1)
    X_test = create_time_window_features(X_test, 5)

    #Predict on test data
    y_pred = gb_clf.predict(X_test)

    # print("Predictions:")
    # for i, pred in enumerate(y_pred):
    #     print(f"Sample {i}: {Label(pred).name}")
    #     print(f"Confidence: {gb_clf.predict_proba([X_test[i]])[0][pred] * 100:.2f}%")
    #     print(f"True label: {Label(y_test[i]).name}")

    #Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    #Print classification report
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(gb_clf, 'model.pkl')

def trainAndTestModel():
    data = dataLoader.loadTrainData()
    data.extend(dataLoader.loadTestData())

    X = [data_point[:-1] for data_point in data]
    X = create_time_window_features(X, 5)
    y = [data_point[-1] for data_point in data[4:]]

    #Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    #Initialize model
    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, verbose=1)

    #Train the model
    gb_clf.fit(X_train, y_train)

    #Predict on test data
    y_pred = gb_clf.predict(X_test)

    #Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    #Print classification report
    print(classification_report(y_test, y_pred, zero_division=0))

def main():
    method = int(input("Enter 1 for training or 2 for testing or 3 for training + testing: "))

    if method == 1:
        trainModel()
    elif method == 2:
        testModel()
    else:
        trainAndTestModel()

main()