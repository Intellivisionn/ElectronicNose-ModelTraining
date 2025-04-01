import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

from transformer import transform
from enum import Enum


class Label(Enum):
    ARTIFICIAL_CINNAMON = 0
    ARTIFICIAL_CHOCOLATE_ICE_CREAM = 1
    ARTIFICIAL_LAVENDER = 2

full_list = []

transformed_data = transform('Data/acinnamon_20250328_161302.json', Label.ARTIFICIAL_CINNAMON.value)
full_list.extend(transformed_data)
transformed_data = transform('Data/achocolate_ice_cream_20250331_155336.json', Label.ARTIFICIAL_CHOCOLATE_ICE_CREAM.value)
full_list.extend(transformed_data)
transformed_data = transform('Data/alavender_20250331_153415.json', Label.ARTIFICIAL_LAVENDER.value)
full_list.extend(transformed_data)

X = [list(data_point.values())[:-1] for data_point in full_list]
y = [data_point["label"] for data_point in full_list]

#Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize model
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

#Train the model
gb_clf.fit(X_train, y_train)

#Predict on test data
y_pred = gb_clf.predict(X_test)

new_pred = gb_clf.predict([[27.89, 27.8, 1030.98, 94941, 400, 0, 826, 919, 926, 977, 777, 718],[28.84, 26.67, 1030.99, 162718, 416,7 ,754, 900, 868, 970, 774, 716]])
                           
print(new_pred)

#Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

#Print classification report
print(classification_report(y_test, y_pred))
