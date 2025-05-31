from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

class Model:
    def __init__(self, model, name):
        self.model = model
        self.name = name

    def predict(self, X):
        return self.model.predict(X)

MODELS = [
    Model(RandomForestClassifier(), 'RandomForest'),
    Model(GradientBoostingClassifier(), 'GradientBoosting'),
    Model(XGBClassifier(), 'XGBoost'),
    Model(CatBoostClassifier(verbose=0), 'CatBoost'),
    Model(LogisticRegression(max_iter=1000), 'LogisticRegression'),
]