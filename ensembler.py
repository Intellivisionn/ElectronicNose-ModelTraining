import joblib
import numpy as np

class Ensembler:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        probas = None
        for model in self.models:
            activeModel = joblib.load(f'Models/{model.name}.pkl')
            activeProbas = activeModel.predict_proba(X)
            if probas is None:
                probas = activeProbas
            else:
                probas = np.add(probas, activeProbas)

        probas /= len(self.models)
        
        predictions = np.argmax(probas, axis=1)

        return predictions, probas
    
    def modelTest(self, X, y):
        for model in self.models:
            activeModel = joblib.load(f'Models/{model.name}.pkl')
            accuracy = activeModel.score(X, y)
            print(f"Accuracy of {model.name}: {accuracy:.2f}")

    def fit(self, X, y):
        for model in self.models:
            model.model.fit(X, y)
            joblib.dump(model.model, f'Models/{model.name}.pkl')