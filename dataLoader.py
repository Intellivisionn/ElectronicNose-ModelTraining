from enums import Label
from transformer import transform
import os

LABELS = {
    'ba': Label.BANANA.value,
    'bo': Label.BLOOD_ORANGE.value,
    'bb': Label.BLUEBERRY.value,
    'la': Label.LAVENDER.value,
    'pi': Label.PINEAPPLE.value
}

def loadTrainData() -> list[list]:
    full_data = []

    for f in os.listdir('Data/train_data'):
        if f.endswith('.json'):
            label = LABELS[f.split('_')[0]]
            file_path = os.path.join('Data/train_data', f)
            transformed_data = transform(file_path, label)
            full_data.append(transformed_data)

    return full_data

def loadTestData() -> list[list]:
    full_data = []

    for f in os.listdir('Data/test_data'):
        if f.endswith('.json'):
            label = LABELS[f.split('_')[0]]
            file_path = os.path.join('Data/test_data', f)
            transformed_data = transform(file_path, label)
            full_data.append(transformed_data)

    return full_data

def loadAllData() -> list[list]:
    return loadTrainData() + loadTestData()