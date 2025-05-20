import json
from datetime import datetime
import numpy as np

def transform(file_path, label_value=None) -> list[float]:

    with open(file_path, 'r') as data_file:
        data = json.load(data_file)

    timepoint_vectors = []

    for data_point in data[:90]:
        data_point_attr = []
        for sensor, readings in data_point.items():
            if sensor == "timestamp" or sensor == "SGP30Sensor":
                continue
            elif sensor == "BME680Sensor":
                for i, reading in enumerate(readings.values()):
                    if i in [0, 1, 2]:  # Skip temperature, pressure, humidity
                        continue
                    data_point_attr.append(reading)
            elif sensor == "GroveGasSensor":
                for i, reading in enumerate(readings.values()):
                    if i in [4, 5]:  # Skip irrelevant channels
                        continue
                    data_point_attr.append(reading)
            else:
                for reading in readings.values():
                    data_point_attr.append(reading)
        timepoint_vectors.append(data_point_attr)
    
    gradients = []
    for i in range(1, len(timepoint_vectors)):
        prev = np.array(timepoint_vectors[i - 1])
        curr = np.array(timepoint_vectors[i])
        gradient = (curr - prev).tolist()
        gradients.append(gradient)

    flattened_readings = [item for sublist in timepoint_vectors for item in sublist]
    flattened_gradients = [item for sublist in gradients for item in sublist]

    transformed_data = flattened_readings + flattened_gradients

    if label_value is not None:
        transformed_data.append(label_value)

    return transformed_data