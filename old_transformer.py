import json
from datetime import datetime

def transform(file_path, label_value=None) -> list[list]:

    with open(file_path, 'r') as data_file:
        data = json.load(data_file)

    # Create an empty list to hold the new entries
    transformed_data = []
    start_time = None

    for data_point in data[:150]:
        data_point_attr = []
        for sensor, readings in data_point.items():
            if sensor == "timestamp":
                current_time = datetime.fromisoformat(readings)
                if start_time is None:
                    start_time = current_time
                time_delta = current_time - start_time
                data_point_attr.append(time_delta.total_seconds())
            elif sensor == "BME680Sensor":
                for i, reading in enumerate(readings.values()):
                    if i in [0, 1, 2]:
                        continue
                    data_point_attr.append(reading)
            elif sensor == "SGP30Sensor":
                continue
            elif sensor == "GroveGasSensor":
                for i, reading in enumerate(readings.values()):
                    if i in [4, 5]:
                        continue
                    data_point_attr.append(reading)
            else:
                for reading in readings.values():
                    data_point_attr.append(reading)
        if label_value is not None:
            data_point_attr.append(label_value)
        transformed_data.append(data_point_attr)

    return transformed_data