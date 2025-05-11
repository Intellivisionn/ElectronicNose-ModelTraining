import json
from datetime import datetime

def transform(file_path, label_value):

    with open(file_path, 'r') as data_file:
        data = json.load(data_file)

    # Create an empty list to hold the new entries
    transformed_data = []
    start_time = None

    for data_point in data[:3000]:
        data_point_attr = {}
        for sensor, readings in data_point.items():
            if sensor == "timestamp":
                    current_time = datetime.fromisoformat(readings)
                    if start_time is None:
                        start_time = current_time
                    time_delta = current_time - start_time
                    data_point_attr["timestamp"] = time_delta.total_seconds()
            else:
                if sensor == "GroveGasSensor":
                    entry = dict(list(readings.items())[2:4])
                else:
                    entry = readings.copy()
                data_point_attr.update(entry)
        data_point_attr["label"] = label_value
        transformed_data.append(data_point_attr)

    return transformed_data