import json

def transform(file_path, label_value):

    with open(file_path, 'r') as data_file:
        data = json.load(data_file)

    # Create an empty list to hold the new entries
    transformed_data = []

    for data_point in data:
        data_point_attr = {}
        # Iterate over the sensor data (exclude timestamp)
        for sensor, readings in data_point.items():
            if sensor != "timestamp":  # Skip the timestamp entry
                # Create a new dictionary with the sensor readings and add a label
                entry = readings.copy()  # Copy the sensor data
                data_point_attr.update(entry)
        data_point_attr["label"] = label_value  # Add the label
        transformed_data.append(data_point_attr)

    return transformed_data