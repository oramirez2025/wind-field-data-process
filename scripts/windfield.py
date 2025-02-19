import csv
from datetime import datetime, timedelta

# Open Maximet data and extract [Time, Wind Speed, Wind Direction]
maximet_data = []
with open("Maximet_data/maximet_data.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        parsed = line.split(',')
        if len(parsed) >= 10:
            wind_direction = parsed[1]
            wind_speed = parsed[2]
            time = parsed[10].strip()
            
            # Convert time to datetime and adjust by 2.5 seconds
            time_obj = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f")
            adjusted_time_obj = time_obj + timedelta(seconds=2.5)
            
            maximet_data.append({
                "Time": adjusted_time_obj,
                "Wind Speed": float(wind_speed),
                "Wind Direction": float(wind_direction)
            })

# Open the Anemometer data and extract [Time, Position, Rotation]
meta_data = {}
mocap_data = []
with open("Anemometer_data/Anemometer1.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)

    # Read metadata
    line = next(reader)
    for i in range(0, len(line), 2):
        meta_data[line[i]] = line[i + 1]
    
    # Skip the next 5 lines
    for _ in range(5):
        next(reader)
    
    # Read start time
    start_time = datetime.strptime(meta_data["Capture Start Time"], "%Y-%m-%d %I.%M.%S.%f %p")
    
    # Read motion capture data
    for line in reader:
        time_offset_ms = float(line[1]) * 1000  # Convert seconds to milliseconds
        new_time = start_time + timedelta(milliseconds=time_offset_ms)
        new_time += timedelta(hours=5)

        rotation = {"x": line[2], "y": line[3], "z": line[4], "w": line[5]}
        position = {"x": line[6], "y": line[7], "z": line[8]}

        mocap_data.append({"Time": new_time, "Position": position, "Rotation": rotation})


# **Ensure Maximet data starts AFTER the first Mocap timestamp**
first_mocap_time = mocap_data[0]["Time"]
maximet_data = [entry for entry in maximet_data if entry["Time"] >= first_mocap_time]

# **Sync maximet_data with the closest mocap_data**
final_data = []
mocap_index = 0

for wind_entry in maximet_data:
    wind_time = wind_entry["Time"]

    # Find the closest mocap time
    while mocap_index < len(mocap_data) - 1 and \
          abs(mocap_data[mocap_index + 1]["Time"] - wind_time) < abs(mocap_data[mocap_index]["Time"] - wind_time):
        mocap_index += 1

    # Use the closest mocap data
    closest_mocap = mocap_data[mocap_index]
    
    # Store the merged data
    final_data.append({
        "Time": wind_time,
        "Position": closest_mocap["Position"],
        "Rotation": closest_mocap["Rotation"],
        "Wind Speed": wind_entry["Wind Speed"],
        "Wind Direction": wind_entry["Wind Direction"]
    })


