# NOTE: To keep things systematic, I will start the motion capture system first then the anemometer next
# NOTE: When running this experiment, turn off the motion capture system FIRST then the Maximet 

import csv
from datetime import datetime, timedelta
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable

maximet_file_name = "../maximet_data/maxi002.txt" 
anemometer_file_name = "../mocap_data/tro_002.csv"
final_data_file_name = "../final_data/final_data.csv"

def grab_wind_direction(s):
    return s[17:20]

def grab_wind_speed(s):
    return s[14:19]

def grab_time(s):
    return datetime.strptime(s[11:19],"%H:%M:%S") 

# Open Maximet data and extract [Time, Wind Speed, Wind Direction]
maximet_data = []
with open(maximet_file_name, 'r') as file:
    lines = file.readlines()
    for line in lines:
        parsed = line.split('|')
        wind_direction = grab_wind_direction(parsed[1])
        wind_speed = grab_wind_speed(parsed[2])
        time = grab_time(parsed[0])
        try:
            maximet_data.append({
                "Time": time,
                "Wind Speed": float(wind_speed),
                "Wind Direction": float(wind_direction)
            }) 
        except ValueError:
            continue

# # Open the Anemometer data and extract [Time, Position, Rotation]
meta_data = {}
mocap_data = []
with open(anemometer_file_name, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)

    # Read metadata
    line = next(reader)
    for i in range(0, len(line), 2):
        meta_data[line[i]] = line[i + 1]

    # Read start time
    start_time = datetime.strptime(meta_data["Capture Start Time"][11:23], "%I.%M.%S.%f")
    start_time += timedelta(hours=12)
    
    # Skip the next 6 lines
    for _ in range(6):
        next(reader)

    # Read motion capture data
    for line in reader:
        time_offset_ms = float(line[1]) * 1000  # Convert seconds to milliseconds
        new_time = start_time + timedelta(milliseconds=time_offset_ms)

        rotation = {"x": line[2], "y": line[3], "z": line[4], "w": line[5]}
        position = {"x": line[6], "y": line[7], "z": line[8]}

        mocap_data.append({"Time": new_time, "Position": position, "Rotation": rotation})

with open(final_data_file_name, "w", newline="") as csvFile:
    fieldnames = [
        "Time",
        "pos_x", "pos_y", "pos_z",
        "quat_x", "quat_y", "quat_z", "quat_w"
    ]
    writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
    writer.writeheader()

    for row in mocap_data:
        writer.writerow({
            "Time": row["Time"].isoformat(),
            "pos_x": float(row["Position"]["x"]),
            "pos_y": float(row["Position"]["y"]),
            "pos_z": float(row["Position"]["z"]),
            "quat_x": float(row["Rotation"]["x"]),
            "quat_y": float(row["Rotation"]["y"]),
            "quat_z": float(row["Rotation"]["z"]),
            "quat_w": float(row["Rotation"]["w"]),
        })

# **Ensure Maximet data starts AFTER the first Mocap timestamp**
first_mocap_time = mocap_data[0]["Time"]
first_maximet_time = maximet_data[0]["Time"]

if (first_maximet_time > first_mocap_time):
    mocap_data = [entry for entry in mocap_data if entry["Time"] >= first_maximet_time]
else:
    maximet_data = [entry for entry in maximet_data if entry["Time"] >= first_mocap_time]

# **Sync maximet_data with the closest mocap_data**
final_data = []
mocap_index = 0
last_mocap_data = mocap_data[-1]['Time']

for wind_entry in maximet_data:
    wind_time = wind_entry["Time"]

    # Find the closest mocap time
    while (mocap_index < len(mocap_data) - 1 and abs(mocap_data[mocap_index + 1]["Time"] - wind_time) < abs(mocap_data[mocap_index]["Time"] - wind_time)):
        mocap_index += 1

    if wind_time > last_mocap_data:
        break

    # Use the closest mocap data
    if (mocap_index < len(mocap_data)):
        closest_mocap = mocap_data[mocap_index]
        
        # Store the merged data
        final_data.append({
            "Time": wind_time,
            "Position": closest_mocap["Position"],
            "Rotation": closest_mocap["Rotation"],
            "Wind Speed": wind_entry["Wind Speed"],
            "Wind Direction": wind_entry["Wind Direction"]
        })

def flatten_record(d):
    return {
        "Time": d["Time"].isoformat(),
        "Pos_x": d["Position"]["x"],
        "Pos_y": d["Position"]["y"],
        "Pos_z": d["Position"]["z"],
        "Rot_x": d["Rotation"]["x"],
        "Rot_y": d["Rotation"]["y"],
        "Rot_z": d["Rotation"]["z"],
        "Rot_w": d["Rotation"]["w"],
        "Wind Speed": d["Wind Speed"],
        "Wind Direction": d["Wind Direction"],
    }

# let's save off this csv data so we can see
rows = [flatten_record(r) for r in final_data]

# conver the quaternion angles into euler angles
for i in range(len(final_data)):
    f = final_data[i]
    q = [f["Rotation"]['w'], f["Rotation"]['x'], f["Rotation"]['y'], f["Rotation"]['z']]  # Replace with your quaternion
    r = R.from_quat(q)
    euler = r.as_euler('xyz', degrees=True)  # Convert to Euler angles (roll, yaw, pitch)
    final_data[i]["Rotation"] = {"Roll":euler[0], "Yaw":euler[1], "Pitch":euler[2]}
    # need to subtract the wind direction from the yaw to move to a different frame convention 
    # the Roll and Pitch should stay consistent as the yaw is the only frame changing 
    curr_wind_speed = final_data[i]["Wind Direction"]
    mocap_yaw = final_data[i]["Rotation"]["Yaw"]
    final_data[i]["Wind Direction"] = (mocap_yaw - curr_wind_speed) + 360

# Convert what we need into numpy arrays
x = np.array([float(entry["Position"]["x"]) for entry in final_data])
y = np.array([float(entry["Position"]["y"]) for entry in final_data])
z = np.array([float(entry["Position"]["z"]) for entry in final_data])
print(f"the mean y before is {np.mean(y)}")
print(f"the mean z before is {np.mean(z)}")
wind_direction = np.array([float(entry["Wind Direction"]) for entry in final_data])
wind_speed = np.array([float(entry["Wind Speed"]) for entry in final_data])

# Angle wrap the wind direction to be between 180 and -180
wind_direction = np.deg2rad(wind_direction)  # Convert to radians
wind_direction = np.arctan2(np.sin(wind_direction), np.cos(wind_direction))  # Wrap angles

# Remove outliers
remove_flag = (x < -1500) | (x > 500) | (y < 750) | (y > 1500) | (z > 750) | (z < -500)
x = x[~remove_flag]
y = y[~remove_flag]
z = z[~remove_flag]
print(f"the mean y after is {np.mean(y)}")
print(f"the mean z after is {np.mean(z)}")
wind_direction = wind_direction[~remove_flag]
wind_speed = wind_speed[~remove_flag]

# Convert wind data to vector components (u, v)
u = wind_speed * np.cos(wind_direction)
v = wind_speed * np.sin(wind_direction)

def support_count(z0, dz=50):
    return np.sum(np.abs(z - z0) < dz)

for z0 in np.linspace(z.min(), z.max(), 20):
    print(z0, support_count(z0))

# ##############################################3
plot_raw_data = True
if plot_raw_data:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.quiver(x, y, z, u, v, np.zeros_like(z), length=30.5)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.axis('equal')
    ax.set_title("Raw 3D Wind Data")
    plt.show()

# Combine spatial data into input features
X = np.vstack((x, y, z)).T

# Define Gaussian Process kernel
kernel = C(1.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))

# Train GPR models separately for u and v
gp_speed = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=10)
gp_direction = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=10)

gp_speed.fit(X, wind_speed)
gp_direction.fit(X, wind_direction)

print("Optimized Kernel for Speed:", gp_speed.kernel_)
print("Optimized Kernel for Direction:", gp_direction.kernel_)

# Create a grid for prediction
x_pred = np.linspace(-1500, 500, 20)
y_pred = np.linspace(750, 1500, 10)
z_pred = np.linspace(-500, 1750, 10)
X_pred = np.array(np.meshgrid(x_pred, y_pred, z_pred)).T.reshape(-1, 3)

# Predict smoothed u and v
pred_speed, sigma_speed = gp_speed.predict(X_pred, return_std=True)
pred_direction, sigma_direction = gp_direction.predict(X_pred, return_std=True)

u_pred = pred_speed * np.cos(pred_direction)
v_pred = pred_speed * np.sin(pred_direction)

# Create a colormap
cmap = plt.get_cmap("viridis")
norm = plt.Normalize(pred_speed.min(), pred_speed.max())

# 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.quiver(X_pred[:,0], X_pred[:,1], X_pred[:,2], u_pred, v_pred, np.zeros_like(u_pred), color=cmap(norm(pred_speed)), length=15.5, normalize=False)

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.1)
cbar.set_label("Wind Speed")

ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.axis('equal')
ax.set_title("Interpolated 3D Wind Data")
plt.show()

# 2d prediction
x_pred = np.linspace(-1500, 500, 20)
y_pred = np.linspace(750, 1500, 10)
z_pred = 254
X_pred = np.array(np.meshgrid(x_pred, y_pred, z_pred)).T.reshape(-1, 3)

pred_speed, sigma_speed = gp_speed.predict(X_pred, return_std=True)
pred_direction, sigma_direction = gp_direction.predict(X_pred, return_std=True)

u_pred = pred_speed * np.cos(pred_direction)
v_pred = pred_speed * np.sin(pred_direction)

plt.figure(figsize=(10, 8))
plt.quiver(X_pred[:, 0], X_pred[:, 1], u_pred, v_pred, pred_speed, cmap="viridis", scale=50)
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title(f"Wind Vector Field at z = {z_pred}")
plt.colorbar(label="Wind Speed")
plt.show()