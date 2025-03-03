import csv
from datetime import datetime, timedelta
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable

maximet_file_name = "Maximet_data/maximet_data.txt" 
anemometer_file_name = "data/Anemometer1.csv"

# maximet_file_name = "Maximet_data/maximet_data2.txt" 
# anemometer_file_name = "data/Anemometor2.csv"


# Open Maximet data and extract [Time, Wind Speed, Wind Direction]
maximet_data = []
with open(maximet_file_name, "r") as file:
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
with open(anemometer_file_name, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)

    # Read metadata
    line = next(reader)
    for i in range(0, len(line), 2):
        meta_data[line[i]] = line[i + 1]

    # Read start time
    start_time = datetime.strptime(meta_data["Capture Start Time"], "%Y-%m-%d %I.%M.%S.%f %p")
    
    # Skip the next 6 lines
    for _ in range(6):
        next(reader)
    adjusted_hour = ((maximet_data[0]["Time"]).hour) - (start_time.hour)
    # Read motion capture data
    for line in reader:
        time_offset_ms = float(line[1]) * 1000  # Convert seconds to milliseconds
        new_time = start_time + timedelta(milliseconds=time_offset_ms)
        
        new_time += timedelta(hours=adjusted_hour)

        rotation = {"x": line[2], "y": line[3], "z": line[4], "w": line[5]}
        position = {"x": line[6], "y": line[7], "z": line[8]}

        mocap_data.append({"Time": new_time, "Position": position, "Rotation": rotation})


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



# conver the quaternion angles into euler angles
for i in range(len(final_data)):
    f = final_data[i]
    q = [f["Rotation"]['w'], f["Rotation"]['x'], f["Rotation"]['y'], f["Rotation"]['z']]  # Replace with your quaternion
    r = R.from_quat(q)
    euler = r.as_euler('zyx', degrees=True)  # Convert to Euler angles (roll, yaw, pitch)
    final_data[i]["Rotation"] = {"Roll":euler[0], "Yaw":euler[1], "Pitch":euler[2]}
    # need to subtract the wind direction from the yaw to move to a different frame convention 
    # the Roll and Pitch should stay consistent as the yaw is the only frame changing 
    curr_wind_speed = final_data[i]["Wind Direction"]
    mocap_yaw = final_data[i]["Rotation"]["Yaw"]
    final_data[i]["Wind Direction"] = (mocap_yaw - curr_wind_speed) + 360
    # print(f"current wind speed: {curr_wind_speed}, mocap_yaw: {mocap_yaw}, adjusted wind direction: {(mocap_yaw - curr_wind_speed) + 360}")



# # Convert what we need into numpy arrays
x = np.array([float(entry["Position"]["x"]) for entry in final_data])
y = np.array([float(entry["Position"]["y"]) for entry in final_data])
z = np.array([float(entry["Position"]["z"]) for entry in final_data])
wind_direction = np.array([float(entry["Wind Direction"]) for entry in final_data])
wind_speed = np.array([float(entry["Wind Speed"]) for entry in final_data])

# Angle wrap the wind direction to be between 180 and -180
wind_direction = np.deg2rad(wind_direction)  # Convert to radians
wind_direction = np.arctan2(np.sin(wind_direction), np.cos(wind_direction))  # Wrap angles
# wind_direction = np.rad2deg(wind_direction)  # Convert back to degrees

# Remove outliers (you can adjust these thresholds)
remove_flag = (x > -1500) | (x < -2500) | (y < 0) | (y > 2500) | (z < 0) | (z > 2000)
x = x[~remove_flag]
y = y[~remove_flag]
z = z[~remove_flag]
wind_direction = wind_direction[~remove_flag]
wind_speed = wind_speed[~remove_flag]

# Convert wind data to vector components (u, v)
u = wind_speed * np.cos(wind_direction)
v = wind_speed * np.sin(wind_direction)


# ##############################################3
plot_raw_data = True
# Plot the raw data
if plot_raw_data:

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.quiver(x, z, y, u, v, np.zeros_like(z), length=30.5)

    ax.set_xlabel("X Position")
    ax.set_ylabel("Z Position")
    ax.set_zlabel("Y Position")
    # Set the limits of the plot to the limits of the data
    # ax.set_xlim([-1500 , -2500])
    # ax.set_ylim([np.min(y), np.max(y)])
    # ax.set_zlim([np.min(z), np.max(z)])
    # axis equal
    ax.axis('equal')
    ax.set_title("Raw 3D Wind Data")
    plt.show()


# Combine spatial data into input features
X = np.vstack((x, y, z)).T  # Shape (N, 3)

# Define Gaussian Process kernel (you can tune these hyperparameters)
# kernel = C(1.0) * RBF(length_scale=2.0)
kernel = C(1.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))

# Train GPR models separately for u and v
# gp_speed = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
# gp_direction = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
gp_speed = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=10)
gp_direction = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=10)

gp_speed.fit(X, wind_speed)
gp_direction.fit(X, wind_direction)

# Print optimized kernel parameters
print("Optimized Kernel for Speed:", gp_speed.kernel_)
print("Optimized Kernel for Direction:", gp_direction.kernel_)

# Create a grid for prediction
x_pred = np.linspace(-2200, -1700, 10)
y_pred = np.linspace(800, 1100, 10) # If want 3d plot
# y_pred = 900
z_pred = np.linspace(400, 1000, 10) 
X_pred = np.array(np.meshgrid(x_pred, y_pred, z_pred)).T.reshape(-1, 3)

# Predict smoothed u and v
pred_speed, sigma_speed = gp_speed.predict(X_pred, return_std=True)
pred_direction, sigma_direction = gp_direction.predict(X_pred, return_std=True)

# Convert predicted speed and direction to u and v
u_pred = pred_speed * np.cos(pred_direction)
v_pred = pred_speed * np.sin(pred_direction)

# Create a colormap
cmap = plt.get_cmap("viridis")  # You can use 'plasma', 'coolwarm', etc.
norm = plt.Normalize(pred_speed.min(), pred_speed.max())  # Normalize colors

# 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.quiver(X_pred[:,0], X_pred[:,2], X_pred[:,1], u_pred, v_pred, np.zeros_like(u_pred), color=cmap(norm(pred_speed)), length=15.5, normalize=False)

# Add colorbar
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Dummy array for colorbar
cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.1)
cbar.set_label("Wind Speed")

ax.set_xlabel("X Position")
ax.set_ylabel("Z Position")
ax.set_zlabel("Y Position")
ax.axis('equal')
ax.set_title("Raw 3D Wind Data")
plt.show()

######################
# 2d prediction


# Create a grid for prediction
x_pred = np.linspace(-2200, -1700, 20)
y_pred = 900
z_pred = np.linspace(400, 1000, 20) 
X_pred = np.array(np.meshgrid(x_pred, y_pred, z_pred)).T.reshape(-1, 3)

# Predict smoothed u and v
pred_speed, sigma_speed = gp_speed.predict(X_pred, return_std=True)
pred_direction, sigma_direction = gp_direction.predict(X_pred, return_std=True)

# Convert predicted speed and direction to u and v
u_pred = pred_speed * np.cos(pred_direction)
v_pred = pred_speed * np.sin(pred_direction)


# Create 2D quiver plot
plt.figure(figsize=(10, 8))
plt.quiver(X_pred[:, 0], X_pred[:, 2], u_pred, v_pred, 
           pred_speed, cmap="viridis", scale=50)

plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title(f"Wind Vector Field at y = {y_pred}")
plt.colorbar(label="Wind Speed")
plt.show()
