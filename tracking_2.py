import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

df = pd.read_csv("test_data.csv", dtype={"DATE": str, "TIME": str})  # Ensure DATE and TIME are read as strings

# Convert DATE (DDMMYY) to YYYY-MM-DD
df["formatted_date"] = pd.to_datetime(df["DATE"], format="%d%m%y").dt.strftime("%Y-%m-%d")

# Ensure TIME is a string and remove any decimals
df["TIME"] = df["TIME"].astype(str).str.split('.').str[0]  # Remove any decimal places if present

# Extract HH:MM:SS from TIME (HHMMSS)
df["formatted_time"] = df["TIME"].apply(lambda x: f"{x[:2]}:{x[2:4]}:{x[4:6]}")

# Merge and parse timestamp
df["timestamp"] = pd.to_datetime(df["formatted_date"] + " " + df["formatted_time"])

# Drop unnecessary columns
df = df.drop(columns=["formatted_date", "formatted_time"])

# end of date time mess

# Convert ALT to numeric
df["ALT"] = pd.to_numeric(df["ALT"], errors="coerce")

# Sort by timestamp
df = df.sort_values("timestamp").reset_index(drop=True)

# Compute time differences (in seconds)
df["time_diff"] = df["timestamp"].diff().dt.total_seconds()
#df["time_diff"].fillna(1, inplace=True)  # Replace NaN with 1s for first row
df["time_diff"] = df["time_diff"].fillna(1)

# Compute vertical velocity (m/s)
df["vertical_velocity"] = df["ALT"].diff() / df["time_diff"]

# Apply Savitzky-Golay filter for smoothing
df["smoothed_velocity"] = savgol_filter(df["vertical_velocity"], window_length=11, polyorder=2)

# Detect takeoff: When vertical velocity becomes significantly positive
takeoff_index = df[df["smoothed_velocity"] > 1].index[0]  # Adjust threshold if needed

# Detect landing: When vertical velocity stabilizes near zero for a sustained period
landing_index = df[(df["smoothed_velocity"].abs() < 0.5)].index[-1]  # Adjust threshold if needed

# Trim dataset to flight phase
flight_df = df.loc[takeoff_index:landing_index]

# Plot altitude vs. timestamp
plt.figure(figsize=(10, 5))
plt.plot(df["timestamp"], df["ALT"], label="Raw Altitude", color="gray", alpha=0.5)
plt.plot(flight_df["timestamp"], flight_df["ALT"], label="Flight Phase", color="blue")
plt.xlabel("Time")
plt.ylabel("Altitude (m)")
plt.title("Weather Balloon Altitude Profile")
plt.legend()
plt.grid()
plt.show()

# Plot vertical velocity for reference
plt.figure(figsize=(10, 3))
plt.plot(df["timestamp"], df["smoothed_velocity"], label="Smoothed Vertical Velocity", color="red")
plt.axhline(y=1, color="green", linestyle="--", label="Takeoff Threshold")
plt.axhline(y=-0.5, color="purple", linestyle="--", label="Landing Threshold")
plt.xlabel("Time")
plt.ylabel("Vertical Velocity (m/s)")
plt.title("Vertical Velocity Profile")
plt.legend()
plt.grid()
plt.show()
