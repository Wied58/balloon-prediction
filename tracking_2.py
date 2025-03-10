import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from geopy.distance import geodesic
from geopy import Point
from geopy.distance import distance

# Load and parse CSV
df = pd.read_csv("test_data.csv", dtype={"DATE": str, "TIME": str})

# Convert DDMM.MMMMM to Decimal Degrees
def convert_ddmm_to_decimal(ddmm, direction):
    degrees = int(ddmm / 100)
    minutes = ddmm % 100
    decimal = degrees + minutes / 60
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal

# Ensure DATE and TIME are read as strings
df["DATE"] = df["DATE"].astype(str)
df["TIME"] = df["TIME"].astype(str).str.split('.').str[0]  # Remove decimals
df["formatted_date"] = pd.to_datetime(df["DATE"], format="%d%m%y").dt.strftime("%Y-%m-%d")
df["formatted_time"] = df["TIME"].apply(lambda x: f"{x[:2]}:{x[2:4]}:{x[4:6]}")
df["timestamp"] = pd.to_datetime(df["formatted_date"] + " " + df["formatted_time"])
df["latitude"] = df.apply(lambda row: convert_ddmm_to_decimal(float(row["LAT"]), row["LAT_DIR"]), axis=1)
df["longitude"] = df.apply(lambda row: convert_ddmm_to_decimal(float(row["LONG"]), row["LONG_DIR"]), axis=1)
df["ALT"] = pd.to_numeric(df["ALT"], errors="coerce")  # Ensure altitude is numeric
df = df.drop(columns=["formatted_date", "formatted_time"])  # Clean up

# Apply a rolling average to smooth altitude data
df['smoothed_ALT'] = df['ALT'].rolling(window=10, center=True).mean()

# Calculate the ascent rate (change in altitude over time)
df["ascent_rate"] = df["smoothed_ALT"].diff() / df["timestamp"].diff().dt.total_seconds()

# Define thresholds
ascent_threshold = 5.0  # meters per second
descent_threshold = -5.0  # meters per second

# Detect launch: first significant increase in ascent rate
launch_index = df[df["ascent_rate"] > ascent_threshold].index.min()

# Detect peak: maximum altitude point
peak_index = df["smoothed_ALT"].idxmax()

# Detect landing: first point after peak where altitude stabilizes near ground level
landing_index = df[(df.index > peak_index) & (df["smoothed_ALT"] < 300)].index.min()

# Extract timestamps for key events
launch_time = df.loc[launch_index, 'timestamp'] if pd.notna(launch_index) else None
peak_time = df.loc[peak_index, 'timestamp'] if pd.notna(peak_index) else None
landing_time = df.loc[landing_index, 'timestamp'] if pd.notna(landing_index) else None

print(f"Launch Timestamp: {launch_time}")
print(f"Peak Timestamp: {peak_time}")
print(f"Landing Timestamp: {landing_time}")

# Separate ascent and descent data
df_ascent = df.loc[:peak_index].copy()
df_descent = df.loc[peak_index:].copy()

# Add phase column
df_ascent["phase"] = "ascent"
df_descent["phase"] = "descent"

# Calculate speed and direction for ascent data
def calculate_speed_and_direction(df):
    speeds = []
    directions = []
    for i in range(1, len(df)):
        point1 = Point(df.iloc[i-1]["latitude"], df.iloc[i-1]["longitude"])
        point2 = Point(df.iloc[i]["latitude"], df.iloc[i]["longitude"])
        distance_m = geodesic(point1, point2).meters
        time_s = (df.iloc[i]["timestamp"] - df.iloc[i-1]["timestamp"]).total_seconds()
        speed_m_s = distance_m / time_s if time_s > 0 else 0
        direction = calculate_initial_compass_bearing(point1, point2)
        speeds.append(speed_m_s)
        directions.append(direction)
    speeds.insert(0, 0)  # First point has no speed
    directions.insert(0, 0)  # First point has no direction
    return speeds, directions

def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formula used to calculate the bearing is:
        θ = atan2(sin(Δlong).cos(lat2), cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :param pointA: The tuple representing the latitude/longitude for the first point. Latitude and longitude must be in decimal degrees.
    :param pointB: The tuple representing the latitude/longitude for the second point. Latitude and longitude must be in decimal degrees.
    :return: The bearing in degrees.
    """
    lat1 = np.radians(pointA.latitude)
    lat2 = np.radians(pointB.latitude)
    diffLong = np.radians(pointB.longitude - pointA.longitude)

    x = np.sin(diffLong) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(diffLong))

    initial_bearing = np.arctan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to +180° which is not what we want for a compass bearing
    # The compass bearing needs to be in the range of 0° to 360°
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

df_ascent["speed"], df_ascent["direction"] = calculate_speed_and_direction(df_ascent)

# Concatenate ascent and descent data
df_combined = pd.concat([df_ascent, df_descent])

# Save the combined DataFrame to a new CSV file
df_combined.to_csv("ascent_descent_data.csv", index=False)


# Plot altitude profile with key events
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['smoothed_ALT'], label='Smoothed Altitude (m)', color='gray', alpha=0.7)
if launch_time:
    plt.axvline(x=launch_time, color='green', linestyle='--', label='Launch')
if peak_time:
    plt.axvline(x=peak_time, color='blue', linestyle='--', label='Peak')
if landing_time:
    plt.axvline(x=landing_time, color='red', linestyle='--', label='Landing')
plt.xlabel('Time')
plt.ylabel('Altitude (m)')
plt.title('Weather Balloon Flight Altitude Profile')
plt.legend()
plt.grid(True)
plt.show()
