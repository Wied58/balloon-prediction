import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load and parse CSV
df = pd.read_csv("test_data.csv", dtype={"DATE": str, "TIME": str})

# Ensure DATE and TIME are read as strings
df["DATE"] = df["DATE"].astype(str)
df["TIME"] = df["TIME"].astype(str).str.split('.').str[0]  # Remove decimals
df["formatted_date"] = pd.to_datetime(df["DATE"], format="%d%m%y").dt.strftime("%Y-%m-%d")
df["formatted_time"] = df["TIME"].apply(lambda x: f"{x[:2]}:{x[2:4]}:{x[4:6]}")
df["timestamp"] = pd.to_datetime(df["formatted_date"] + " " + df["formatted_time"])
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
