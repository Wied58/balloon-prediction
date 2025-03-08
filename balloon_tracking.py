import pandas as pd
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt

# Convert DDMM.MMMMM to Decimal Degrees
def convert_ddmm_to_decimal(ddmm):
    degrees = int(abs(ddmm) / 100)  # Extract degrees
    minutes = abs(ddmm) % 100  # Extract minutes
    decimal = degrees + minutes / 60
    return -decimal if ddmm < 0 else decimal  # Keep original sign

# Convert DATE (DDMMYY) to YYYY-MM-DD
def convert_date(ddmmyy):
    day = int(ddmmyy[:2])
    month = int(ddmmyy[2:4])
    year = int(ddmmyy[4:6]) + 2000  # Assuming 20XX
    return f"{year:04d}-{month:02d}-{day:02d}"

# Convert TIME (HHMMSS.S) to HH:MM:SS
def convert_time(hhmmss):
    hhmmss = str(int(float(hhmmss)))  # Remove decimals
    hours = hhmmss.zfill(6)[:2]  # Ensure 2 digits
    minutes = hhmmss.zfill(6)[2:4]
    seconds = hhmmss.zfill(6)[4:6]
    return f"{hours}:{minutes}:{seconds}"


# Load and process CSV
def load_flight_data(csv_file):
    df = pd.read_csv(csv_file)

    # Convert lat/lon from DDMM.MMMMM format
    df["latitude"] = df["LAT"].apply(convert_ddmm_to_decimal)
    df["longitude"] = df["LONG"].apply(convert_ddmm_to_decimal)

    # Drop unnecessary columns
    df.drop(columns=["LAT", "LAT_DIR", "LONG", "LONG_DIR"], inplace=True)

    # Convert DATE and TIME to proper format
    df["DATE"] = df["DATE"].astype(str).apply(convert_date)
    df["TIME"] = df["TIME"].astype(str).apply(convert_time)

    # Create TIMESTAMP column
    df["TIMESTAMP"] = pd.to_datetime(df["DATE"] + " " + df["TIME"], format="%Y-%m-%d %H:%M:%S")

    # Sort by timestamp
    df = df.sort_values(by="TIMESTAMP", ascending=True)

    # Find peak altitude (highest point)
    peak_index = df["ALT"].idxmax()

    # Keep only ascent data (everything before peak altitude)
    df_ascent = df.loc[:peak_index]

    return df_ascent




# Simulate descent using wind drift
def simulate_descent(df, parachute_diameter, default_wind_speed=5):
    peak = df.iloc[-1]  # Last row is peak altitude
    lat, lon, alt = peak["latitude"], peak["longitude"], peak["ALT"]

    descent_rate = 5  # m/s (adjust based on parachute size)
    time_to_ground = alt / descent_rate  # Total descent time in seconds

    # Approximate horizontal drift
    wind_speed = default_wind_speed  # Use a constant or replace with real wind profile per altitude
    drift_distance = wind_speed * time_to_ground  # Meters

    # Compute landing coordinates
    landing_coords = geodesic(meters=drift_distance).destination((lat, lon), 180)  # Assume southward drift
    return landing_coords.latitude, landing_coords.longitude

# Plot flight path
def plot_flight_path(df, landing_coords):
    plt.figure(figsize=(8, 6))
    plt.scatter(df["longitude"], df["latitude"], c=df["ALT"], cmap="viridis", label="Ascent Path")
    plt.scatter(landing_coords[1], landing_coords[0], color="red", marker="X", s=100, label="Predicted Landing")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Balloon Flight Path and Predicted Landing Zone")
    plt.legend()
    plt.colorbar(label="Altitude (m)")
    plt.show()

# parachust choices are 6, 9, and 12 feet but probably 9 or 274.32
# Run simulation
csv_file = "test_data.csv"  # Replace with your file path
parachute_diameter =  274.32 # 9 feet in cm
df = load_flight_data(csv_file)
landing_coords = simulate_descent(df, parachute_diameter)
print(f"Predicted Landing Coordinates: {landing_coords}")

# Plot results
plot_flight_path(df, landing_coords)
