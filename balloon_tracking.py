import pandas as pd
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt

# Load ascent data from CSV
def load_flight_data(csv_file):
    df = pd.read_csv(csv_file)
    df = df.sort_values(by="altitude", ascending=True)  # Ensure altitude is in ascending order
    return df

# Simulate descent using wind drift
def simulate_descent(df, parachute_diameter, default_wind_speed=5):
    peak = df.iloc[-1]  # Last row is peak altitude
    lat, lon, alt = peak["latitude"], peak["longitude"], peak["altitude"]

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
    plt.scatter(df["longitude"], df["latitude"], c=df["altitude"], cmap="viridis", label="Ascent Path")
    plt.scatter(landing_coords[1], landing_coords[0], color="red", marker="X", s=100, label="Predicted Landing")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Balloon Flight Path and Predicted Landing Zone")
    plt.legend()
    plt.colorbar(label="Altitude (m)")
    plt.show()

# Run simulation
csv_file = "previous_flight.csv"  # Replace with your file path
parachute_diameter = 100  # cm
df = load_flight_data(csv_file)
landing_coords = simulate_descent(df, parachute_diameter)
print(f"Predicted Landing Coordinates: {landing_coords}")

# Plot results
plot_flight_path(df, landing_coords)