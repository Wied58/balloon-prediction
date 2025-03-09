import pandas as pd
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

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

    return df

# Trim ground data using rolling average to detect when the ascent stops
def trim_ground_data(df, window_size=20, threshold=0.5):
    # Apply a rolling mean to the altitude to smooth the data
    df['smoothed_altitude'] = df['ALT'].rolling(window=window_size).mean()

    # Detect when the altitude stops increasing (ascent ends) and when the balloon is at a stable altitude
    diff = np.diff(df['smoothed_altitude'].dropna())
    stable_altitude = np.abs(diff).mean() < threshold  # When the difference between consecutive altitudes is small

    # If the balloon is stable, the ascent phase ends here
    peak_index = df['smoothed_altitude'].idxmax()

    print(f"Peak index: {peak_index}")  # Debugging the peak index
    print(f"Length of the dataset: {len(df)}")  # Check dataset length

    # Ensure peak_index is within bounds
    peak_index = min(peak_index, len(df) - 1)

    return df[:peak_index], peak_index

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

# Function to plot ascent and descent with map and elevation plot
def plot_flight_path(df_ascent, df_descent, peak_index, predicted_landing_coords):
    # Ensure peak_index is within the bounds of the ascent DataFrame
    peak_index = min(peak_index, len(df_ascent) - 1)

    # Define the map region based on the balloon's flight location
    lat_min = df_ascent["latitude"].min() - 0.5  # Zoom in by reducing the bounds
    lat_max = df_ascent["latitude"].max() + 0.5
    lon_min = df_ascent["longitude"].min() - 0.5
    lon_max = df_ascent["longitude"].max() + 0.5

    # Setup Basemap
    m = Basemap(projection='lcc', lat_0=np.mean([lat_min, lat_max]), lon_0=np.mean([lon_min, lon_max]), resolution='h',
                llcrnrlat=lat_min, urcrnrlat=lat_max, llcrnrlon=lon_min, urcrnrlon=lon_max)

    fig, ax = plt.subplots(2, 1, figsize=(12, 15))

    # Top plot: Flight path map
    m.drawcoastlines(ax=ax[0])
    m.drawcountries(ax=ax[0])
    m.drawstates(ax=ax[0])  # Add state boundaries
    m.drawrivers(ax=ax[0])
    m.drawparallels(np.arange(-90., 91., 0.5), ax=ax[0])  # Latitude lines
    m.drawmeridians(np.arange(-180., 181., 0.5), ax=ax[0])  # Longitude lines

    # Plot the ascent path in blue
    x_ascent, y_ascent = m(df_ascent["longitude"].values, df_ascent["latitude"].values)
    m.plot(x_ascent, y_ascent, color="blue", linewidth=2, label="Ascent", ax=ax[0])

    # Plot the descent path in red (this is using the remaining data after peak_index)
    x_descent, y_descent = m(df_descent["longitude"].values, df_descent["latitude"].values)
    m.plot(x_descent, y_descent, color="red", linewidth=2, label="Descent", ax=ax[0])

    # Plot black X at max altitude (peak)
    peak_lat, peak_lon = df_ascent.iloc[peak_index][["latitude", "longitude"]]
    peak_x, peak_y = m(peak_lon, peak_lat)
    ax[0].scatter(peak_x, peak_y, color="black", marker="x", s=200, label="Max Altitude", edgecolors="white", linewidths=2)

    # Plot landing prediction (as an X marker) using the predicted landing coordinates
    predicted_landing_x, predicted_landing_y = m(predicted_landing_coords[1], predicted_landing_coords[0])
    ax[0].scatter(predicted_landing_x, predicted_landing_y, color="red", marker="x", s=200, label="Predicted Landing", edgecolors="black", linewidths=2)

    # Bottom plot: Elevation (Altitude) over time
    ax[1].plot(df_ascent["TIMESTAMP"], df_ascent["ALT"], color="green", label="Altitude (m)", linewidth=2)
    ax[1].scatter(df_ascent["TIMESTAMP"].iloc[peak_index], df_ascent["ALT"].iloc[peak_index], color="black", marker="x", s=100, label="Max Altitude")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Altitude (m)")
    ax[1].set_title("Elevation (Altitude) over Time")
    ax[1].legend()
    ax[1].grid(True)

    # Show the map with the flight path and landing prediction
    plt.tight_layout()
    plt.show()


# Run simulation
csv_file = "test_data.csv"  # Replace with your file path
parachute_diameter =  274.32 # 9 feet in cm
df_flight = load_flight_data(csv_file)
df_ascent, peak_index = trim_ground_data(df_flight)

# Simulate descent and landing
predicted_landing_coords = simulate_descent(df_ascent, parachute_diameter)

# Split the data into ascent and descent (after peak_index)
df_descent = df_flight.iloc[peak_index:]  # The descent starts after the peak index

# Plot the flight path with the landing prediction
plot_flight_path(df_ascent, df_descent, peak_index, predicted_landing_coords)
