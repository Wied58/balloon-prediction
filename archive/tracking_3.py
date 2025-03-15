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
df = df.drop(columns=["formatted_date", "formatted_time"])  # Clean up


# Ensure 'ALT' column is numeric
df['ALT'] = pd.to_numeric(df['ALT'], errors='coerce')

# Parameters
window_size = 60  # Rolling window size in number of samples
std_threshold = 1  # Standard deviation threshold for anomaly detection

# Calculate rolling mean and standard deviation
df['rolling_mean'] = df['ALT'].rolling(window=window_size, center=True).mean()
df['rolling_std'] = df['ALT'].rolling(window=window_size, center=True).std()

# Detect anomalies based on standard deviation threshold
df['anomaly'] = (df['ALT'] > df['rolling_mean'] + std_threshold * df['rolling_std']) | \
                 (df['ALT'] < df['rolling_mean'] - std_threshold * df['rolling_std'])

# Identify phases based on anomalies
# Ensure 'anomaly' column has no NaN values
df['anomaly'] = df['anomaly'].fillna(False)

# Initialize 'phase' column with StringDtype
df['phase'] = pd.Series([pd.NA] * len(df), dtype=pd.StringDtype())

# Assign 'Anomaly' to rows where 'anomaly' is True
df.loc[df['anomaly'], 'phase'] = 'Anomaly'

# Optional: Further classify anomalies into 'Launch', 'Peak', 'Landing' based on context
# For example, you can use the following approach:
# - 'Launch' could be the first set of anomalies with increasing altitude
# - 'Landing' could be the last set of anomalies with decreasing altitude
# - 'Peak' could be anomalies at high altitude with descending altitude before and after

# Example classification (this is a simplified approach and may need adjustment):
# Detect 'Launch' phase (anomalies at the start with increasing altitude)
launch_start = df[df['phase'] == 'Anomaly'].iloc[0]['timestamp']
df.loc[(df['timestamp'] >= launch_start) & (df['ALT'] > df['rolling_mean']), 'phase'] = 'Launch'

# Detect 'Landing' phase (anomalies at the end with decreasing altitude)
landing_end = df[df['phase'] == 'Anomaly'].iloc[-1]['timestamp']
df.loc[(df['timestamp'] <= landing_end) & (df['ALT'] < df['rolling_mean']), 'phase'] = 'Landing'

# Detect 'Peak' phase (anomalies at high altitude)
peak_altitude = df['ALT'].max()
df.loc[(df['ALT'] == peak_altitude) & (df['phase'] == 'Anomaly'), 'phase'] = 'Peak'

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['ALT'], label='Altitude')
plt.plot(df['timestamp'], df['rolling_mean'], label='Rolling Mean', linestyle='--')
plt.fill_between(df['timestamp'], df['rolling_mean'] - std_threshold * df['rolling_std'],
                 df['rolling_mean'] + std_threshold * df['rolling_std'], color='gray', alpha=0.3, label='±1 Std Dev')

# Add vertical lines for detected phases
for phase, color in zip(['Launch', 'Peak', 'Landing'], ['green', 'red', 'blue']):
    phase_times = df[df['phase'] == phase]['timestamp']
    for time in phase_times:
       plt.axvline(x=time, color=color, linestyle='-', label=f'{phase} Phase' if phase_times.index.get_loc(time) == 0 else "")

plt.xlabel('Time')
plt.ylabel('Altitude (m)')
plt.legend()
plt.show()

# Output the DataFrame with detected phases
print(df[['timestamp', 'ALT', 'phase']])