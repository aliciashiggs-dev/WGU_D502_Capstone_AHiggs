import os
import glob
import pandas as pd
import statsmodels.formula.api as smf

# Make sure outputs directory exists
os.makedirs("outputs", exist_ok=True)

print("1. INGESTING & COMBINING CITI BIKE CHUNKS")
# Catch all chunked monthly CSV files
trip_files = glob.glob("data/*-citibike-tripdata*.csv")

if not trip_files:
    raise FileNotFoundError("No Citi Bike trip CSV files found in 'data/'. Verify folder.")

print(f"Found {len(trip_files)} chunked files. Loading and combining...")

# Read and combine all chunks into one master dataframe
trip_dfs = [pd.read_csv(file, low_memory=False) for file in trip_files]
master_trips = pd.concat(trip_dfs, ignore_index=True)
print(f"Successfully combined! Total raw trip records across all chunks: {len(master_trips):,}")

print("\n2. AGGREGATING RIDERSHIP BY STATION")
station_ridership = master_trips.groupby(['start_station_id', 'start_station_name']).size().reset_index(name='total_ridership')
print(f"Aggregated into {len(station_ridership):,} unique stations with recorded trips.")

print("\n3. LOADING INFRASTRUCTURE METRICS")
# Point directly to infrastructure file
infra_path = "data/New_York_City_Bike_Routes_20261017.csv"
if not os.path.exists(infra_path):
    # Fallback search if file extension differs
    alt_infra = glob.glob("data/New_York_City_Bike_Routes*.csv")
    infra_path = alt_infra[0] if alt_infra else infra_path

infra_df = pd.read_csv(infra_path, low_memory=False)
print(f"Loaded infrastructure dataset with {len(infra_df):,} rows.")
print("Infrastructure columns found:", infra_df.columns.tolist())