import os
import glob
import logging
import pandas as pd
import statsmodels.formula.api as smf

# Make sure outputs directory exists (must happen before logging opens its file in outputs/)
os.makedirs("outputs", exist_ok=True)

# Set up logging to see changes as I work through the analysis and write a timestamped record to outputs/pipeline_log.txt.
# Prints to console. Each run starts a fresh log file because mode="w"
# Note to self to review after running analysis for Task 2 governance evidence.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler("outputs/pipeline_log.txt", mode="w"),
        logging.StreamHandler(),
    ],
)

# Set up attrition List.
# This is an empty list that collects one entry per step
attrition = []

# This function appends attrition for each step of the analysis with the number of rows after that step is completed
# and writes a log message in console output and log file.
def log_step(step, n_rows):
    logging.info(f"{step}: {n_rows:,} rows")
    attrition.append({"step": step, "rows": n_rows})

# Define save_attrition function to convert attrition list into a table and save to outputs/attrition_table.csv
# Note to self: Call once at the end, when attrition holds every step. Each call rewrites the entire file.
def save_attrition():
    pd.DataFrame(attrition).to_csv("outputs/attrition_table.csv", index=False)

# Define trip data parameters
TRIP_COLUMNS = ["started_at", "member_casual", "start_station_id", "start_lat", "start_lng"]
START_DATE = "2026-06-01"      # first day of the analysis period
END_DATE = "2026-09-01"        # the day AFTER the last day, so trips must start before this
EXCLUDED_DATE = "2026-07-03"   # Observed Independence Day holiday, not a normal commuting day
AM_START_HOUR = 7              # morning window: trips starting 7:00 to 9:59
AM_END_HOUR = 10
PM_START_HOUR = 16             # evening window: trips starting 16:00 to 19:59
PM_END_HOUR = 20

print("1. INGESTING & COMBINING CITI BIKE CHUNKS")
# Find the 2026 chunked monthly CSV files
trip_files = glob.glob("data/2026*-citibike-tripdata*.csv")

if not trip_files:
    raise FileNotFoundError("No Citi Bike trip CSV files found in 'data/'. Verify folder.")

print(f"Found {len(trip_files)} chunked files. Loading and combining...")

# Read and combine all chunks into one master dataframe
# usecols added to read only the columns I need for my analysis
trip_dfs = [
    pd.read_csv(file, usecols=TRIP_COLUMNS, dtype={"start_station_id": str}, low_memory=False)
    for file in trip_files
]
master_trips = pd.concat(trip_dfs, ignore_index=True)

# add a line to attrition to show result of combining data chunks
log_step("raw trips, all 2026 files", len(master_trips))

# Convert started_at from text to date/time so that I can filter by date, weekday, and hour
master_trips["started_at"] = pd.to_datetime(master_trips["started_at"], format="mixed")

# Keep only trips that started inside the analysis period
master_trips = master_trips[(master_trips["started_at"] >= START_DATE) & (master_trips["started_at"] < END_DATE)]
log_step("started Jun 1 - Aug 31, 2026", len(master_trips))

# Keep weekdays only, Monday-Friday
master_trips = master_trips[master_trips["started_at"].dt.dayofweek < 5]
log_step("weekdays only (Mon-Fri)", len(master_trips))

# Keep annual members only
master_trips = master_trips[master_trips["member_casual"] == "member"]
log_step("members only", len(master_trips))

# Drop excluded holiday, July 3
master_trips = master_trips[master_trips["started_at"].dt.normalize() != pd.Timestamp(EXCLUDED_DATE)]
log_step("excluded July 3 holiday", len(master_trips))

# Keep peak commute hours: morning and evening windows
hour = master_trips["started_at"].dt.hour
in_am = (hour >= AM_START_HOUR) & (hour < AM_END_HOUR)
in_pm = (hour >= PM_START_HOUR) & (hour < PM_END_HOUR)
master_trips = master_trips[in_am | in_pm]
log_step("peak hours (7-10 and 16-20)", len(master_trips))

# Drop trips with no start station or location that cannot be mapped to a district
master_trips = master_trips.dropna(subset=["start_station_id", "start_lat", "start_lng"])
log_step("dropped trips with no start station", len(master_trips))

# One row per start station: trip count and location
stations = master_trips.groupby("start_station_id").agg(
    trips=("start_lat", "size"),
    lat=("start_lat", "first"),
    lng=("start_lng", "first"),
).reset_index()
logging.info(f"unique start stations: {len(stations):,}")

# check formatting for start times
# print(master_trips["started_at"].dtype)
# print(master_trips["started_at"].head())

# print("\n2. AGGREGATING RIDERSHIP BY STATION")
# station_ridership = master_trips.groupby(['start_station_id', 'start_station_name']).size().reset_index(name='total_ridership')
# print(f"Aggregated into {len(station_ridership):,} unique stations with recorded trips.")

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

# Call save_attrition at end of code to ensure all log steps are saved for the analysis
save_attrition()

# # Test that helper items work as expected
# # Highlight & # out above code & run main.py to test logging helpers in lines 10:35
# log_step("test step A", 1000)
# log_step("test step B", 750)
# log_step("test step B", 750)
# save_attrition()