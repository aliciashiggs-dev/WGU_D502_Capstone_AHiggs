import os
import glob
import logging
import pandas as pd
import statsmodels.formula.api as smf
import shapely
from shapely import wkt
import numpy as np
from pyproj import Transformer
import matplotlib
matplotlib.use("Agg")   # save figures to files only; no pop-up windows (avoids a Tk/Tcl error)
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

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
DISTRICT_FILE = "data/Community_Districts_(Water_areas_included)_20261017.csv"   # water-areas version
LAND_AREA_FILE = "data/NYC_districts.csv"   # standard version, used only for district land area (decision #10)

# Converts lon/lat degrees (EPSG:4326) to NY State Plane feet (EPSG:2263), decision #11
TRANSFORMER = Transformer.from_crs(4326, 2263, always_xy=True)

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

# Load the Community District boundaries (water-areas version) and turn each outline into a shape
districts = pd.read_csv(DISTRICT_FILE)
districts["geometry"] = districts["the_geom"].apply(wkt.loads)
logging.info(f"district boundaries loaded: {len(districts)} rows")

# Check districts
# print(districts[["BoroCD", "geometry"]].head(3))
# print(districts["geometry"].iloc[0].geom_type)

# Find which district each station is in: test all stations against one district at a time
stations["district"] = float("nan")
for _, row in districts.iterrows():
    inside = shapely.contains_xy(row["geometry"], stations["lng"], stations["lat"])
    stations.loc[inside, "district"] = row["BoroCD"]
logging.info(f"stations with no district: {stations['district'].isna().sum()}")

# Drop stations in Joint Interest Areas
# The last two digits of BoroCD are the district number; above 25 marks a JIA.
is_jia = stations["district"] % 100 > 25
logging.info(f"stations in JIAs dropped: {is_jia.sum()}")
stations = stations[~is_jia]
log_step("dropped stations in JIAs", stations["trips"].sum())
logging.info(f"stations remaining: {len(stations):,}")

# Add up trips per district and convert to average weekday peak trips
n_weekdays = master_trips["started_at"].dt.normalize().nunique()
logging.info(f"weekdays in analysis period: {n_weekdays}")

district_trips = stations.groupby("district")["trips"].sum().reset_index()
district_trips["district"] = district_trips["district"].astype(int)
district_trips["daily_trips"] = district_trips["trips"] / n_weekdays
logging.info(f"districts with trips: {len(district_trips)}")

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
# Check loaded rows and columns
# print(f"Loaded infrastructure dataset with {len(infra_df):,} rows.")
# print("Infrastructure columns found:", infra_df.columns.tolist())

logging.info(f"bike routes loaded: {len(infra_df):,} segments")

# Keep current, protected segments: "Protected" in either direction column
is_current = infra_df["status"] == "Current"
is_protected = (infra_df["ft_facilit"] == "Protected") | (infra_df["tf_facilit"] == "Protected")
protected = infra_df[is_current & is_protected].copy()
logging.info(f"protected current segments: {len(protected):,}")

# Turn each segment's outline text into a shape, and drop the few segments that have no shape
protected["geometry"] = protected["the_geom"].apply(wkt.loads)
has_shape = ~shapely.is_empty(protected["geometry"])
logging.info(f"segments with empty geometry dropped: {(~has_shape).sum()}")
protected = protected[has_shape].copy()

# Measure each segment's length in miles (decision #11)
def to_feet(coords):
    x, y = TRANSFORMER.transform(coords[:, 0], coords[:, 1])
    return np.column_stack((x, y))

def length_in_miles(geom):
    return shapely.transform(geom, to_feet).length / 5280

protected["miles"] = protected["geometry"].apply(length_in_miles)
logging.info(f"total protected miles: {protected['miles'].sum():.1f}")

# Find each segment's midpoint (decision #12): merge the single-part line, then take the point halfway along it
lines = shapely.line_merge(protected["geometry"].values)
midpoints = shapely.line_interpolate_point(lines, 0.5, normalized=True)
mid_x = shapely.get_x(midpoints)
mid_y = shapely.get_y(midpoints)

# Find which district each midpoint is in, one district at a time (same method as for stations)
protected["district"] = float("nan")
for _, row in districts.iterrows():
    inside = shapely.contains_xy(row["geometry"], mid_x, mid_y)
    protected.loc[inside, "district"] = row["BoroCD"]
logging.info(f"protected segments with no district: {protected['district'].isna().sum()}")

# Drop segments with no district and segments in JIAs
is_jia = protected["district"] % 100 > 25
logging.info(f"protected segments in JIAs dropped: {is_jia.sum()}")
protected = protected[protected["district"].notna() & ~is_jia]
logging.info(f"protected segments kept: {len(protected):,} ({protected['miles'].sum():.1f} miles)")

# Add up protected miles per district
lane_miles = protected.groupby("district")["miles"].sum().reset_index()
lane_miles["district"] = lane_miles["district"].astype(int)
logging.info(f"districts with protected lanes: {len(lane_miles)}")

# Protected lane density = protected miles per square mile of LAND area
# The water-areas file's area includes water, so land area comes from the standard file.
land_area = pd.read_csv(LAND_AREA_FILE)
# Shape_Area is text with commas (ex: "172,077,374.82") in square feet: 27,878,400 sq ft = 1 sq mile
land_area["land_sq_miles"] = land_area["Shape_Area"].str.replace(",", "", regex=False).astype(float) / 27_878_400
land_area = land_area[["BoroCd", "land_sq_miles"]].rename(columns={"BoroCd": "district"})
lane_miles = lane_miles.merge(land_area, on="district", how="left")
lane_miles["lane_density"] = lane_miles["miles"] / lane_miles["land_sq_miles"]
logging.info(f"districts missing land area: {lane_miles['land_sq_miles'].isna().sum()}")

# Build the district table: one row per district with trips (Y) and lane density (X)
district_data = district_trips.merge(
    lane_miles[["district", "miles", "lane_density"]], on="district", how="left"
)
no_lanes = district_data["lane_density"].isna()
logging.info(f"districts with trips but no protected lanes (set to 0): {district_data.loc[no_lanes, 'district'].tolist()}")
district_data[["miles", "lane_density"]] = district_data[["miles", "lane_density"]].fillna(0)

district_data.to_csv("outputs/district_data.csv", index=False)
logging.info(f"district table saved: {len(district_data)} rows")

# Exploratory data analysis (EDA): summary statistics for the two model variables
summary = district_data[["daily_trips", "lane_density"]].describe().round(2)
print("\n4. SUMMARY STATISTICS")
print(summary)
summary.to_csv("outputs/summary_statistics.csv")

# Histograms: the shape of each variable
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(data=district_data, x="daily_trips", bins=15, ax=axes[0])
axes[0].set_title("Average weekday peak trips per district")
axes[0].set_xlabel("Daily peak trips")
axes[0].set_ylabel("Number of districts")
sns.histplot(data=district_data, x="lane_density", bins=15, ax=axes[1])
axes[1].set_title("Protected lane density per district")
axes[1].set_xlabel("Protected miles per square mile of land")
axes[1].set_ylabel("Number of districts")
plt.tight_layout()
plt.savefig("outputs/fig_distributions.png", dpi=150)
plt.close()

# Scatter plot: lane density (X) vs. daily peak trips (Y), one dot per district
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=district_data, x="lane_density", y="daily_trips", ax=ax)
ax.set_title("Protected lane density vs. weekday peak trips, by district")
ax.set_xlabel("Protected miles per square mile of land")
ax.set_ylabel("Average weekday peak trips (member riders)")
plt.tight_layout()
plt.savefig("outputs/fig_scatter.png", dpi=150)
plt.close()

# Fit the simple linear regression (OLS): daily_trips depends on lane_density
model = smf.ols("daily_trips ~ lane_density", data=district_data).fit()
print("\n5. OLS REGRESSION RESULTS")
print(model.summary())
with open("outputs/model_summary.txt", "w") as f:
    f.write(model.summary().as_text())

# The key results from the model
slope = model.params["lane_density"]
intercept = model.params["Intercept"]
p_value = model.pvalues["lane_density"]
r_squared = model.rsquared
rmse = np.sqrt((model.resid ** 2).mean())
logging.info(f"OLS: slope={slope:.2f}, intercept={intercept:.2f}, p-value={p_value:.2e}, R-squared={r_squared:.3f}, RMSE={rmse:.1f}, n={int(model.nobs)}")

# Residual plot: check linearity and constant spread
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(x=model.fittedvalues, y=model.resid, ax=ax)
ax.axhline(0, color="gray", linestyle="--")
ax.set_title("Residuals vs. fitted values")
ax.set_xlabel("Fitted (predicted) daily peak trips")
ax.set_ylabel("Residual (actual minus predicted)")
plt.tight_layout()
plt.savefig("outputs/fig_residuals.png", dpi=150)
plt.close()

# Formal assumption tests
bp_stat, bp_p, bp_f, bp_f_p = het_breuschpagan(model.resid, model.model.exog)
sw_stat, sw_p = stats.shapiro(model.resid)
logging.info(f"Breusch-Pagan (constant spread) p-value: {bp_p:.4f}")
logging.info(f"Shapiro-Wilk (normal residuals) p-value: {sw_p:.4f}")

# Robustness check: same model, but standard errors that do not assume steady spread (HC3)
model_robust = smf.ols("daily_trips ~ lane_density", data=district_data).fit(cov_type="HC3")
robust_low, robust_high = model_robust.conf_int().loc["lane_density"]
logging.info(f"Robust (HC3): slope={model_robust.params['lane_density']:.2f}, std err={model_robust.bse['lane_density']:.2f}, p-value={model_robust.pvalues['lane_density']:.2e}, 95% CI [{robust_low:.0f}, {robust_high:.0f}]")
with open("outputs/model_summary_robust.txt", "w") as f:
    f.write(model_robust.summary().as_text())

# Influence: which districts have the most pull on the fitted line (Cook's distance)
influence = model.get_influence()
district_data["cooks_d"] = influence.cooks_distance[0]
threshold = 4 / len(district_data)
influential = district_data[district_data["cooks_d"] > threshold].sort_values("cooks_d", ascending=False)
logging.info(f"Cook's distance threshold (4/n): {threshold:.2f}; districts above it: {influential['district'].tolist()}")
influential[["district", "daily_trips", "lane_density", "cooks_d"]].round(2).to_csv("outputs/influential_districts.csv", index=False)

# Final figure: scatter plot with the OLS line, influential districts labeled
fig, ax = plt.subplots(figsize=(8, 5.5))
sns.scatterplot(data=district_data, x="lane_density", y="daily_trips", ax=ax)
x_line = np.linspace(0, district_data["lane_density"].max(), 100)
ax.plot(x_line, intercept + slope * x_line, color="red",
        label=f"OLS line (slope = {slope:.0f}, R² = {r_squared:.2f})")
for _, row in influential.iterrows():
    ax.annotate(int(row["district"]), (row["lane_density"], row["daily_trips"]),
                textcoords="offset points", xytext=(5, 5), fontsize=9)
ax.set_title("Protected lane density vs. weekday peak trips, by district")
ax.set_xlabel("Protected miles per square mile of land")
ax.set_ylabel("Average weekday peak trips (member riders)")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/fig_scatter_line.png", dpi=150)
plt.close()

# Call save_attrition at end of code to ensure all log steps are saved for the analysis
save_attrition()

# # Test that helper items work as expected
# # Highlight & # out above code & run main.py to test logging helpers in lines 10:35
# log_step("test step A", 1000)
# log_step("test step B", 750)
# log_step("test step B", 750)
# save_attrition()