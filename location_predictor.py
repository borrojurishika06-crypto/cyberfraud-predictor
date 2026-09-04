# location_predictor.py
"""
Rule-based cash-out location predictor.

Inputs:
    data/complaint_features.csv
        Full historical dataset, including completed withdrawals.

    data/active_cases_scored.csv
        Currently active / at-risk cases.

Output:
    data/predicted_locations.csv

Method:
    1. Filter historical cases to was_cash_withdrawn == True.
    2. Group historical cases by:
           crime_type + city_changed
    3. Calculate the frequency of each final_hop_city.
    4. For each active case, use the matching historical group to
       predict the top 3 likely cash-out cities.
    5. Generate 3-5 synthetic hotspot coordinates around the
       predicted city's approximate center.

This is a SYNTHETIC prototype and the coordinates are approximate,
not real ATM/branch locations.

Requirements:
    pip install pandas numpy
"""

import os
import random

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

HISTORICAL_FILE = "data/complaint_features.csv"
ACTIVE_FILE = "data/active_cases_scored.csv"
OUTPUT_FILE = "data/predicted_locations.csv"


# ============================================================================
# APPROXIMATE CITY COORDINATES
# ============================================================================

CITY_COORDINATES = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Patna": (25.5941, 85.1376),
    "Kolkata": (22.5726, 88.3639),
    "Ahmedabad": (23.0225, 72.5714),
}


# ============================================================================
# 1. LOAD DATA
# ============================================================================

historical_df = pd.read_csv(HISTORICAL_FILE)
active_df = pd.read_csv(ACTIVE_FILE)


# ============================================================================
# 2. NORMALIZE BOOLEAN COLUMNS
# ============================================================================

def to_bool(value):
    """Convert common CSV representations to Python bool."""
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return str(value).strip().lower() in (
        "true",
        "1",
        "yes",
        "y",
    )


historical_df["was_cash_withdrawn"] = (
    historical_df["was_cash_withdrawn"].apply(to_bool)
)

historical_df["city_changed"] = (
    historical_df["city_changed"].apply(to_bool)
)

active_df["city_changed"] = (
    active_df["city_changed"].apply(to_bool)
)


# ============================================================================
# 3. GET HISTORICAL CASH-WITHDRAWAL CASES
# ============================================================================

"""
Only completed cases are used to learn historical cash-out patterns.

For example, the lookup might effectively contain:

    ("UPI Fraud", True)
        Mumbai       42%
        Bengaluru    31%
        Delhi        15%
        Hyderabad    12%

The percentages become the confidence values for future predictions.
"""

historical_withdrawals = historical_df[
    historical_df["was_cash_withdrawn"] == True
].copy()


if historical_withdrawals.empty:
    raise ValueError(
        "No historical cases with was_cash_withdrawn == True were found."
    )


# ============================================================================
# 4. BUILD HISTORICAL PATTERN LOOKUP
# ============================================================================

"""
Create:

    (crime_type, city_changed)
        ->
        {city: percentage}

This is a simple frequency-based statistical model.
"""

pattern_lookup = {}

group_columns = [
    "crime_type",
    "city_changed",
]

for (crime_type, city_changed), group in historical_withdrawals.groupby(
    group_columns
):

    city_counts = (
        group["final_hop_city"]
        .value_counts()
    )

    total_cases = city_counts.sum()

    # Convert counts into percentages.
    city_percentages = (
        city_counts / total_cases * 100
    )

    pattern_lookup[
        (crime_type, city_changed)
    ] = city_percentages.to_dict()


# ============================================================================
# 5. OVERALL FALLBACK PATTERN
# ============================================================================

"""
It is possible that an active case has a crime_type + city_changed
combination that does not exist in the historical data.

For example, if there are no historical:

    "Investment Fraud" + False

cases.

In that situation, use the overall historical cash-out city distribution
instead of failing.
"""

overall_city_counts = (
    historical_withdrawals["final_hop_city"]
    .value_counts()
)

overall_total = overall_city_counts.sum()

overall_city_percentages = (
    overall_city_counts / overall_total * 100
).to_dict()


# ============================================================================
# 6. GENERATE SYNTHETIC HOTSPOTS
# ============================================================================

def generate_hotspots(city, city_confidence):
    """
    Generate 3-5 synthetic hotspot coordinates around the city's
    approximate center.

    The individual hotspot confidence values sum to the predicted
    city's overall confidence.

    Example:

        Mumbai overall confidence = 60%

        Mumbai hotspot 1 = 24%
        Mumbai hotspot 2 = 18%
        Mumbai hotspot 3 = 18%

        Total = 60%
    """

    if city not in CITY_COORDINATES:
        return []

    base_lat, base_lon = CITY_COORDINATES[city]

    # Randomly create 3-5 hotspot clusters.
    num_hotspots = random.randint(3, 5)

    # Generate random weights and normalize them.
    raw_weights = np.random.dirichlet(
        np.ones(num_hotspots)
    )

    hotspot_confidences = (
        raw_weights * city_confidence
    )

    hotspots = []

    for confidence in hotspot_confidences:

        # Small random jitter around city center.
        #
        # This is intentionally synthetic and represents approximate
        # cluster zones rather than real ATM locations.
        lat_jitter = np.random.normal(
            loc=0,
            scale=0.025
        )

        lon_jitter = np.random.normal(
            loc=0,
            scale=0.025
        )

        hotspot_lat = round(
            base_lat + lat_jitter,
            6
        )

        hotspot_lon = round(
            base_lon + lon_jitter,
            6
        )

        hotspots.append({
            "lat": hotspot_lat,
            "lon": hotspot_lon,
            "confidence": round(
                float(confidence),
                2
            ),
        })

    # Correct tiny floating-point rounding differences so that
    # displayed hotspot confidences add up exactly to the city confidence.
    rounded_sum = sum(
        hotspot["confidence"]
        for hotspot in hotspots
    )

    difference = round(
        city_confidence - rounded_sum,
        2
    )

    hotspots[-1]["confidence"] = round(
        hotspots[-1]["confidence"] + difference,
        2
    )

    return hotspots


# ============================================================================
# 7. PREDICT LOCATIONS FOR ACTIVE CASES
# ============================================================================

prediction_rows = []

for _, active_case in active_df.iterrows():

    complaint_id = active_case["complaint_id"]
    crime_type = active_case["crime_type"]
    city_changed = bool(active_case["city_changed"])

    lookup_key = (
        crime_type,
        city_changed,
    )

    # Try the specific historical pattern first.
    if lookup_key in pattern_lookup:
        city_distribution = pattern_lookup[lookup_key]
    else:
        # Fall back to the overall historical distribution.
        city_distribution = overall_city_percentages

    # ------------------------------------------------------------------------
    # Get top 3 cities
    # ------------------------------------------------------------------------

    sorted_cities = sorted(
        city_distribution.items(),
        key=lambda item: item[1],
        reverse=True
    )

    top_3 = sorted_cities[:3]

    # In the extremely unlikely event that fewer than 3 historical cities
    # exist, fill the remaining positions with None.
    while len(top_3) < 3:
        top_3.append(
            (None, 0.0)
        )

    predicted_city_1, confidence_1 = top_3[0]
    predicted_city_2, confidence_2 = top_3[1]
    predicted_city_3, confidence_3 = top_3[2]

    # ------------------------------------------------------------------------
    # Generate hotspots only for the #1 predicted city.
    # ------------------------------------------------------------------------

    hotspots = generate_hotspots(
        predicted_city_1,
        float(confidence_1)
    )

    # The requested output format has one set of hotspot columns.
    #
    # Therefore, store the generated hotspot cluster points as
    # semicolon-separated values:
    #
    #   hotspot_lat = "19.0762;19.0521;19.0984"
    #   hotspot_lon = "72.8812;72.8563;72.9012"
    #   hotspot_confidence = "21.32;18.44;20.24"
    #
    # This keeps all 3-5 hotspot points in a single CSV row.

    hotspot_lat = ";".join(
        str(hotspot["lat"])
        for hotspot in hotspots
    )

    hotspot_lon = ";".join(
        str(hotspot["lon"])
        for hotspot in hotspots
    )

    hotspot_confidence = ";".join(
        f"{hotspot['confidence']:.2f}"
        for hotspot in hotspots
    )

    prediction_rows.append({
        "complaint_id": complaint_id,

        "predicted_city_1": predicted_city_1,
        "confidence_1": round(float(confidence_1), 2),

        "predicted_city_2": predicted_city_2,
        "confidence_2": round(float(confidence_2), 2),

        "predicted_city_3": predicted_city_3,
        "confidence_3": round(float(confidence_3), 2),

        "hotspot_lat": hotspot_lat,
        "hotspot_lon": hotspot_lon,
        "hotspot_confidence": hotspot_confidence,
    })


# ============================================================================
# 8. CREATE OUTPUT DATAFRAME
# ============================================================================

predicted_locations = pd.DataFrame(
    prediction_rows
)


# ============================================================================
# 9. SAVE RESULTS
# ============================================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

predicted_locations.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================================
# 10. PRINT SUMMARY
# ============================================================================

print("=" * 75)
print("Cash-Out Location Prediction Complete")
print("=" * 75)

print(
    f"Historical withdrawal cases used : "
    f"{len(historical_withdrawals):,}"
)

print(
    f"Active cases scored              : "
    f"{len(active_df):,}"
)

print(
    f"Predictions generated             : "
    f"{len(predicted_locations):,}"
)

print(
    f"\nOutput saved to:\n"
    f"  {OUTPUT_FILE}"
)


# ============================================================================
# 11. PRINT TOP 5 MOST URGENT CASES
# ============================================================================

"""
Use urgency_rank from active_cases_scored.csv so the preview shows
the same five cases that the risk model considers most urgent.
"""

urgent_ids = (
    active_df
    .sort_values("urgency_rank")
    .head(5)["complaint_id"]
)

urgent_predictions = predicted_locations[
    predicted_locations["complaint_id"].isin(urgent_ids)
].copy()

# Restore urgency order.
urgent_predictions = (
    urgent_predictions
    .merge(
        active_df[
            ["complaint_id", "urgency_rank"]
        ],
        on="complaint_id",
        how="left",
    )
    .sort_values("urgency_rank")
)


print("\nTop 5 Most Urgent Cases")
print("=" * 75)

preview_columns = [
    "complaint_id",
    "urgency_rank",
    "predicted_city_1",
    "confidence_1",
    "predicted_city_2",
    "confidence_2",
    "predicted_city_3",
    "confidence_3",
]

print(
    urgent_predictions[
        preview_columns
    ].to_string(index=False)
)

print("=" * 75)