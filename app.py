
# app.py

"""
Cybercrime Cash-Out Early Warning System
----------------------------------------

Streamlit dashboard for cybercrime investigation officers.

Input files:
    data/active_cases_scored.csv
    data/predicted_locations.csv

Requirements:
    pip install streamlit pandas folium streamlit-folium

Run:
    streamlit run app.py
"""

from datetime import datetime

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Cybercrime Cash-Out Early Warning System",
    page_icon="🚨",
    layout="wide",
)


# ============================================================================
# CONSTANTS
# ============================================================================

ACTIVE_FILE = "data/active_cases_scored.csv"
LOCATIONS_FILE = "data/predicted_locations.csv"


# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_data():
    """Load and merge active case scores with predicted locations."""

    active_df = pd.read_csv(ACTIVE_FILE)
    locations_df = pd.read_csv(LOCATIONS_FILE)

    # Merge both datasets using complaint_id.
    merged_df = active_df.merge(
        locations_df,
        on="complaint_id",
        how="left",
    )

    # Ensure numeric fields are correctly typed.
    numeric_columns = [
        "fraud_amount",
        "minutes_remaining",
        "urgency_rank",
        "predicted_minutes_to_cashout",
        "confidence_1",
    ]

    for column in numeric_columns:
        if column in merged_df.columns:
            merged_df[column] = pd.to_numeric(
                merged_df[column],
                errors="coerce",
            )

    return merged_df


try:
    df = load_data()

except FileNotFoundError as error:
    st.error(
        f"Could not find one of the required CSV files: {error}"
    )
    st.stop()


# ============================================================================
# PAGE HEADER
# ============================================================================

st.title("🚨 Cybercrime Cash-Out Early Warning System")

st.markdown(
    """
    **Early-warning intelligence dashboard for identifying active money trails
    that may be approaching cash withdrawal.**
    """
)


# ============================================================================
# SIDEBAR FILTER
# ============================================================================

st.sidebar.header("🔎 Filters")

crime_types = sorted(
    df["crime_type"]
    .dropna()
    .unique()
    .tolist()
)

selected_crime = st.sidebar.selectbox(
    "Crime Type",
    options=["All Crime Types"] + crime_types,
)


# Apply filter.
if selected_crime == "All Crime Types":
    filtered_df = df.copy()

else:
    filtered_df = df[
        df["crime_type"] == selected_crime
    ].copy()


# ============================================================================
# METRIC BAR
# ============================================================================

total_active = len(filtered_df)

high_urgency = len(
    filtered_df[
        filtered_df["minutes_remaining"] < 30
    ]
)

total_amount_at_risk = filtered_df[
    "fraud_amount"
].sum()

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "Total Active Cases",
        f"{total_active:,}",
    )

with metric2:
    st.metric(
        "High Urgency (< 30 min)",
        f"{high_urgency:,}",
    )

with metric3:
    st.metric(
        "Total Amount at Risk",
        f"₹{total_amount_at_risk:,.0f}",
    )


st.divider()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def urgency_indicator(minutes):
    """
    Return an emoji + text urgency indicator.

    RED    : < 30 minutes
    ORANGE : 30-90 minutes
    YELLOW : > 90 minutes
    """

    if pd.isna(minutes):
        return "⚪ UNKNOWN"

    if minutes < 30:
        return "🔴 HIGH"

    elif minutes <= 90:
        return "🟠 MEDIUM"

    else:
        return "🟡 LOW"


def format_minutes(minutes):
    """Format remaining minutes cleanly."""

    if pd.isna(minutes):
        return "Unknown"

    return f"{minutes:.1f} minutes"


def format_confidence(confidence):
    """Format prediction confidence as a percentage."""

    if pd.isna(confidence):
        return "N/A"

    return f"{confidence:.1f}%"


def parse_hotspot_coordinates(value):
    """
    Parse the semicolon-separated hotspot coordinates produced by
    location_predictor.py.

    Example:
        "19.0762;19.0521;19.0984"
    """

    if pd.isna(value):
        return []

    try:
        return [
            float(x.strip())
            for x in str(value).split(";")
            if x.strip()
        ]

    except (ValueError, TypeError):
        return []


# ============================================================================
# TOP 10 URGENT CASES
# ============================================================================

st.subheader("🚨 Top 10 Most Urgent Active Cases")
st.caption("📊 Model calibration note: hop timing and chain-depth thresholds in this model are aligned with I4C's published fraud-layering patterns, where most mule-chain transfers complete within 45–60 minutes of the original fraud.")

if filtered_df.empty:

    st.info(
        "No active cases match the selected crime type."
    )

    st.stop()


# Sort by urgency rank and keep the top 10.
top_cases = (
    filtered_df
    .sort_values("urgency_rank")
    .head(10)
    .copy()
)


# ============================================================================
# MAIN TWO-COLUMN LAYOUT
# ============================================================================

left_column, right_column = st.columns(
    [1.15, 1],
    gap="large",
)


# ============================================================================
# LEFT COLUMN — RANKED CASE LIST
# ============================================================================

with left_column:

    st.markdown(
        "### Ranked Cases"
    )

    # Create complaint ID options for the case selector.
    complaint_ids = top_cases[
        "complaint_id"
    ].tolist()

    # Keep selection in session state so the selected case doesn't
    # unexpectedly change when the page reruns.
    if (
        "selected_complaint" not in st.session_state
        or st.session_state["selected_complaint"]
        not in complaint_ids
    ):
        st.session_state["selected_complaint"] = (
            complaint_ids[0]
        )

    selected_complaint = st.radio(
        "Select a case to investigate:",
        options=complaint_ids,
        key="selected_complaint",
        horizontal=False,
    )

    st.markdown("---")

    # ========================================================================
    # DISPLAY EACH CASE
    # ========================================================================

    for _, case in top_cases.iterrows():

        complaint_id = case["complaint_id"]

        minutes = case["minutes_remaining"]

        urgency_rank = int(case["urgency_rank"])

        urgency = urgency_indicator(
            minutes
        )

        is_selected = (
            complaint_id == selected_complaint
        )

        # --------------------------------------------------------------------
        # Case heading
        # --------------------------------------------------------------------

        if is_selected:

            st.markdown(
                f"### 👉 {complaint_id}"
            )

        else:

            st.markdown(
                f"**{complaint_id}**"
            )

        # --------------------------------------------------------------------
        # Urgency + estimated time
        # --------------------------------------------------------------------

        st.markdown(
            f"{urgency} &nbsp;&nbsp; "
            f"**Estimated time to cash-out:** "
            f"{format_minutes(minutes)}"
        )

        # --------------------------------------------------------------------
        # SHAP-based explanation
        # --------------------------------------------------------------------

        reason = case.get(
            "top_reason_text",
            "No explanation available.",
        )

        st.markdown(
            f"**Why this case is urgent:** {reason}"
        )

        # --------------------------------------------------------------------
        # Predicted location
        # --------------------------------------------------------------------

        predicted_city = case.get(
            "predicted_city_1",
            "Unknown",
        )

        confidence = case.get(
            "confidence_1",
            float("nan"),
        )

        st.markdown(
            f"📍 **Predicted cash-out city:** "
            f"{predicted_city} "
            f"({format_confidence(confidence)} confidence)"
        )

        # --------------------------------------------------------------------
        # Financial and crime information
        # --------------------------------------------------------------------

        fraud_amount = case.get(
            "fraud_amount",
            float("nan"),
        )

        crime_type = case.get(
            "crime_type",
            "Unknown",
        )

        st.markdown(
            f"💰 **Fraud amount:** "
            f"₹{fraud_amount:,.0f} &nbsp;&nbsp; "
            f"| &nbsp;&nbsp; "
            f"**Crime:** {crime_type}"
        )

        # --------------------------------------------------------------------
        # BRANCH ALERT BUTTON
        # --------------------------------------------------------------------
        # Only the top 5 most urgent cases receive the alert button.
        # No real backend, webhook, or external service is contacted.
        # The button simply displays a convincing confirmation message.

        if urgency_rank <= 5:

            if st.button(
                "🚨 Send Alert to Branch",
                key=f"branch_alert_{complaint_id}",
                use_container_width=True,
            ):

                current_time = datetime.now().strftime(
                    "%d-%m-%Y %I:%M:%S %p"
                )

                st.success(
                    f"✅ Alert sent to {predicted_city} Branch Network "
                    f"at {current_time} — Case {complaint_id} "
                    f"flagged for priority monitoring."
                )

        # Separator between cases
        st.markdown("---")


# ============================================================================
# RIGHT COLUMN — SELECTED CASE MAP
# ============================================================================

with right_column:

    st.subheader("📍 Predicted Cash-Out Location")

    selected_case = filtered_df[
        filtered_df["complaint_id"] == selected_complaint
    ]

    if selected_case.empty:

        st.warning(
            "Selected case could not be found."
        )

    else:

        selected_case = selected_case.iloc[0]

        predicted_city = selected_case.get(
            "predicted_city_1",
            "Unknown",
        )

        confidence = selected_case.get(
            "confidence_1",
            float("nan"),
        )

        # --------------------------------------------------------------------
        # Parse hotspot coordinates
        # --------------------------------------------------------------------

        hotspot_lats = parse_hotspot_coordinates(
            selected_case.get("hotspot_lat")
        )

        hotspot_lons = parse_hotspot_coordinates(
            selected_case.get("hotspot_lon")
        )

        hotspot_confidences = parse_hotspot_coordinates(
            selected_case.get("hotspot_confidence")
        )

        # Make sure coordinates are paired correctly.
        num_points = min(
            len(hotspot_lats),
            len(hotspot_lons),
        )

        # --------------------------------------------------------------------
        # Choose map center
        # --------------------------------------------------------------------

        if num_points > 0:

            map_lat = sum(
                hotspot_lats[:num_points]
            ) / num_points

            map_lon = sum(
                hotspot_lons[:num_points]
            ) / num_points

        else:

            # Fall back to the final hop coordinates if hotspot
            # coordinates aren't available.
            map_lat = selected_case.get(
                "final_hop_lat",
                20.5937,
            )

            map_lon = selected_case.get(
                "final_hop_lon",
                78.9629,
            )

            if pd.isna(map_lat):
                map_lat = 20.5937

            if pd.isna(map_lon):
                map_lon = 78.9629

        # --------------------------------------------------------------------
        # Case summary above map
        # --------------------------------------------------------------------

        st.markdown(
            f"**Case:** `{selected_complaint}`"
        )

        st.markdown(
            f"**Predicted city:** {predicted_city}"
        )

        st.markdown(
            f"**Historical confidence:** "
            f"{format_confidence(confidence)}"
        )

        st.caption(
            "⚠️ Hotspots are synthetic probability zones generated "
            "around the city's approximate center. They are not real "
            "ATM or branch locations."
        )

        # --------------------------------------------------------------------
        # Create Folium map
        # --------------------------------------------------------------------

        investigation_map = folium.Map(
            location=[
                map_lat,
                map_lon,
            ],
            zoom_start=12,
            tiles="OpenStreetMap",
        )

        # Add one marker for every predicted hotspot.
        for i in range(num_points):

            lat = hotspot_lats[i]

            lon = hotspot_lons[i]

            if i < len(hotspot_confidences):

                hotspot_confidence = (
                    hotspot_confidences[i]
                )

            else:

                hotspot_confidence = 0

            popup_text = (
                f"<b>Predicted Cash-Out Hotspot</b><br>"
                f"City: {predicted_city}<br>"
                f"Overall city confidence: "
                f"{format_confidence(confidence)}<br>"
                f"Hotspot confidence: "
                f"{hotspot_confidence:.2f}%"
            )

            folium.Marker(
                location=[
                    lat,
                    lon,
                ],
                popup=folium.Popup(
                    popup_text,
                    max_width=300,
                ),
                tooltip=(
                    f"{predicted_city} hotspot "
                    f"({hotspot_confidence:.2f}%)"
                ),
                icon=folium.Icon(
                    icon="warning",
                    prefix="glyphicon",
                ),
            ).add_to(investigation_map)

        # Display map.
        st_folium(
            investigation_map,
            width=None,
            height=500,
            returned_objects=[],
        )


# ============================================================================
# ADDITIONAL PREDICTION DETAILS
# ============================================================================

st.divider()

st.subheader("📊 Selected Case Prediction Details")

selected_details = filtered_df[
    filtered_df["complaint_id"] == selected_complaint
]

if not selected_details.empty:

    case = selected_details.iloc[0]

    detail1, detail2, detail3, detail4 = st.columns(4)

    with detail1:

        st.metric(
            "Urgency Rank",
            f"#{int(case['urgency_rank'])}",
        )

    with detail2:

        st.metric(
            "Predicted Time to Cash-Out",
            f"{case['predicted_minutes_to_cashout']:.1f} min",
        )

    with detail3:

        st.metric(
            "Minutes Remaining",
            f"{case['minutes_remaining']:.1f}",
        )

    with detail4:

        st.metric(
            "City Confidence",
            format_confidence(
                case["confidence_1"]
            ),
        )

    # Show alternative predicted cities.
    st.markdown("**Alternative predicted cities:**")

    alternatives = []

    for i in [2, 3]:

        city = case.get(
            f"predicted_city_{i}"
        )

        confidence = case.get(
            f"confidence_{i}"
        )

        if (
            pd.notna(city)
            and pd.notna(confidence)
        ):

            alternatives.append(
                f"{city} ({confidence:.1f}%)"
            )

    if alternatives:

        st.write(
            " • ".join(alternatives)
        )

    else:

        st.write(
            "No alternative predictions available."
        )


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

st.caption(
    "Synthetic hackathon prototype • Predictions are statistical estimates "
    "for demonstration purposes and should not be treated as real-world "
    "financial or law-enforcement intelligence."
)
