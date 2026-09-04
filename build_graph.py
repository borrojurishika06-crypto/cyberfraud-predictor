# build_graph.py
"""
Build complaint-level money-trail graphs and features.

Input:
    complaints.csv
    transaction_hops.csv

Output:
    data/complaint_features.csv

Requirements:
    pip install pandas networkx
"""

import os
import random

import networkx as nx
import pandas as pd


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)

COMPLAINTS_FILE = "complaints.csv"
TRANSACTIONS_FILE = "transaction_hops.csv"
OUTPUT_FILE = "data/complaint_features.csv"


# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------

complaints_df = pd.read_csv(COMPLAINTS_FILE)
transactions_df = pd.read_csv(TRANSACTIONS_FILE)

# Convert timestamp columns to datetime objects so that time differences
# can be calculated correctly.
complaints_df["complaint_time"] = pd.to_datetime(
    complaints_df["complaint_time"]
)

transactions_df["hop_timestamp"] = pd.to_datetime(
    transactions_df["hop_timestamp"]
)


# ---------------------------------------------------------------------------
# 2. BUILD ONE DIRECTED GRAPH PER COMPLAINT
# ---------------------------------------------------------------------------

# Dictionary:
#     complaint_id -> networkx.DiGraph
#
# Each graph represents the money trail for exactly one complaint.
complaint_graphs = {}


for complaint_id, chain in transactions_df.groupby("complaint_id"):

    # Sort hops chronologically / by hop number to guarantee
    # the correct direction of money movement.
    chain = chain.sort_values("hop_number")

    graph = nx.DiGraph()

    # Add every account as a node.
    for _, hop in chain.iterrows():
        graph.add_node(
            hop["account_id"],
            city=hop["hop_city"]
        )

    # Each hop points from the previous account to the current account.
    #
    # Example:
    #   ACC123 -> ACC456 -> ACC789
    #
    # The first account has no previous account, so there is no edge
    # before the first hop.
    for i in range(1, len(chain)):

        previous_hop = chain.iloc[i - 1]
        current_hop = chain.iloc[i]

        graph.add_edge(
            previous_hop["account_id"],
            current_hop["account_id"],
            amount=current_hop["amount_at_this_hop"],
            timestamp=current_hop["hop_timestamp"],
            city=current_hop["hop_city"]
        )

    complaint_graphs[complaint_id] = graph


# ---------------------------------------------------------------------------
# 3. CALCULATE COMPLAINT-LEVEL FEATURES
# ---------------------------------------------------------------------------

# "Right now" for the demo is defined as:
#
# maximum hop timestamp across the entire dataset
#     +
# random offset between 0 and 120 minutes
#
# This gives us a reproducible simulated current time.
max_hop_timestamp = transactions_df["hop_timestamp"].max()

simulated_now = (
    max_hop_timestamp
    + pd.Timedelta(minutes=random.randint(0, 120))
)


feature_rows = []


for complaint_id, chain in transactions_df.groupby("complaint_id"):

    # Sort by hop number to ensure the first and last rows are correct.
    chain = chain.sort_values("hop_number").reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Basic chain information
    # -----------------------------------------------------------------------

    hop_count = len(chain)

    first_hop_time = chain.iloc[0]["hop_timestamp"]
    last_hop_time = chain.iloc[-1]["hop_timestamp"]

    # Total time from first hop to final hop.
    total_elapsed_minutes = (
        last_hop_time - first_hop_time
    ).total_seconds() / 60

    # -----------------------------------------------------------------------
    # Time gaps between consecutive hops
    # -----------------------------------------------------------------------

    if hop_count > 1:

        # Calculate the timestamp difference between consecutive hops.
        time_gaps = (
            chain["hop_timestamp"]
            .diff()
            .dropna()
            .dt.total_seconds()
            / 60
        )

        avg_minutes_between_hops = time_gaps.mean()
        min_minutes_between_hops = time_gaps.min()

    else:
        # Chains are expected to have 2–7 hops, but handle a single-hop
        # chain safely in case the input data changes.
        avg_minutes_between_hops = 0.0
        min_minutes_between_hops = 0.0

    # -----------------------------------------------------------------------
    # Victim and final-hop information
    # -----------------------------------------------------------------------

    complaint_row = complaints_df[
        complaints_df["complaint_id"] == complaint_id
    ]

    if complaint_row.empty:
        # Skip a transaction chain if its complaint record doesn't exist.
        continue

    complaint_row = complaint_row.iloc[0]

    victim_city = complaint_row["victim_city"]

    final_hop = chain.iloc[-1]

    final_hop_city = final_hop["hop_city"]

    # True when the money ends up in a different city from the victim.
    city_changed = final_hop_city != victim_city

    # -----------------------------------------------------------------------
    # Minutes since the final hop
    # -----------------------------------------------------------------------

    minutes_since_last_hop = (
        simulated_now - last_hop_time
    ).total_seconds() / 60

    # -----------------------------------------------------------------------
    # Final-hop withdrawal information
    # -----------------------------------------------------------------------

    was_cash_withdrawn = final_hop["was_cash_withdrawn"]

    final_hop_lat = final_hop["withdrawal_lat"]
    final_hop_lon = final_hop["withdrawal_lon"]

    # -----------------------------------------------------------------------
    # Store feature row
    # -----------------------------------------------------------------------

    feature_rows.append({
        "complaint_id": complaint_id,
        "hop_count": hop_count,
        "total_elapsed_minutes": round(
            total_elapsed_minutes, 2
        ),
        "avg_minutes_between_hops": round(
            avg_minutes_between_hops, 2
        ),
        "min_minutes_between_hops": round(
            min_minutes_between_hops, 2
        ),
        "city_changed": city_changed,
        "fraud_amount": complaint_row["fraud_amount"],
        "crime_type": complaint_row["crime_type"],
        "minutes_since_last_hop": round(
            minutes_since_last_hop, 2
        ),
        "was_cash_withdrawn": was_cash_withdrawn,
        "final_hop_lat": final_hop_lat,
        "final_hop_lon": final_hop_lon,
        "final_hop_city": final_hop_city,
    })


# ---------------------------------------------------------------------------
# 4. CREATE FEATURE DATAFRAME
# ---------------------------------------------------------------------------

complaint_features = pd.DataFrame(feature_rows)


# ---------------------------------------------------------------------------
# 5. SAVE FEATURES
# ---------------------------------------------------------------------------

# Create the output directory if it doesn't already exist.
os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

complaint_features.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------------------------
# 6. PRINT SUMMARY
# ---------------------------------------------------------------------------

print("=" * 65)
print("Cybercrime Money-Trail Graph Processing Complete")
print("=" * 65)

print(f"Complaints processed : {len(complaint_features):,}")
print(f"Graphs created       : {len(complaint_graphs):,}")
print(f"Total transaction hops: {len(transactions_df):,}")

print(f"\nSimulated current time: {simulated_now}")

print(f"\nOutput saved to:")
print(f"  {OUTPUT_FILE}")

print("\nFirst 5 rows of complaint_features:")
print("-" * 65)
print(complaint_features.head(5).to_string(index=False))

print("=" * 65)