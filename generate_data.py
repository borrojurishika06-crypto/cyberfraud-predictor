# generate_data.py
"""
Synthetic Cybercrime Complaint & Money-Trail Data Generator

Generates:
    1. complaints.csv
       - One row per cybercrime complaint

    2. transaction_hops.csv
       - One row per bank-account hop in each synthetic mule chain

All data is SYNTHETIC and intended only for prototyping/testing.

Requirements:
    pip install pandas faker numpy
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

SEED = 42
NUM_COMPLAINTS = 800

# Set all random generators to fixed seeds for reproducibility.
random.seed(SEED)
np.random.seed(SEED)
Faker.seed(SEED)

fake = Faker("en_IN")


# ---------------------------------------------------------------------------
# CITY DATA
# Approximate coordinates are used only for synthetic GPS generation.
# ---------------------------------------------------------------------------

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

CITIES = list(CITY_COORDINATES.keys())

CRIME_TYPES = [
    "UPI Fraud",
    "Digital Arrest Scam",
    "Loan App Fraud",
    "Investment Fraud",
    "OTP Fraud",
]

# Synthetic bank/IFSC prefixes.
# These are deliberately fictionalized identifiers.
FAKE_BANK_CODES = [
    "FAKE000",
    "SYNTH000",
    "TEST000",
    "DEMO000",
    "MOCK000",
]


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def generate_complaint_id(index):
    """Generate a unique complaint ID."""
    return f"CMP{index:05d}"


def generate_account_id(used_accounts):
    """
    Generate a unique synthetic bank account ID.

    Example:
        ACC483920
    """
    while True:
        account_id = f"ACC{random.randint(100000, 999999)}"

        if account_id not in used_accounts:
            used_accounts.add(account_id)
            return account_id


def generate_fake_ifsc():
    """
    Generate a synthetic IFSC-like identifier.

    Example:
        FAKE0004821

    These identifiers are synthetic and should not be interpreted
    as real bank IFSC codes.
    """
    bank_code = random.choice(FAKE_BANK_CODES)
    branch_number = random.randint(1000, 9999)

    return f"{bank_code}{branch_number}"


def generate_fraud_amount():
    """
    Generate a realistically skewed fraud amount.

    A lognormal distribution creates:
        - Many relatively small/medium frauds
        - Fewer large frauds
        - Very few extremely large frauds

    The result is clipped to:
        ₹15,000 - ₹500,000
    """

    # Median is approximately around ₹55,000.
    amount = np.random.lognormal(
        mean=np.log(55000),
        sigma=0.85
    )

    amount = np.clip(amount, 15000, 500000)

    # Round to the nearest ₹100 for a more realistic transaction amount.
    amount = round(amount / 100) * 100

    return int(amount)


def generate_random_timestamp(days_back=30):
    """Generate a random timestamp within the previous N days."""

    now = datetime.now()

    random_seconds = random.randint(
        0,
        days_back * 24 * 60 * 60
    )

    return now - timedelta(seconds=random_seconds)


def choose_hop_city(victim_city):
    """
    Choose a city for a money-trail hop.

    Sometimes the money stays within the victim's city,
    but frequently moves to another city.
    """

    # 45% chance to remain in the victim's city.
    if random.random() < 0.45:
        return victim_city

    # Otherwise choose another city.
    other_cities = [
        city for city in CITIES
        if city != victim_city
    ]

    return random.choice(other_cities)


def generate_gps(city):
    """
    Generate synthetic GPS coordinates around a city's approximate center.

    Jitter is small enough to represent different ATM locations
    while keeping the point within/near the city.
    """

    base_lat, base_lon = CITY_COORDINATES[city]

    # Approximately a few kilometers of jitter.
    latitude_jitter = np.random.normal(0, 0.025)
    longitude_jitter = np.random.normal(0, 0.025)

    latitude = round(base_lat + latitude_jitter, 6)
    longitude = round(base_lon + longitude_jitter, 6)

    return latitude, longitude


# ---------------------------------------------------------------------------
# GENERATE COMPLAINTS
# ---------------------------------------------------------------------------

complaints = []

for i in range(1, NUM_COMPLAINTS + 1):

    victim_city = random.choice(CITIES)

    complaint = {
        "complaint_id": generate_complaint_id(i),
        "victim_city": victim_city,
        "fraud_amount": generate_fraud_amount(),
        "complaint_time": generate_random_timestamp(),
        "crime_type": random.choice(CRIME_TYPES),
    }

    complaints.append(complaint)


# Convert complaint data into a DataFrame.
complaints_df = pd.DataFrame(complaints)

# Sort chronologically for easier analysis.
complaints_df = complaints_df.sort_values(
    "complaint_time"
).reset_index(drop=True)


# ---------------------------------------------------------------------------
# GENERATE MULE CHAINS / TRANSACTION HOPS
# ---------------------------------------------------------------------------

transaction_hops = []

used_accounts = set()

for _, complaint in complaints_df.iterrows():

    complaint_id = complaint["complaint_id"]
    victim_city = complaint["victim_city"]
    original_amount = complaint["fraud_amount"]

    # Each complaint gets between 2 and 7 account hops.
    num_hops = random.randint(2, 7)

    # Start the money trail shortly after the complaint.
    current_timestamp = complaint["complaint_time"]

    # Track the remaining amount.
    current_amount = float(original_amount)

    for hop_number in range(1, num_hops + 1):

        # Add a random delay between 2 and 90 minutes.
        gap_minutes = random.randint(2, 90)

        current_timestamp += timedelta(
            minutes=gap_minutes
        )

        hop_city = choose_hop_city(victim_city)

        account_id = generate_account_id(used_accounts)

        # The final hop is the account where cash withdrawal
        # is expected to happen.
        is_final_hop = hop_number == num_hops

        # -------------------------------------------------------------------
        # Amount movement
        # -------------------------------------------------------------------
        if is_final_hop:
            # Keep most of the remaining amount at the final account.
            amount_at_hop = current_amount

        else:
            # Simulate fees/splitting by reducing the amount slightly.
            reduction_percentage = random.uniform(0.01, 0.08)

            amount_at_hop = current_amount * (
                1 - reduction_percentage
            )

            # Round to nearest ₹100.
            amount_at_hop = round(amount_at_hop / 100) * 100

            # Ensure it never becomes zero.
            amount_at_hop = max(amount_at_hop, 100)

        # Update amount for next hop.
        current_amount = amount_at_hop

        # -------------------------------------------------------------------
        # Default fields for non-final hops
        # -------------------------------------------------------------------
        withdrawal_lat = np.nan
        withdrawal_lon = np.nan
        was_cash_withdrawn = np.nan

        # -------------------------------------------------------------------
        # Final-hop specific information
        # -------------------------------------------------------------------
        if is_final_hop:

            withdrawal_lat, withdrawal_lon = generate_gps(
                hop_city
            )

            # 70% = cash already withdrawn
            # 30% = cash NOT withdrawn yet
            was_cash_withdrawn = random.random() < 0.70

        # -------------------------------------------------------------------
        # Add transaction record
        # -------------------------------------------------------------------
        transaction_hops.append({
            "complaint_id": complaint_id,
            "hop_number": hop_number,
            "account_id": account_id,
            "bank_ifsc": generate_fake_ifsc(),
            "amount_at_this_hop": int(amount_at_hop),
            "hop_timestamp": current_timestamp,
            "hop_city": hop_city,
            "is_final_hop": is_final_hop,
            "withdrawal_lat": withdrawal_lat,
            "withdrawal_lon": withdrawal_lon,
            "was_cash_withdrawn": was_cash_withdrawn,
        })


# ---------------------------------------------------------------------------
# CREATE TRANSACTION DATAFRAME
# ---------------------------------------------------------------------------

transaction_hops_df = pd.DataFrame(transaction_hops)

# Sort by complaint and hop sequence.
transaction_hops_df = transaction_hops_df.sort_values(
    ["complaint_id", "hop_number"]
).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLEAN UP DATA TYPES
# ---------------------------------------------------------------------------

complaints_df["complaint_time"] = pd.to_datetime(
    complaints_df["complaint_time"]
)

transaction_hops_df["hop_timestamp"] = pd.to_datetime(
    transaction_hops_df["hop_timestamp"]
)

# Make final-hop withdrawal status a proper nullable Boolean.
transaction_hops_df["was_cash_withdrawn"] = (
    transaction_hops_df["was_cash_withdrawn"].astype("boolean")
)


# ---------------------------------------------------------------------------
# SAVE CSV FILES
# ---------------------------------------------------------------------------

complaints_df.to_csv(
    "complaints.csv",
    index=False
)

transaction_hops_df.to_csv(
    "transaction_hops.csv",
    index=False
)


# ---------------------------------------------------------------------------
# PRINT SUMMARY
# ---------------------------------------------------------------------------

print("=" * 60)
print("Synthetic Cybercrime Dataset Generated")
print("=" * 60)

print(f"Complaints generated : {len(complaints_df):,}")
print(f"Transaction hops     : {len(transaction_hops_df):,}")

print(
    f"Average hops/complaint: "
    f"{len(transaction_hops_df) / len(complaints_df):.2f}"
)

print(
    "\nCash withdrawal status "
    "(final hops only):"
)

final_hops = transaction_hops_df[
    transaction_hops_df["is_final_hop"]
]

withdrawal_counts = final_hops["was_cash_withdrawn"].value_counts(
    dropna=False
)

print(withdrawal_counts)

print("\nFraud amount statistics:")
print(
    complaints_df["fraud_amount"].describe()
)

print("\nFiles created:")
print("  - complaints.csv")
print("  - transaction_hops.csv")

print("=" * 60)