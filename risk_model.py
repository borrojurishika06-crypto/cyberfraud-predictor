# risk_model.py
"""
Cybercrime Cash-Withdrawal Risk Model

This script:
    1. Loads complaint-level features.
    2. Uses completed cash withdrawals as labeled training data.
    3. Trains an XGBoost Regressor to predict time-to-cashout.
    4. Evaluates the model using MAE.
    5. Scores currently active cases where cash has NOT been withdrawn.
    6. Uses SHAP to explain every active prediction individually.
    7. Saves the scored active cases to:
           data/active_cases_scored.csv

Required packages:
    pip install pandas numpy scikit-learn xgboost shap
"""

import os
import random

import numpy as np
import pandas as pd
import shap

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42

INPUT_FILE = "data/complaint_features.csv"
OUTPUT_FILE = "data/active_cases_scored.csv"

random.seed(SEED)
np.random.seed(SEED)


# ============================================================================
# 1. LOAD DATA
# ============================================================================

df = pd.read_csv(INPUT_FILE)

# Convert boolean-like columns into actual booleans where necessary.
#
# CSV files sometimes load True/False as strings depending on how they
# were generated.
def convert_to_bool(value):
    """Safely convert common boolean representations to bool."""
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


df["was_cash_withdrawn"] = df["was_cash_withdrawn"].apply(
    convert_to_bool
)

df["city_changed"] = df["city_changed"].apply(
    convert_to_bool
)


# ============================================================================
# 2. CREATE URGENCY SCORE / TARGET
# ============================================================================

"""
For completed cases:

    was_cash_withdrawn == True

we know how long the money trail took before the final cash-out.

Therefore:

    urgency_score = total_elapsed_minutes

For active cases:

    was_cash_withdrawn == False

the actual cash-out time is unknown, so urgency_score is left as NaN.

The model will learn:

    Features -> time until/at cash-out

from the completed cases.
"""

df["urgency_score"] = np.where(
    df["was_cash_withdrawn"],
    df["total_elapsed_minutes"],
    np.nan,
)


# ============================================================================
# 3. DEFINE MODEL FEATURES
# ============================================================================

FEATURE_COLUMNS = [
    "hop_count",
    "avg_minutes_between_hops",
    "min_minutes_between_hops",
    "city_changed",
    "fraud_amount",
    "crime_type",
]

NUMERIC_FEATURES = [
    "hop_count",
    "avg_minutes_between_hops",
    "min_minutes_between_hops",
    "fraud_amount",
]

CATEGORICAL_FEATURES = [
    "crime_type",
]

# city_changed is boolean, but treating it as a numeric 0/1 feature makes
# it easy for XGBoost to use.
df["city_changed"] = df["city_changed"].astype(int)


# ============================================================================
# 4. SPLIT INTO COMPLETED AND ACTIVE CASES
# ============================================================================

completed_cases = df[
    df["was_cash_withdrawn"] == True
].copy()

active_cases = df[
    df["was_cash_withdrawn"] == False
].copy()


print("=" * 70)
print("Cybercrime Cash-Withdrawal Risk Model")
print("=" * 70)

print(f"Total cases      : {len(df):,}")
print(f"Completed cases  : {len(completed_cases):,}")
print(f"Active cases     : {len(active_cases):,}")


# ============================================================================
# SAFETY CHECK
# ============================================================================

if len(completed_cases) < 10:
    raise ValueError(
        "Not enough completed cash-withdrawal cases to train the model."
    )

if len(active_cases) == 0:
    raise ValueError(
        "No active cases found where was_cash_withdrawn == False."
    )


# ============================================================================
# 5. CREATE TRAINING DATA
# ============================================================================

X = completed_cases[FEATURE_COLUMNS].copy()
y = completed_cases["total_elapsed_minutes"].copy()


# ============================================================================
# 6. TRAIN / TEST SPLIT
# ============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
)


# ============================================================================
# 7. PREPROCESSING
# ============================================================================

"""
crime_type is categorical, so one-hot encoding converts:

    UPI Fraud
    OTP Fraud
    Investment Fraud
    ...

into machine-learning-friendly columns.

The OneHotEncoder uses handle_unknown="ignore" so the model will not fail
if a category appears in the active dataset that wasn't present in the
training split.
"""

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            CATEGORICAL_FEATURES,
        ),
    ],
    remainder="passthrough",
)


# ============================================================================
# 8. CREATE XGBOOST REGRESSOR
# ============================================================================

model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    random_state=SEED,
    n_jobs=-1,
)


# ============================================================================
# 9. BUILD PIPELINE
# ============================================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)


# ============================================================================
# 10. TRAIN MODEL
# ============================================================================

print("\nTraining XGBoost model...")

pipeline.fit(
    X_train,
    y_train,
)


# ============================================================================
# 11. EVALUATE MODEL
# ============================================================================

test_predictions = pipeline.predict(X_test)

mae = mean_absolute_error(
    y_test,
    test_predictions,
)

print("\nModel evaluation")
print("-" * 70)
print(f"Mean Absolute Error (MAE): {mae:.2f} minutes")


# ============================================================================
# 12. PREPARE ACTIVE CASES
# ============================================================================

X_active = active_cases[FEATURE_COLUMNS].copy()

# Predict the estimated time from the beginning of the money trail
# until cash-out.
predicted_minutes = pipeline.predict(X_active)

# Prevent physically impossible negative predictions.
predicted_minutes = np.maximum(
    predicted_minutes,
    0
)

active_cases["predicted_minutes_to_cashout"] = (
    predicted_minutes
)


# ============================================================================
# 13. CALCULATE MINUTES REMAINING
# ============================================================================

"""
minutes_remaining:

    predicted_minutes_to_cashout
        -
    minutes_since_last_hop

Example:

    predicted time-to-cashout = 45 minutes
    already elapsed since last hop = 20 minutes

    minutes_remaining = 25 minutes

If the result is <= 0, the model thinks cash-out may already be due.

For the demo we clamp this to a minimum of 5 minutes, as requested.
"""

active_cases["minutes_remaining"] = (
    active_cases["predicted_minutes_to_cashout"]
    - active_cases["minutes_since_last_hop"]
)

active_cases["minutes_remaining"] = np.maximum(
    active_cases["minutes_remaining"],
    5
)

active_cases["minutes_remaining"] = (
    active_cases["minutes_remaining"].round(2)
)


# ============================================================================
# 14. CREATE URGENCY RANK
# ============================================================================

"""
Smallest minutes_remaining = highest urgency.

Therefore:

    1 = most urgent
"""

active_cases = active_cases.sort_values(
    "minutes_remaining",
    ascending=True
).reset_index(drop=True)

active_cases["urgency_rank"] = (
    np.arange(len(active_cases)) + 1
)


# ============================================================================
# 15. SHAP EXPLANATIONS
# ============================================================================

"""
SHAP explains individual predictions by showing how each feature moved
the prediction away from the model's baseline prediction.

For each active case, we select the two features with the largest
absolute SHAP values.

Important:

    Negative SHAP value
        -> pushes predicted cash-out time DOWN
        -> therefore increases urgency

    Positive SHAP value
        -> pushes predicted cash-out time UP
        -> therefore decreases immediate urgency
"""

# Get the fitted preprocessing transformer.
fitted_preprocessor = pipeline.named_steps["preprocessor"]

# Transform active cases into the feature representation that XGBoost sees.
X_active_transformed = fitted_preprocessor.transform(X_active)

# Get transformed feature names.
transformed_feature_names = (
    fitted_preprocessor.get_feature_names_out()
)

# Get the fitted XGBoost model.
xgb_model = pipeline.named_steps["model"]


# ---------------------------------------------------------------------------
# Create SHAP TreeExplainer
# ---------------------------------------------------------------------------

explainer = shap.TreeExplainer(xgb_model)

shap_values = explainer.shap_values(
    X_active_transformed
)


# ============================================================================
# 16. MAP SHAP FEATURES BACK TO HUMAN-READABLE FEATURES
# ============================================================================

def human_feature_name(feature_name):
    """
    Convert sklearn's transformed feature names into readable names.
    """

    if feature_name.endswith("hop_count"):
        return "hop count"

    if feature_name.endswith("avg_minutes_between_hops"):
        return "average hop timing"

    if feature_name.endswith("min_minutes_between_hops"):
        return "fastest hop timing"

    if feature_name.endswith("fraud_amount"):
        return "fraud amount"

    if feature_name.endswith("city_changed"):
        return "city change"

    if "crime_type_" in feature_name:
        crime = feature_name.split("crime_type_", 1)[1]
        return f"crime type ({crime})"

    return feature_name


# ============================================================================
# 17. GENERATE PLAIN-ENGLISH REASON FOR EACH CASE
# ============================================================================

def describe_feature(feature_name, original_row, shap_value):
    """
    Generate a human-readable explanation for a feature's SHAP contribution.

    Negative SHAP means the feature pushed the predicted time-to-cashout
    lower, which means higher urgency.
    """

    feature = human_feature_name(feature_name)

    if feature == "hop count":
        value = int(original_row["hop_count"])

        if shap_value < 0:
            return f"many hops completed ({value} hops)"
        else:
            return f"hop count ({value} hops)"

    if feature == "average hop timing":
        value = original_row["avg_minutes_between_hops"]

        if shap_value < 0:
            return (
                f"fast hop timing "
                f"(avg {value:.1f} min between hops)"
            )
        else:
            return (
                f"slower hop timing "
                f"(avg {value:.1f} min between hops)"
            )

    if feature == "fastest hop timing":
        value = original_row["min_minutes_between_hops"]

        if shap_value < 0:
            return (
                f"very fast transfer "
                f"(minimum gap {value:.1f} min)"
            )
        else:
            return (
                f"longer minimum transfer gap "
                f"({value:.1f} min)"
            )

    if feature == "fraud amount":
        value = original_row["fraud_amount"]

        if shap_value < 0:
            return f"fraud amount (₹{value:,.0f})"
        else:
            return f"higher fraud amount (₹{value:,.0f})"

    if feature == "city change":
        changed = bool(original_row["city_changed"])

        if changed:
            if shap_value < 0:
                return "money moved to a different city"
            else:
                return "city change pattern"
        else:
            return "money stayed in the same city"

    if feature.startswith("crime type"):
        crime_type = original_row["crime_type"]

        return f"crime pattern ({crime_type})"

    return feature


# ---------------------------------------------------------------------------
# Generate explanations
# ---------------------------------------------------------------------------

reason_texts = []

for row_index in range(len(active_cases)):

    # SHAP values corresponding to this particular prediction.
    case_shap_values = shap_values[row_index]

    # Find indices of the two largest contributions by magnitude.
    top_indices = np.argsort(
        np.abs(case_shap_values)
    )[::-1][:2]

    reasons = []

    for feature_index in top_indices:

        feature_name = transformed_feature_names[
            feature_index
        ]

        shap_value = case_shap_values[
            feature_index
        ]

        reason = describe_feature(
            feature_name,
            active_cases.iloc[row_index],
            shap_value,
        )

        reasons.append(
            (
                reason,
                shap_value,
            )
        )

    # Determine the overall wording based on the direction of the
    # strongest contribution.
    strongest_shap = reasons[0][1]

    if strongest_shap < 0:
        prefix = "High urgency driven by"
    else:
        prefix = "Urgency influenced by"

    reason_string = (
        f"{prefix}: "
        + ", ".join(
            reason[0]
            for reason in reasons
        )
    )

    reason_texts.append(reason_string)


active_cases["top_reason_text"] = reason_texts


# ============================================================================
# 18. SAVE FINAL ACTIVE-CASE DATASET
# ============================================================================

# Create output directory if it doesn't exist.
os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

# Save all original columns plus the new scoring/explanation columns.
active_cases.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================================
# 19. PRINT FINAL SANITY CHECK
# ============================================================================

print("\nActive cases scored successfully.")
print(f"Output file: {OUTPUT_FILE}")

print("\nTop 5 most urgent active cases")
print("=" * 70)

preview_columns = [
    "complaint_id",
    "hop_count",
    "fraud_amount",
    "crime_type",
    "minutes_since_last_hop",
    "predicted_minutes_to_cashout",
    "minutes_remaining",
    "urgency_rank",
    "top_reason_text",
]

print(
    active_cases[
        preview_columns
    ].head(5).to_string(index=False)
)

print("=" * 70)

print(
    f"\nCompleted training cases : {len(completed_cases):,}"
)

print(
    f"Active cases scored      : {len(active_cases):,}"
)

print(
    f"Test-set MAE             : {mae:.2f} minutes"
)

print(
    f"Simulated output         : {OUTPUT_FILE}"
)