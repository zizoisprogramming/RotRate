import argparse
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print(f"[1/7] Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"      Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 2. INITIAL COLUMN DROPS  (relationships / leakage)
# ─────────────────────────────────────────────

def drop_initial_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns identified as leaky, redundant, or derived targets."""
    print("[2/7] Dropping initial columns …")
    # Target-related columns dropped as labels / leaky
    drop_cols = [
        "academic_risk_score",
        "financial_risk_score",
        "wellbeing_index",
        # Highly correlated country-level columns
        "late_night_score",
        "country",
        "poverty_rate_percent",
        "internet_infrastructure_index",
        "brain_rot_level",          # derived from brain_rot_index
        # Identified as high-risk columns with unrealistic zeros
        "attention_span_minutes",
        "productivity_score",
        "impulse_purchase_score",
        "digital_addiction_score",
        "academic_motivation",
        # Content-hour columns dropped for accuracy / correlation reasons
        "news_content_hours",
        "entertainment_content_hours",
        "education_content_hours",
        # Dropped for completeness (>50 % null)
        "field_of_study",
    ]
    existing = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing)
    print(f"      Remaining columns: {df.shape[1]}")
    return df


# ─────────────────────────────────────────────
# 3. DATA CLEANING
# ─────────────────────────────────────────────

def _get_unusual_values_pct(df: pd.DataFrame, col: str, value: float = 0) -> float:
    return len(df[df[col] <= value]) / len(df) * 100


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/7] Cleaning data …")

    # --- Accuracy: drop rows with social_media_hours <= 0 ---
    before = len(df)
    df = df[df["social_media_hours"] > 0].copy()
    print(f"      Dropped {before - len(df)} rows with social_media_hours <= 0")

    # --- Completeness: drop remaining nulls ---
    before = len(df)
    df = df.dropna()
    print(f"      Dropped {before - len(df)} rows with nulls")

    return df


# ─────────────────────────────────────────────
# 4. OUTLIER HANDLING
# ─────────────────────────────────────────────

def _get_outlier_pct(df: pd.DataFrame, col: str) -> float:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
    return mask.sum() / len(df) * 100


def _solve_outliers_drop(df: pd.DataFrame, col: str) -> pd.DataFrame:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    return df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]


def _cap_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    df = df.copy()
    df.loc[df[col] > q3 + 1.5 * iqr, col] = q3 + 1.5 * iqr
    df.loc[df[col] < q1 - 1.5 * iqr, col] = q1 - 1.5 * iqr
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    print("[4/7] Handling outliers …")

    numeric_cols = (
        df.drop(columns=["student_id"])
        .select_dtypes(include=["int64", "float64"])
        .columns
        .tolist()
    )

    outlier_pcts = {col: _get_outlier_pct(df, col) for col in numeric_cols}
    outlier_df = (
        pd.DataFrame(outlier_pcts, index=["pct"])
        .T
        .query("pct > 0")
        .sort_values("pct")
    )

    small_outlier_cols = outlier_df[outlier_df["pct"] <= 1.5].index.tolist()
    high_outlier_cols  = outlier_df[outlier_df["pct"] >  1.5].index.tolist()

    # Drop rows for small-outlier columns
    for col in small_outlier_cols:
        df = _solve_outliers_drop(df, col)

    # Cap rows for high-outlier columns
    for col in high_outlier_cols:
        df = _cap_outliers(df, col)

    # Drop social_media_hours after log-transform (high outlier + redundant)
    if "social_media_hours" in df.columns:
        df = df.drop(columns=["social_media_hours"])

    print(f"      Shape after outlier handling: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[5/7] Engineering features …")

    # --- Log-transform skewed columns ---
    for col in ["average_internet_speed_mbps", "short_video_hours"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: np.log(x + 0.001))

    # --- Encode binary columns ---
    df["gender"] = df["gender"].map({"Female": 0, "Male": 1, "Other": 2})
    df["urban_rural"] = df["urban_rural"].map({"Rural": 0, "Urban": 1})
    df["cyberbullying_exposure"] = df["cyberbullying_exposure"].map({"No": 0, "Yes": 1})
    df["adult_content_exposure"] = df["adult_content_exposure"].map({"No": 0, "Yes": 1})

    # --- Ordinal encode ---
    df["development_level"] = df["development_level"].map(
        {"Underdeveloped": 0, "Developed": 1, "Developing": 2}
    )
    df["family_income_level"] = df["family_income_level"].map(
        {"Low": 1, "Middle": 2, "High": 3}
    )
    df["education_level"] = df["education_level"].map(
        {"Dropout": 0, "School": 1, "Diploma": 2, "Graduate": 3, "Postgraduate": 4, "PhD": 5}
    )
    df["late_night_usage"] = df["late_night_usage"].map(
        {"Never": 0, "Sometimes": 1, "Often": 2, "Always": 3}
    )

    # --- Drop remaining nulls introduced by mapping ---
    df = df.dropna()

    # --- One-hot encode device_access ---
    df = pd.get_dummies(df, columns=["device_access"])

    # --- Constructed features (reduce multicollinearity) ---
    df["ads_clicked_per_view"] = (
        df["ads_clicked_per_week"] / (df["ads_viewed_per_day"] * 7)
    )
    df["stress_anxiety"] = df["stress_level"] + df["anxiety_score"]
    df["likes_per_sessions"] = df["likes_given_per_day"] / df["sessions_per_day"]
    df["comnent_per_sessions"] = df["comments_written_per_day"] / df["sessions_per_day"]
    df["digital_spending_per_income_level"] = (
        df["digital_spending_per_month"] / df["family_income_level"]
    )

    # --- Drop highly-correlated source columns ---
    to_drop = [
        "short_video_hours",
        "ads_viewed_per_day",
        "stress_level",
        "anxiety_score",
        "likes_given_per_day",
        "comments_written_per_day",
        "digital_spending_per_month",
    ]
    df = df.drop(columns=[c for c in to_drop if c in df.columns])

    print(f"      Final feature count: {df.shape[1]}")
    return df


# ─────────────────────────────────────────────
# 6. FEATURE SCALING
# ─────────────────────────────────────────────

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[6/7] Scaling numerical features …")

    # Columns NOT to scale
    skip_cols = {"student_id", "brain_rot_index"}

    numerical_cols = [
        c for c in df.select_dtypes(include=["int64", "float64"]).columns
        if c not in skip_cols
    ]

    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df


# ─────────────────────────────────────────────
# 7. TRAIN & EVALUATE
# ─────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame) -> None:
    print("[7/7] Training baseline Decision Tree …")

    X = df.drop(columns=["student_id", "brain_rot_index"]).values
    y = df["brain_rot_index"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"\n{'='*40}")
    print(f"  Test RMSE : {rmse:.4f}")
    print(f"{'='*40}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Brain Rot Index Prediction Pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="/Users/ziadsamer/Documents/RotRate/data/brainrot.csv",
        help="Path to brainrot.csv",
    )
    args = parser.parse_args()

    df = load_data(args.data)
    df = drop_initial_columns(df)
    df = clean_data(df)
    df = handle_outliers(df)
    df = engineer_features(df)
    df = scale_features(df)
    train_and_evaluate(df)


if __name__ == "__main__":
    main()