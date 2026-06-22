#   Code for the first three lectures:
#   Machine Learning for Aging: Exploratory Data Analysis
#   Machine Learning for Aging: Outliers
#   Machine Learning for Aging: Data Preprocessing

import os
import sys
import math
import logging

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import missingno as msno
    HAS_MISSINGNO = True
except ImportError:
    HAS_MISSINGNO = False

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ons_pipeline")

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# pandas 4.x: try 'str' first, fall back to 'object'
def get_text_columns(dataframe):
    try:
        return dataframe.select_dtypes(include='str').columns.tolist()
    except TypeError:
        return dataframe.select_dtypes(include='object').columns.tolist()


# IQR outlier detection
# Machine Learning for Aging: Outliers, p15
def is_outlier_iqr(series, threshold=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return (series < lower_bound) | (series > upper_bound)


# Scatter plot at multiple IQR thresholds to pick the best one
# Machine Learning for Aging: Outliers, p16-17
def visualize_iqr_thresholds(df, columns, thresholds, plots_dir):
    plt.figure(figsize=(15, len(columns) * 5))
    for i, column in enumerate(columns, 1):
        for j, threshold in enumerate(thresholds, 1):
            plt.subplot(len(columns), len(thresholds), (i - 1) * len(thresholds) + j)
            outliers = is_outlier_iqr(df[column], threshold=threshold)
            plt.scatter(df.index[~outliers], df[column][~outliers], s=10, label="Good", color="#4CAF50")
            plt.scatter(df.index[outliers],  df[column][outliers],  s=10, label="Outliers", color="#F44336")
            plt.title(f"{column} (Threshold={threshold})")
            plt.xlabel("Index")
            plt.ylabel(column)
            if i == 1:
                plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/outlier_thresholds.png", dpi=100)
    plt.close()


def run_pipeline():

    # Exploratory Data Analysis
    # Machine Learning for Aging: Exploratory Data Analysis, p5-12
    log.info("EDA -- loading and exploring the data")

    patients   = pd.read_csv('mimic_demo/PATIENTS.csv')
    admissions = pd.read_csv('mimic_demo/ADMISSIONS.csv')
    icustays   = pd.read_csv('mimic_demo/ICUSTAYS.csv')

    log.info("Table sizes -- patients: %s | admissions: %s | icustays: %s",
             patients.shape, admissions.shape, icustays.shape)

    # Left join: keep every row from the left table even without a match on the right
    df = pd.merge(admissions, patients, on='subject_id', how='left')
    df = pd.merge(df, icustays, on=['subject_id', 'hadm_id'], how='left')
    log.info("After merge: %d rows x %d columns", df.shape[0], df.shape[1])

    log.info("\n--- dtypes ---\n%s", df.dtypes.to_string())
    log.info("\n--- describe ---\n%s", df.describe().to_string())

    counts = df['hospital_expire_flag'].value_counts()
    log.info("Survived (0): %d  |  Died (1): %d", counts.get(0, 0), counts.get(1, 0))

    # Check for duplicate rows
    # Machine Learning for Aging: Exploratory Data Analysis, p12
    n_dupes = df.duplicated().sum()
    log.info("Duplicate rows: %d", n_dupes)
    df = df.drop_duplicates().reset_index(drop=True)

    # Missing value percentages per column
    # Machine Learning for Aging: Exploratory Data Analysis, p59-60
    missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    log.info("\n--- Missing %% per column ---\n%s", missing_pct[missing_pct > 0].to_string())

    assert df.duplicated().sum() == 0
    assert 'hospital_expire_flag' in df.columns
    log.info("[PASS] EDA (load & explore) -- %d rows, %d columns", df.shape[0], df.shape[1])


    # Remove data leakage columns, create age
    # Machine Learning for Aging: Exploratory Data Analysis
    log.info("EDA -- removing leakage columns, creating age")

    df['admittime'] = pd.to_datetime(df['admittime'])
    df['dob']       = pd.to_datetime(df['dob'])
    df['age'] = (df['admittime'] - df['dob']).dt.days / 365.25

    # MIMIC privacy: patients >= 89 have dob shifted back so they appear ~300 years old
    df['age'] = df['age'].clip(upper=89)
    log.info("Age -- min: %.1f  max: %.1f  mean: %.1f",
             df['age'].min(), df['age'].max(), df['age'].mean())

    # Columns known only after death or discharge -- including these would let the model cheat
    leakage_cols = [
        'deathtime', 'dod', 'dod_hosp', 'dod_ssn',
        'expire_flag', 'discharge_location', 'dischtime',
        'los', 'last_careunit', 'last_wardid',
    ]
    id_cols   = ['subject_id', 'hadm_id', 'icustay_id', 'row_id', 'row_id_x', 'row_id_y']
    date_cols = ['admittime', 'dob', 'edregtime', 'edouttime', 'intime', 'outtime']

    all_to_drop = [c for c in leakage_cols + id_cols + date_cols if c in df.columns]
    df = df.drop(columns=all_to_drop)
    log.info("Dropped %d columns. Remaining: %s", len(all_to_drop), df.columns.tolist())

    for col in ['deathtime', 'dod', 'expire_flag', 'los']:
        assert col not in df.columns
    assert 'age' in df.columns

    # Sanity check after removing implausible values (age clip)
    # Machine Learning for Aging: Outliers, p9
    log.info("Sanity check after age clip:\n%s", df.describe().to_string())

    log.info("[PASS] EDA (leakage removal) -- %d columns left", df.shape[1])


    # Outliers -- Find
    # Machine Learning for Aging: Outliers, p10-21
    log.info("Outliers -- finding extreme values")

    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    n = len(numerical_columns)
    nrows = math.ceil(n / 3)

    # Histograms on cleaned data
    # Machine Learning for Aging: Outliers, p10-12
    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5 * nrows))
    axes = axes.flatten()
    for i, column in enumerate(numerical_columns):
        axes[i].hist(df[column].dropna(), bins=10, alpha=0.7, color="#4CAF50")
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f"{column} (Cleaned)")
    for i in range(len(numerical_columns), len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/histograms.png", dpi=100)
    plt.close()
    log.info("Histograms saved.")

    # Boxplots on cleaned data
    # Machine Learning for Aging: Outliers, p13-14
    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5 * nrows))
    axes = axes.flatten()
    for i, column in enumerate(numerical_columns):
        sns.boxplot(x=df[column], ax=axes[i], orient='h', color="#4CAF50")
        axes[i].set_title(f"{column} (Cleaned)")
    for i in range(len(numerical_columns), len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/boxplots.png", dpi=100)
    plt.close()
    log.info("Boxplots saved.")

    # Scatter plots at four IQR thresholds to pick the best level
    # Machine Learning for Aging: Outliers, p16-17
    continuous_cols = ['age']
    iqr_thresholds = [0.5, 1.5, 2.5, 3.0]
    visualize_iqr_thresholds(df, continuous_cols, iqr_thresholds, PLOTS_DIR)
    log.info("Outlier threshold scatter plots saved.")

    # Replace outliers with NaN -- threshold=2.5 chosen from the scatter comparison
    # Machine Learning for Aging: Outliers, p19-20
    chosen_threshold = 2.5
    for col in continuous_cols:
        outlier_mask = is_outlier_iqr(df[col], threshold=chosen_threshold)
        df.loc[outlier_mask, col] = np.nan
        log.info("  '%s': %d values replaced (threshold=%.1f)",
                 col, outlier_mask.sum(), chosen_threshold)

    # Sanity check after outlier removal
    # Machine Learning for Aging: Outliers, p21
    log.info("describe() after outlier removal:\n%s", df.describe().to_string())

    log.info("[PASS] Outliers (find) -- extreme values replaced with NaN")


    # Outliers -- Fill (missing value imputation)
    # Machine Learning for Aging: Outliers, p22-25
    log.info("Outliers -- filling missing values")

    # Column-wise deletion for columns > 30% missing
    # Machine Learning for Aging: Outliers, p22
    # (30% is a common rule of thumb; the lecture shows the calculation without specifying a cutoff)
    col_missing_rate = df.isnull().mean()
    high_missing_cols = [
        c for c in col_missing_rate[col_missing_rate > 0.30].index
        if c != 'hospital_expire_flag'
    ]
    if high_missing_cols:
        log.info("  Columns >30%% missing, dropped: %s", high_missing_cols)
        df = df.drop(columns=high_missing_cols)
    else:
        log.info("  No columns exceed 30%% missing.")

    # Row-wise NaN histogram
    # Machine Learning for Aging: Outliers, p23
    nan_per_row = df.isna().sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(nan_per_row, bins=30, alpha=0.7, color='#4CAF50')
    plt.title("Histogram of NaN Values Across Rows")
    plt.xlabel("Number of NaN values per row")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/nan_per_row.png", dpi=100)
    plt.close()
    log.info("Row-wise NaN histogram saved.")

    # Missing pattern visualisation
    # Machine Learning for Aging: Outliers, p24
    if HAS_MISSINGNO:
        fig, ax = plt.subplots(figsize=(14, 5))
        msno.bar(df, ax=ax)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/missing_bar.png", dpi=100)
        plt.close()

        fig, ax = plt.subplots(figsize=(14, 5))
        msno.matrix(df, ax=ax, sparkline=False)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/missing_matrix.png", dpi=100)
        plt.close()
        log.info("MSNO missing pattern plots saved.")

    # Median imputation for numeric columns, 'Unknown' for text
    # Machine Learning for Aging: Outliers, p25
    num_cols = [c for c in df.select_dtypes(include='number').columns
                if c != 'hospital_expire_flag']
    for col in num_cols:
        n_miss = df[col].isnull().sum()
        if n_miss > 0:
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            log.info("  '%s': %d cells filled with median (%.2f)", col, n_miss, fill_val)

    cat_cols = get_text_columns(df)
    for col in cat_cols:
        n_miss = df[col].isnull().sum()
        if n_miss > 0:
            df[col] = df[col].fillna('Unknown')
            log.info("  '%s': %d cells filled with 'Unknown'", col, n_miss)

    assert df.isnull().sum().sum() == 0
    log.info("[PASS] Outliers (fill) -- no missing values remain. Shape: %s", df.shape)


    # Data Preprocessing -- Encoding
    # Machine Learning for Aging: Data Preprocessing, p18-21
    log.info("Data Preprocessing -- encoding text columns")

    cat_cols = get_text_columns(df)
    log.info("  Text columns and unique value counts:")
    for col in cat_cols:
        log.info("    %-26s  %d unique values", col, df[col].nunique())

    # Drop columns with > 15 unique values before encoding
    # Machine Learning for Aging: Data Preprocessing, p21
    high_cardinality = [c for c in cat_cols if df[c].nunique() > 15]
    if high_cardinality:
        log.info("  Dropping high-cardinality columns: %s", high_cardinality)
        df = df.drop(columns=high_cardinality)

    # Dummy encoding -- drop_first=False to keep all categories
    # Machine Learning for Aging: Data Preprocessing, p20
    cat_cols_to_encode = get_text_columns(df)
    log.info("  Encoding: %s", cat_cols_to_encode)
    df = pd.get_dummies(df, columns=cat_cols_to_encode, drop_first=False, dtype=int)

    assert len(get_text_columns(df)) == 0
    log.info("[PASS] Data Preprocessing (encoding) -- shape: %s", df.shape)


    # Data Preprocessing -- Scaling
    # Machine Learning for Aging: Data Preprocessing, p28-32
    log.info("Data Preprocessing -- scaling numeric columns")

    X = df.drop(columns=['hospital_expire_flag'])
    y = df['hospital_expire_flag']

    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns)

    # NOTE: fit the scaler on training data only in a real project -- Machine Learning for Aging: Data Preprocessing, p41
    # Here we fit on all data because no train/test split has been done yet.

    col_means = X_scaled.mean().abs()
    assert (col_means < 1e-6).all()
    log.info("[PASS] Data Preprocessing (scaling) -- mean~0, std~1")


    # Data Preprocessing -- Feature Selection (RFE)
    # Machine Learning for Aging: Data Preprocessing, p51-54
    log.info("Data Preprocessing -- feature selection (RFE)")

    n_keep = 10
    log.info("  %d columns in, keeping best %d", X_scaled.shape[1], n_keep)

    # n_features_to_select=2 in the lecture is a demo example; we use 10 here
    estimator = LogisticRegression(max_iter=2000, random_state=42, solver='liblinear')
    rfe = RFE(estimator=estimator, n_features_to_select=n_keep, step=1)
    rfe.fit(X_scaled, y)

    selected_features = X_scaled.columns[rfe.support_].tolist()
    log.info("  Selected features: %s", selected_features)

    X_final = X_scaled[selected_features]

    assert X_final.shape[1] == n_keep
    log.info("[PASS] Data Preprocessing (feature selection) -- final shape: %s", X_final.shape)


    log.info("ALL DONE -- %d rows x %d columns | plots: %s/", *X_final.shape, PLOTS_DIR)
    return X_final, y


if __name__ == "__main__":
    X_final, y = run_pipeline()
