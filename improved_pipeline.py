import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve


# Create the plots folder if it does not exist
os.makedirs("plots", exist_ok=True)

# Read core tables
patients = pd.read_csv("mimic_demo/PATIENTS.csv")
admissions = pd.read_csv("mimic_demo/ADMISSIONS.csv")
icustays = pd.read_csv("mimic_demo/ICUSTAYS.csv")

# Preview the shape and first rows of each table
print("PATIENTS.csv shape:", patients.shape)
print(patients.head())

print("\nADMISSIONS.csv shape:", admissions.shape)
print(admissions.head())

print("\nICUSTAYS.csv shape:", icustays.shape)
print(icustays.head())

# Merge patients and admissions on subject_id
# subject_id is kept until the split because it is needed for grouped splitting
adm_pat = pd.merge(admissions, patients, on="subject_id", how="left")

# Prediction point:
# Predict in-hospital mortality at the first ICU admission using only
# information available up to that time

# Convert ICU admission time to datetime so the first ICU stay can be identified
icustays["intime"] = pd.to_datetime(icustays["intime"])

# Keep only the first ICU stay for each hospital admission
first_icustays = (
    icustays.sort_values("intime")
    .drop_duplicates(subset="hadm_id", keep="first")
)

# Keep only hospital admissions that have an ICU stay
adm_pat_icu = pd.merge(
    adm_pat,
    first_icustays,
    on=["subject_id", "hadm_id"],
    how="inner",
)

# Calculate age from admission year and birth year
# This avoids datetime overflow caused by shifted MIMIC birth dates
admission_date = pd.to_datetime(adm_pat_icu["admittime"])
birth_date = pd.to_datetime(adm_pat_icu["dob"])

adm_pat_icu["age"] = admission_date.dt.year - birth_date.dt.year

# Subtract one year if the birthday has not happened yet in the admission year
birthday_not_reached = (
    (admission_date.dt.month < birth_date.dt.month)
    | (
        (admission_date.dt.month == birth_date.dt.month)
        & (admission_date.dt.day < birth_date.dt.day)
    )
)
adm_pat_icu["age"] = adm_pat_icu["age"] - birthday_not_reached.astype(int)

# MIMIC-III shifts dates for patients older than 89
adm_pat_icu["age"] = adm_pat_icu["age"].clip(upper=90)

# Confirm that there is only one row per hospital admission
rows_per_admission = adm_pat_icu.groupby("hadm_id").size()

print("\nRows per hospital admission after selecting the first ICU stay:")
print(rows_per_admission.value_counts().sort_index())

assert rows_per_admission.max() == 1, (
    "Some hospital admissions still appear more than once."
)

# Check the shape and sample rows of the merged table
print("\nMerged table shape:", adm_pat_icu.shape)
print(adm_pat_icu.head())

# Check the distribution of the target variable
print("\nHospital Expire Flag distribution:")
print(adm_pat_icu["hospital_expire_flag"].value_counts())

# Calculate missing value rate for each column
missing_rate = adm_pat_icu.isnull().mean().sort_values(ascending=False)
print("\nMissing value rate per column:\n", missing_rate)

# Separate numeric and categorical columns for EDA
eda_numeric_cols = adm_pat_icu.select_dtypes(include=["number"]).columns.tolist()
eda_categorical_cols = adm_pat_icu.select_dtypes(include=["object", "string"]).columns.tolist()

print("\nNumeric columns:", eda_numeric_cols)
print("\nCategorical columns:", eda_categorical_cols)

# Describe numeric variables
print("\nDescriptive statistics for numeric variables:")
print(adm_pat_icu[eda_numeric_cols].describe())

# Frequency count for categorical variables
for col in eda_categorical_cols:
    print(f"\nValue counts for categorical variable '{col}':")
    print(adm_pat_icu[col].value_counts(dropna=False).head(3))

# Plot distribution of the target variable
adm_pat_icu["hospital_expire_flag"].value_counts().plot(kind="bar")
plt.title("Distribution of Hospital Expire Flag")
plt.xlabel("Hospital Expire Flag (0=Survived, 1=Died)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/hospital_expire_flag_dist.png", dpi=200)
plt.close()

# Plot gender distribution
adm_pat_icu["gender"].value_counts().plot(kind="bar")
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/gender_dist.png", dpi=200)
plt.close()

# Plot admission type distribution
adm_pat_icu["admission_type"].value_counts().plot(kind="bar")
plt.title("Admission Type Distribution")
plt.xlabel("Admission Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/admission_type_dist.png", dpi=200)
plt.close()

# Plot insurance distribution
adm_pat_icu["insurance"].value_counts().plot(kind="bar")
plt.title("Insurance Distribution")
plt.xlabel("Insurance")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/insurance_dist.png", dpi=200)
plt.close()

# Plot ethnicity distribution
adm_pat_icu["ethnicity"].value_counts().plot(kind="bar")
plt.title("Ethnicity Distribution")
plt.xlabel("Ethnicity")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("plots/ethnicity_dist.png", dpi=200)
plt.close()

# LOS is used only for EDA and removed before modelling because it is future information
adm_pat_icu["los"].plot(kind="hist", bins=20)
plt.title("Distribution of Length of Stay (LOS)")
plt.xlabel("Length of Stay (days)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plots/los_hist.png", dpi=200)
plt.close()

adm_pat_icu.boxplot(column="los", by="hospital_expire_flag")
plt.title("LOS by Hospital Expire Flag")
plt.xlabel("Hospital Expire Flag (0=Survived, 1=Died)")
plt.ylabel("Length of Stay (days)")
plt.suptitle("")
plt.tight_layout()
plt.savefig("plots/los_boxplot.png", dpi=200)
plt.close()

# Death rate by admission type
death_by_admtype = adm_pat_icu.groupby("admission_type")[
    "hospital_expire_flag"
].mean()
death_by_admtype.plot(kind="bar")
plt.title("Death Rate by Admission Type")
plt.xlabel("Admission Type")
plt.ylabel("Death Rate")
plt.tight_layout()
plt.savefig("plots/death_rate_admtype.png", dpi=200)
plt.close()

# Death rate by insurance
death_by_insurance = adm_pat_icu.groupby("insurance")[
    "hospital_expire_flag"
].mean()
death_by_insurance.plot(kind="bar")
plt.title("Death Rate by Insurance")
plt.xlabel("Insurance")
plt.ylabel("Death Rate")
plt.tight_layout()
plt.savefig("plots/death_rate_insurance.png", dpi=200)
plt.close()

# Check the target distribution to assess class imbalance
target_counts = adm_pat_icu["hospital_expire_flag"].value_counts(dropna=False)
print("\nTarget distribution in percent:")
print((target_counts / len(adm_pat_icu) * 100).round(2))

target_counts.plot(
    kind="bar",
    figsize=(10, 6),
    title="Hospital Expire Flag Distribution",
)
plt.tight_layout()
plt.savefig("plots/hospital_expire_flag_count.png", dpi=200)
plt.close()

# Remove IDs, future information and direct leakage variables
cols_to_remove = [
    "row_id_x",
    "row_id_y",
    "row_id",
    "hadm_id",
    "icustay_id",
    "discharge_location",
    "dod",
    "dod_hosp",
    "dod_ssn",
    "deathtime",
    "expire_flag",
    "dischtime",
    "outtime",
    "last_careunit",
    "last_wardid",
    "has_chartevents_data",
    "admittime",
    "dob",
    "edregtime",
    "edouttime",
    "dbsource",
    "los",
]
adm_pat_icu = adm_pat_icu.drop(columns=cols_to_remove, errors="ignore")
print("\nColumns kept for splitting and modelling:")
print(adm_pat_icu.columns.tolist())

# Outer split: grouped by patient and stratified by the target
X = adm_pat_icu.drop(columns=["hospital_expire_flag"])
y = adm_pat_icu["hospital_expire_flag"]
groups = adm_pat_icu["subject_id"]

outer_split = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
train_idx, test_idx = next(outer_split.split(X, y, groups=groups))

# Create the outer train and test datasets
adm_pat_icu_train = adm_pat_icu.iloc[train_idx].copy().reset_index(drop=True)
adm_pat_icu_test = adm_pat_icu.iloc[test_idx].copy().reset_index(drop=True)

# Save patient IDs for grouped inner cross-validation
groups_train = groups.iloc[train_idx].copy().reset_index(drop=True)
groups_test = groups.iloc[test_idx].copy().reset_index(drop=True)

# Check that the same patient is not in train and test
patient_overlap = set(groups_train).intersection(set(groups_test))
print("\nNumber of patients in both train and test:", len(patient_overlap))
assert len(patient_overlap) == 0, "Patient leakage found between train and test."

# Remove subject_id from the model features
adm_pat_icu_train = adm_pat_icu_train.drop(columns=["subject_id"])
adm_pat_icu_test = adm_pat_icu_test.drop(columns=["subject_id"])

# Separate features and target
X_train = adm_pat_icu_train.drop(columns=["hospital_expire_flag"])
y_train = adm_pat_icu_train["hospital_expire_flag"]

X_test = adm_pat_icu_test.drop(columns=["hospital_expire_flag"])
y_test = adm_pat_icu_test["hospital_expire_flag"]

# intime is a timestamp and should not be used directly by the models
X_train = X_train.drop(columns=["intime"], errors="ignore")
X_test = X_test.drop(columns=["intime"], errors="ignore")

# first_wardid is an ID/category, not a continuous numeric measurement
if "first_wardid" in X_train.columns:
    X_train["first_wardid"] = X_train["first_wardid"].astype("string")
    X_test["first_wardid"] = X_test["first_wardid"].astype("string")

# Find numeric and categorical model columns
numeric_cols = X_train.select_dtypes(include="number").columns
categorical_cols = X_train.select_dtypes(
    include=["object", "string"]
).columns

print("\nNumeric model columns:")
print(numeric_cols.tolist())
print("\nCategorical model columns:")
print(categorical_cols.tolist())

# Fill missing numeric values with the median and scale the data
numeric_steps = Pipeline(
    [
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
)

# Fill missing categorical values and group rare categories
categorical_steps = Pipeline([
    ('fill_missing', SimpleImputer(
        strategy='constant',
        fill_value='Unknown'
    )),
    ('encode', OneHotEncoder(
        handle_unknown='infrequent_if_exist',
        max_categories=15
    ))
])

# Apply the correct preprocessing to each type of column
preprocessing = ColumnTransformer(
    [
        ("numeric", numeric_steps, numeric_cols),
        ("categorical", categorical_steps, categorical_cols),
    ]
)

# Group-aware inner cross-validation
# The same patient cannot be in training and validation in one fold
inner_cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

# Logistic Regression with preprocessing
logreg_pipeline = Pipeline(
    [
        ("preprocessing", preprocessing),
        (
            "model",
            LogisticRegression(max_iter=5000, random_state=42),
        ),
    ]
)

param_grid_lr = {"model__C": [0.01, 0.1, 1, 10, 100]}

grid_lr = GridSearchCV(
    logreg_pipeline,
    param_grid_lr,
    cv=inner_cv,
    scoring="roc_auc",
)
grid_lr.fit(X_train, y_train, groups=groups_train)

print("\nBest parameters (Logistic Regression):", grid_lr.best_params_)
print("Best CV ROC-AUC (Logistic Regression):", grid_lr.best_score_)
logreg_best = grid_lr.best_estimator_

# Random Forest with preprocessing
rf_pipeline = Pipeline(
    [
        ("preprocessing", preprocessing),
        ("model", RandomForestClassifier(random_state=42)),
    ]
)

param_grid_rf = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [None, 5, 10],
}

grid_rf = GridSearchCV(
    rf_pipeline,
    param_grid_rf,
    cv=inner_cv,
    scoring="roc_auc",
)
grid_rf.fit(X_train, y_train, groups=groups_train)

print("\nBest parameters (Random Forest):", grid_rf.best_params_)
print("Best CV ROC-AUC (Random Forest):", grid_rf.best_score_)
rf_best = grid_rf.best_estimator_

# XGBoost with preprocessing
xgb_pipeline = Pipeline(
    [
        ("preprocessing", preprocessing),
        (
            "model",
            xgb.XGBClassifier(
                random_state=42,
                eval_metric="logloss",
            ),
        ),
    ]
)

param_grid_xgb = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [3, 5, 7],
}

grid_xgb = GridSearchCV(
    xgb_pipeline,
    param_grid_xgb,
    cv=inner_cv,
    scoring="roc_auc",
)
grid_xgb.fit(X_train, y_train, groups=groups_train)

print("\nBest parameters (XGBoost):", grid_xgb.best_params_)
print("Best CV ROC-AUC (XGBoost):", grid_xgb.best_score_)
xgb_best = grid_xgb.best_estimator_

# Evaluate the final tuned models once on the untouched outer test set
models = {
    "Logistic Regression": logreg_best,
    "Random Forest": rf_best,
    "XGBoost": xgb_best,
}

results = {}
plt.figure(figsize=(8, 6))

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n{name}:")
    print(results[name])

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix: {name}")

    file_name = name.replace(" ", "_")
    plt.tight_layout()
    plt.savefig(
        f"plots/confusion_matrix_{file_name}.png",
        dpi=200,
    )
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(1)
    plt.plot(
        fpr,
        tpr,
        label=f'{name} (AUC={results[name]["roc_auc"]:.2f})',
    )

plt.figure(1)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (All Models)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/roc_curve.png", dpi=200)
plt.close()

# Save the final test results in one table
results_df = pd.DataFrame.from_dict(results, orient="index")
results_df.index.name = "model"
results_df.to_csv("model_results.csv")

print("\nFinal test results:")
print(results_df)
