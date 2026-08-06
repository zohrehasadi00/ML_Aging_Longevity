#This script is only used for Part C of the audit report.
#Compare the original-style and improved pipelines on the same test patients.


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
DATA_DIR = Path("mimic_demo")
OUTPUT_DIR = Path("comparison_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_common_cohort():
    #Load the data and keep one first ICU stay per hospital admission.
    patients = pd.read_csv(DATA_DIR / "PATIENTS.csv")
    admissions = pd.read_csv(DATA_DIR / "ADMISSIONS.csv")
    icustays = pd.read_csv(DATA_DIR / "ICUSTAYS.csv")

    admissions_patients = pd.merge(
        admissions,
        patients,
        on="subject_id",
        how="left",
    )

    # Both pipelines use the same observation unit for this comparison.
    icustays["intime"] = pd.to_datetime(icustays["intime"])
    first_icustays = (
        icustays.sort_values("intime")
        .drop_duplicates(subset="hadm_id", keep="first")
    )

    cohort = pd.merge(
        admissions_patients,
        first_icustays,
        on=["subject_id", "hadm_id"],
        how="inner",
    )

    # Age is useful for the improved pipeline and replaces the raw DOB there
    admission_date = pd.to_datetime(cohort["admittime"])
    birth_date = pd.to_datetime(cohort["dob"])
    cohort["age"] = admission_date.dt.year - birth_date.dt.year

    birthday_not_reached = (
        (admission_date.dt.month < birth_date.dt.month)
        | (
            (admission_date.dt.month == birth_date.dt.month)
            & (admission_date.dt.day < birth_date.dt.day)
        )
    )
    cohort["age"] = cohort["age"] - birthday_not_reached.astype(int)
    cohort["age"] = cohort["age"].clip(upper=90)

    assert cohort.groupby("hadm_id").size().max() == 1
    return cohort.reset_index(drop=True)


def make_common_split(cohort):
    #Create one patient-grouped split that is reused by both pipelines
    target = cohort["hospital_expire_flag"]
    groups = cohort["subject_id"]

    outer_cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    dummy_features = cohort.drop(columns="hospital_expire_flag")
    train_idx, test_idx = next(
        outer_cv.split(dummy_features, target, groups=groups)
    )

    train_patients = set(groups.iloc[train_idx])
    test_patients = set(groups.iloc[test_idx])
    assert train_patients.isdisjoint(test_patients)

    return train_idx, test_idx


def evaluate_model(pipeline_name, model_name, model, X_test, y_test):
    #Calculate the metrics used in the assignment
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    result = {
        "pipeline": pipeline_name,
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    return result, fpr, tpr


def run_original_style(cohort, train_idx, test_idx):
    #Run a compact version of the main methodological choices in the original code
    data = cohort.copy()

    # The original file treated timestamps as categorical text values.
    if "intime" in data.columns:
        data["intime"] = data["intime"].astype("string")

    # The original workflow learned preprocessing from the complete dataset
    # before the outer split. We reproduce that weakness for the comparison.
    missing_rate = data.isna().mean()
    data = data.drop(columns=missing_rate[missing_rate > 0.50].index)

    for column in data.select_dtypes(include=["object", "string"]).columns:
        data[column] = data[column].fillna("Unknown")

    for column in data.select_dtypes(include="number").columns:
        if data[column].isna().any():
            data[column] = data[column].fillna(data[column].median())

    if "los" in data.columns:
        los_cap = data["los"].quantile(0.99)
        data["los"] = data["los"].clip(upper=los_cap)

    categorical_columns = data.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()
    data = pd.get_dummies(
        data,
        columns=categorical_columns,
        drop_first=True,
    )

    X_all = data.drop(columns="hospital_expire_flag")
    y_all = data["hospital_expire_flag"]

    # Scale numeric columns before splitting, as in the original workflow.
    id_columns = {
        "subject_id",
        "hadm_id",
        "icustay_id",
        "row_id_x",
        "row_id_y",
        "row_id",
    }
    scale_columns = [
        column
        for column in X_all.columns
        if column not in id_columns
        and pd.api.types.is_numeric_dtype(X_all[column])
    ]

    scaler = StandardScaler()
    X_all[scale_columns] = scaler.fit_transform(
        X_all[scale_columns].astype(float)
    )

    # sklearn models expect one consistent numeric dtype.
    X_all = X_all.astype(float)

    zero_variance = X_all.nunique()
    X_all = X_all.drop(columns=zero_variance[zero_variance <= 1].index)

    X_train = X_all.iloc[train_idx]
    X_test = X_all.iloc[test_idx]
    y_train = y_all.iloc[train_idx]
    y_test = y_all.iloc[test_idx]

    models = {
        "Logistic Regression": GridSearchCV(
            LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
            {"C": [0.01, 0.1, 1, 10, 100]},
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
        ),
        "Random Forest": GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10],
            },
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
        ),
        "XGBoost": GridSearchCV(
            xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
            ),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
            },
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
        ),
    }

    results = []
    curves = {}
    for name, search in models.items():
        search.fit(X_train, y_train)
        result, fpr, tpr = evaluate_model(
            "Original-style",
            name,
            search.best_estimator_,
            X_test,
            y_test,
        )
        result["best_cv_roc_auc"] = search.best_score_
        result["best_parameters"] = str(search.best_params_)
        results.append(result)
        curves[name] = (fpr, tpr, result["roc_auc"])

    return results, curves


def run_improved(cohort, train_idx, test_idx):
    #Run the final leakage-aware preprocessing and grouped CV workflow
    data = cohort.copy()

    columns_to_remove = [
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
    data = data.drop(columns=columns_to_remove, errors="ignore")

    train_data = data.iloc[train_idx].copy().reset_index(drop=True)
    test_data = data.iloc[test_idx].copy().reset_index(drop=True)

    groups_train = train_data["subject_id"].copy()
    groups_test = test_data["subject_id"].copy()
    assert set(groups_train).isdisjoint(set(groups_test))

    train_data = train_data.drop(columns="subject_id")
    test_data = test_data.drop(columns="subject_id")

    X_train = train_data.drop(columns="hospital_expire_flag")
    y_train = train_data["hospital_expire_flag"]
    X_test = test_data.drop(columns="hospital_expire_flag")
    y_test = test_data["hospital_expire_flag"]

    X_train = X_train.drop(columns="intime", errors="ignore")
    X_test = X_test.drop(columns="intime", errors="ignore")

    if "first_wardid" in X_train.columns:
        X_train["first_wardid"] = X_train["first_wardid"].astype("string")
        X_test["first_wardid"] = X_test["first_wardid"].astype("string")

    numeric_columns = X_train.select_dtypes(include="number").columns
    categorical_columns = X_train.select_dtypes(
        include=["object", "string"]
    ).columns

    numeric_steps = Pipeline(
        [
            ("fill_missing", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_steps = Pipeline(
        [
            (
                "fill_missing",
                SimpleImputer(strategy="constant", fill_value="Unknown"),
            ),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=15,
                ),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_steps, numeric_columns),
            ("categorical", categorical_steps, categorical_columns),
        ]
    )

    inner_cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    searches = {
        "Logistic Regression": GridSearchCV(
            Pipeline(
                [
                    ("preprocessing", preprocessing),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=5000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {"model__C": [0.01, 0.1, 1, 10, 100]},
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1,
        ),
        "Random Forest": GridSearchCV(
            Pipeline(
                [
                    ("preprocessing", preprocessing),
                    (
                        "model",
                        RandomForestClassifier(random_state=RANDOM_STATE),
                    ),
                ]
            ),
            {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [None, 5, 10],
            },
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1,
        ),
        "XGBoost": GridSearchCV(
            Pipeline(
                [
                    ("preprocessing", preprocessing),
                    (
                        "model",
                        xgb.XGBClassifier(
                            random_state=RANDOM_STATE,
                            eval_metric="logloss",
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [3, 5, 7],
            },
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1,
        ),
    }

    results = []
    curves = {}
    for name, search in searches.items():
        search.fit(X_train, y_train, groups=groups_train)
        result, fpr, tpr = evaluate_model(
            "Improved",
            name,
            search.best_estimator_,
            X_test,
            y_test,
        )
        result["best_cv_roc_auc"] = search.best_score_
        result["best_parameters"] = str(search.best_params_)
        results.append(result)
        curves[name] = (fpr, tpr, result["roc_auc"])

    return results, curves


def save_roc_plot(curves, pipeline_name, file_name):
   #Save one ROC figure for a pipeline
    plt.figure(figsize=(8, 6))
    for model_name, (fpr, tpr, auc_value) in curves.items():
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_value:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves: {pipeline_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / file_name, dpi=200)
    plt.close()


def main():
    cohort = load_common_cohort()
    train_idx, test_idx = make_common_split(cohort)

    print("Common cohort size:", len(cohort))
    print("Training observations:", len(train_idx))
    print("Test observations:", len(test_idx))
    print(
        "Test deaths:",
        int(cohort.iloc[test_idx]["hospital_expire_flag"].sum()),
    )

    original_results, original_curves = run_original_style(
        cohort,
        train_idx,
        test_idx,
    )
    improved_results, improved_curves = run_improved(
        cohort,
        train_idx,
        test_idx,
    )

    results = pd.DataFrame(original_results + improved_results)
    results = results[
        [
            "pipeline",
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "best_cv_roc_auc",
            "best_parameters",
        ]
    ]
    results.to_csv(
        OUTPUT_DIR / "pipeline_comparison_results.csv",
        index=False,
    )

    save_roc_plot(
        original_curves,
        "Original-style pipeline",
        "roc_original_style.png",
    )
    save_roc_plot(
        improved_curves,
        "Improved pipeline",
        "roc_improved.png",
    )

    print("\nComparison results:")
    print(results.to_string(index=False))
    print(
        "\nSaved results to:",
        OUTPUT_DIR / "pipeline_comparison_results.csv",
    )


if __name__ == "__main__":
    main()