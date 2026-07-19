import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve

# Read core tables
patients = pd.read_csv('mimic_demo/PATIENTS.csv')
admissions = pd.read_csv('mimic_demo/ADMISSIONS.csv')
icustays = pd.read_csv('mimic_demo/ICUSTAYS.csv')

# Preview the shape and first rows of each table
print("PATIENTS.csv shape:", patients.shape)
print(patients.head())

print("\nADMISSIONS.csv shape:", admissions.shape)
print(admissions.head())

print("\nICUSTAYS.csv shape:", icustays.shape)
print(icustays.head())

# Merge patients and admissions on 'subject_id', this id will remain untill the very end, cause it will be needed for setting up the StratifiedGroupKFold split
adm_pat = pd.merge(admissions, patients, on='subject_id', how='left')

# Merge the result with icustays on ['subject_id', 'hadm_id']
adm_pat_icu = pd.merge(adm_pat, icustays, on=['subject_id', 'hadm_id'], how='left')

# adding the age
adm_pat_icu['age'] = (
    pd.to_datetime(adm_pat_icu['admittime']) -
    pd.to_datetime(adm_pat_icu['dob'])
).dt.days / 365.25

# Cap age at 90 (MIMIC-III shifts DOB for patients >89 to preserve anonymity)
adm_pat_icu['age'] = adm_pat_icu['age'].clip(upper=90)

# Check how many ICU stays each hospital admission has
duplicates = adm_pat_icu.groupby("hadm_id").size()

print("\nNumber of ICU stays per hospital admission:")
print(duplicates.value_counts().sort_index())

# Check the shape and sample rows of the merged table
print("Merged table shape:", adm_pat_icu.shape)
print(adm_pat_icu.head())

# Check the distribution of the target variable
print("\nHospital Expire Flag distribution:")
print(adm_pat_icu['hospital_expire_flag'].value_counts())

# Calculate missing value rate for each column
missing_rate = adm_pat_icu.isnull().mean().sort_values(ascending=False)
print("Missing value rate per column:\n", missing_rate)

# Separate numeric and categorical columns
numeric_cols = adm_pat_icu.select_dtypes(include=['number']).columns.tolist()
categorical_cols = adm_pat_icu.select_dtypes(include=['object']).columns.tolist()

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# Describe numeric variables
print("Descriptive statistics for numeric variables:")
print(adm_pat_icu[numeric_cols].describe())

# Frequency count for categorical variables (top 3 for each)
for col in categorical_cols:
    print(f"\nValue counts for categorical variable '{col}':")
    print(adm_pat_icu[col].value_counts(dropna=False).head(3))

# Plot distribution of the target variable
adm_pat_icu['hospital_expire_flag'].value_counts().plot(kind='bar')
plt.title('Distribution of Hospital Expire Flag')
plt.xlabel('Hospital Expire Flag (0=Survived, 1=Died)')
plt.ylabel('Count')
plt.savefig('plots/hospital_expire_flag_dist.png', dpi=200)
plt.close()

# Plot gender distribution
adm_pat_icu['gender'].value_counts().plot(kind='bar')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.savefig('plots/gender_dist.png', dpi=200)
plt.close()

# Plot admission_type distribution
adm_pat_icu['admission_type'].value_counts().plot(kind='bar')
plt.title('Admission Type Distribution')
plt.xlabel('Admission Type')
plt.ylabel('Count')
plt.savefig('plots/admission_type_dist.png', dpi=200)
plt.close()

# Plot insurance distribution
adm_pat_icu['insurance'].value_counts().plot(kind='bar')
plt.title('Insurance Distribution')
plt.xlabel('Insurance')
plt.ylabel('Count')
plt.savefig('plots/insurance_dist.png', dpi=200)
plt.close()

# Plot ethnicity distribution
adm_pat_icu['ethnicity'].value_counts().plot(kind='bar')
plt.title('Ethnicity Distribution')
plt.xlabel('Ethnicity')
plt.ylabel('Count')
plt.savefig('plots/ethnicity_dist.png', dpi=200)
plt.close()

# Histogram for length of stay (los)
adm_pat_icu['los'].plot(kind='hist', bins=20)
plt.title('Distribution of Length of Stay (LOS)')
plt.xlabel('Length of Stay (days)')
plt.ylabel('Frequency')
plt.savefig('plots/los_hist.png', dpi=200)
plt.close()

# Boxplot for LOS by hospital_expire_flag
adm_pat_icu.boxplot(column='los', by='hospital_expire_flag')
plt.title('LOS by Hospital Expire Flag')
plt.xlabel('Hospital Expire Flag (0=Survived, 1=Died)')
plt.ylabel('Length of Stay (days)')
plt.suptitle('')
plt.savefig('plots/los_boxplot.png', dpi=200)
plt.close()

# Death rate by admission type
death_by_admtype = adm_pat_icu.groupby('admission_type')['hospital_expire_flag'].mean()
death_by_admtype.plot(kind='bar')
plt.title('Death Rate by Admission Type')
plt.xlabel('Admission Type')
plt.ylabel('Death Rate')
plt.savefig('plots/death_rate_admtype.png', dpi=200)
plt.close()

# Death rate by insurance
death_by_insurance = adm_pat_icu.groupby('insurance')['hospital_expire_flag'].mean()
death_by_insurance.plot(kind='bar')
plt.title('Death Rate by Insurance')
plt.xlabel('Insurance')
plt.ylabel('Death Rate')
plt.savefig('plots/death_rate_insurance.png', dpi=200)
plt.close()

# data ditribution check to know if there is data imbalanced.
col = 'hospital_expire_flag'
vc = adm_pat_icu[col].value_counts(dropna=False)
print((vc / len(adm_pat_icu) * 100).round(2))
vc.plot(kind='bar', figsize=(10,6), title=f'{col} Distribution')
plt.savefig(f'plots/{col}_count.png', dpi=200); plt.close()

# drop leaky data, and IDs
cols_to_remove = [
    'row_id_x', 'row_id_y', 'row_id', 'hadm_id', 'icustay_id',
    'discharge_location', 'dod', 'dod_hosp', 'dod_ssn', 'deathtime', 'expire_flag',
    'dischtime', 'outtime', 'last_careunit', 'last_wardid',
    'has_chartevents_data', 'admittime', 'dob', 'edregtime', 'edouttime', 'dbsource', 'los'
]
adm_pat_icu = adm_pat_icu.drop(columns=cols_to_remove, errors='ignore')
print(f"List of columns remained in datafram:{adm_pat_icu.columns.tolist()}")

# Split: grouped and stratified.
X = adm_pat_icu.drop(columns=['hospital_expire_flag'])
y = adm_pat_icu['hospital_expire_flag']
groups = adm_pat_icu['subject_id']  # <-- use patient ID

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(X, y, groups=groups))  # <-- one fold as test

# create split datasets
adm_pat_icu_train = adm_pat_icu.iloc[train_idx].copy()
adm_pat_icu_test = adm_pat_icu.iloc[test_idx].copy()

# Drop subject_id here so it doesn't clutter preprocessing
adm_pat_icu_train = adm_pat_icu_train.drop(columns=['subject_id'])
adm_pat_icu_test = adm_pat_icu_test.drop(columns=['subject_id'], errors='ignore')

# Preprocessing
# Calculate missing value rates
missing_rate = adm_pat_icu_train.isnull().mean().sort_values(ascending=False)

# Drop columns with more than 25% missing data
threshold = 0.25
cols_to_drop = missing_rate[missing_rate > threshold].index.tolist()

adm_pat_icu_train = adm_pat_icu_train.drop(columns=cols_to_drop)
adm_pat_icu_test = adm_pat_icu_test.drop(columns=cols_to_drop, errors='ignore')

# Compare outlier thresholds 0.5, 1.5, 2.5, 3.0 to justify picking 2.5 (Outliers lecture)
thresholds_to_compare = [0.5, 1.5, 2.5, 3.0]
outlier_counts = []

q1_age = adm_pat_icu_train['age'].quantile(0.25)
q3_age = adm_pat_icu_train['age'].quantile(0.75)
iqr_age = q3_age - q1_age

for t in thresholds_to_compare:
    lower = q1_age - t * iqr_age
    upper = q3_age + t * iqr_age
    n_outliers = ((adm_pat_icu_train['age'] < lower) | (adm_pat_icu_train['age'] > upper)).sum()
    outlier_counts.append(n_outliers)
    print(f"threshold {t}: {n_outliers} outliers flagged")

plt.bar([str(t) for t in thresholds_to_compare], outlier_counts)
plt.title('Outlier Count by IQR Threshold (age)')
plt.xlabel('IQR Threshold')
plt.ylabel('Number of Outliers Flagged')
plt.savefig('plots/outlier_thresholds.png', dpi=200, bbox_inches='tight')
plt.close()


# Outlier detection with IQR method, threshold = 2.5 (from lecture Outliers)
# only checking age here
outlier_thresh = 2.5

for col in ['age']:
    q1 = adm_pat_icu_train[col].quantile(0.25)
    q3 = adm_pat_icu_train[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - outlier_thresh * iqr
    upper_bound = q3 + outlier_thresh * iqr

    train_is_outlier = (adm_pat_icu_train[col] < lower_bound) | (adm_pat_icu_train[col] > upper_bound)
    test_is_outlier = (adm_pat_icu_test[col] < lower_bound) | (adm_pat_icu_test[col] > upper_bound)
    print(f"{col}: bounds [{lower_bound:.1f}, {upper_bound:.1f}], "
          f"{train_is_outlier.sum()} outliers in train, {test_is_outlier.sum()} in test")

    # set outliers to NaN
    adm_pat_icu_train.loc[train_is_outlier, col] = np.nan
    adm_pat_icu_test.loc[test_is_outlier, col] = np.nan

# MSNO plots to check missingness after outliers are removed (from Outliers lecture)
msno.matrix(adm_pat_icu_train)
plt.savefig('plots/missing_matrix.png', dpi=200, bbox_inches='tight')
plt.close()

msno.bar(adm_pat_icu_train)
plt.savefig('plots/missing_bar.png', dpi=200, bbox_inches='tight')
plt.close()

# Row-wise NaN histogram, bins=30 
nan_per_row = adm_pat_icu_train.isnull().sum(axis=1)
nan_per_row.plot(kind='hist', bins=30)
plt.title('Missing Values per Row')
plt.xlabel('Number of Missing Values')
plt.ylabel('Number of Rows')
plt.savefig('plots/nan_per_row.png', dpi=200, bbox_inches='tight')
plt.close()

# Fill missing values in categorical variables with 'Unknown'
for col in adm_pat_icu_train.select_dtypes(include=['object', 'string']).columns:
    adm_pat_icu_train[col] = adm_pat_icu_train[col].fillna('Unknown')
    adm_pat_icu_test[col] = adm_pat_icu_test[col].fillna('Unknown')

# Fill missing values in numeric variables with TRAIN median
for col in adm_pat_icu_train.select_dtypes(include='number').columns:
    if adm_pat_icu_train[col].isnull().sum() > 0:
        median_val = adm_pat_icu_train[col].median()
        adm_pat_icu_train[col] = adm_pat_icu_train[col].fillna(median_val)
        adm_pat_icu_test[col] = adm_pat_icu_test[col].fillna(median_val)

# Histogram and boxplot for age after cleaning
adm_pat_icu_train['age'].plot(kind='hist', bins=20)
plt.title('age (Cleaned)')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('plots/age_histogram_cleaned.png', dpi=200, bbox_inches='tight')
plt.close()

adm_pat_icu_train.boxplot(column='age')
plt.title('age (Cleaned)')
plt.savefig('plots/age_boxplot_cleaned.png', dpi=200, bbox_inches='tight')
plt.close()

# intime is a timestamp(not a real category)
adm_pat_icu_train = adm_pat_icu_train.drop(columns=['intime'])
adm_pat_icu_test = adm_pat_icu_test.drop(columns=['intime'], errors='ignore')

# List of categorical columns
categorical_cols = [col for col in adm_pat_icu_train.select_dtypes(include=['object', 'string']).columns if col != 'hospital_expire_flag']

# Cardinality cutoff (from lecture Dta Preprocessing)
cardinality_cutoff = 15

for col in categorical_cols:
    n_unique = adm_pat_icu_train[col].nunique()
    if n_unique > cardinality_cutoff:
        top_categories = adm_pat_icu_train[col].value_counts().head(cardinality_cutoff).index
        print(f"{col}: {n_unique} categories, keeping top {cardinality_cutoff}, rest -> 'Other'")
        adm_pat_icu_train[col] = adm_pat_icu_train[col].apply(lambda x: x if x in top_categories else 'Other')
        adm_pat_icu_test[col] = adm_pat_icu_test[col].apply(lambda x: x if x in top_categories else 'Other')

# Apply one-hot encoding
adm_pat_icu_train = pd.get_dummies(adm_pat_icu_train, columns=categorical_cols, drop_first=True)
adm_pat_icu_test = pd.get_dummies(adm_pat_icu_test, columns=categorical_cols, drop_first=True)

# Align columns
adm_pat_icu_train, adm_pat_icu_test = adm_pat_icu_train.align(adm_pat_icu_test, join='left', axis=1, fill_value=0)

num_cols = adm_pat_icu_train.select_dtypes(include='number').columns.drop('hospital_expire_ratio', errors='ignore').drop('hospital_expire_flag', errors='ignore').tolist()
scaler = StandardScaler()
adm_pat_icu_train[num_cols] = scaler.fit_transform(adm_pat_icu_train[num_cols])
adm_pat_icu_test[num_cols] = scaler.transform(adm_pat_icu_test[num_cols])

# Remove features with zero variance or that are duplicates
nunique = adm_pat_icu_train.nunique()
zero_var_cols = nunique[nunique <= 1].index.tolist()

adm_pat_icu_train = adm_pat_icu_train.drop(columns=zero_var_cols)
adm_pat_icu_test = adm_pat_icu_test.drop(columns=zero_var_cols, errors='ignore')

# Final X/y, also drop subject_id here since it's just an ID, not a real feature
X_train = adm_pat_icu_train.drop(columns=['hospital_expire_flag'])
y_train = adm_pat_icu_train['hospital_expire_flag']

X_test = adm_pat_icu_test.drop(columns=['hospital_expire_flag'])
y_test = adm_pat_icu_test['hospital_expire_flag']

# group-aware CV
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# Set up grid search for Logistic Regression
# print(adm_pat_icu['hospital_expire_flag'].unique()) ->[0 1] -> Binary LR
logreg = LogisticRegression(max_iter=5000, random_state=42)
param_grid_lr = {'C': [0.01, 0.1, 1, 10, 100]}
grid_lr = GridSearchCV(logreg, param_grid_lr, cv=cv, scoring='roc_auc')
grid_lr.fit(X_train, y_train, groups=groups.iloc[train_idx])

print("Best parameters (Logistic Regression):", grid_lr.best_params_)
logreg_best = grid_lr.best_estimator_

# Odds ratios - top 10 most influential features affecting odds of death
odds_ratios = pd.DataFrame({
    'feature': X_train.columns,
    'odds_ratio': np.exp(logreg_best.coef_[0])
}).sort_values('odds_ratio', ascending=False)

# Odds ratios of age OR=1.03 is clinically sensible but it wasnt in top 10
# print(odds_ratios[odds_ratios['feature'] == 'age'])

mask = ~odds_ratios['feature'].str.startswith(('intime_', 'diagnosis_'))
odds_ratios[mask].head(10).plot(x='feature', y='odds_ratio', kind='bar', figsize=(12,5))
plt.title('Top 10 Odds Ratios')
plt.axhline(y=1, color='r', linestyle='--')
plt.tight_layout()
plt.savefig('plots/odds_ratios.png', dpi=200)
plt.close()

# Set up grid search for Random Forest
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10]
}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=cv, scoring='roc_auc')
grid_rf.fit(X_train, y_train, groups=groups.iloc[train_idx])
print("Best parameters (Random Forest):", grid_rf.best_params_)
rf_best = grid_rf.best_estimator_
xgb_clf = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7]
}
grid_xgb = GridSearchCV(xgb_clf, param_grid_xgb, cv=cv, scoring='roc_auc')
grid_xgb.fit(X_train, y_train, groups=groups.iloc[train_idx])  # <-- important
print("Best parameters (XGBoost):", grid_xgb.best_params_)
xgb_best = grid_xgb.best_estimator_
models = {
    'Logistic Regression': logreg_best,
    'Random Forest': rf_best,
    'XGBoost': xgb_best
}
results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    print(f"\n{name}:")
    print(results[name])
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix: {name}')
    plt.savefig(f'plots/confusion_matrix_{name}.png', dpi=200)
    plt.close()

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC={results[name]["roc_auc"]:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (All Models)')
plt.legend()
plt.savefig('plots/roc_curve.png', dpi=200)
plt.close()
