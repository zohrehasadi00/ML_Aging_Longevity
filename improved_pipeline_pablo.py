import pandas as pd

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

# Merge patients and admissions on 'subject_id'
adm_pat = pd.merge(admissions, patients, on='subject_id', how='left')

# Merge the result with icustays on ['subject_id', 'hadm_id']
adm_pat_icu = pd.merge(adm_pat, icustays, on=['subject_id', 'hadm_id'], how='left')

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

import matplotlib.pyplot as plt

# Plot distribution of the target variable
adm_pat_icu['hospital_expire_flag'].value_counts().plot(kind='bar')
plt.title('Distribution of Hospital Expire Flag')
plt.xlabel('Hospital Expire Flag (0=Survived, 1=Died)')
plt.ylabel('Count')
plt.show()

# Plot gender distribution
adm_pat_icu['gender'].value_counts().plot(kind='bar')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Plot admission_type distribution
adm_pat_icu['admission_type'].value_counts().plot(kind='bar')
plt.title('Admission Type Distribution')
plt.xlabel('Admission Type')
plt.ylabel('Count')
plt.show()

# Plot insurance distribution
adm_pat_icu['insurance'].value_counts().plot(kind='bar')
plt.title('Insurance Distribution')
plt.xlabel('Insurance')
plt.ylabel('Count')
plt.show()

# Plot ethnicity distribution
adm_pat_icu['ethnicity'].value_counts().plot(kind='bar')
plt.title('Ethnicity Distribution')
plt.xlabel('Ethnicity')
plt.ylabel('Count')
plt.show()

# Histogram for length of stay (los)
adm_pat_icu['los'].plot(kind='hist', bins=20)
plt.title('Distribution of Length of Stay (LOS)')
plt.xlabel('Length of Stay (days)')
plt.ylabel('Frequency')
plt.show()

# Boxplot for LOS by hospital_expire_flag
adm_pat_icu.boxplot(column='los', by='hospital_expire_flag')
plt.title('LOS by Hospital Expire Flag')
plt.xlabel('Hospital Expire Flag (0=Survived, 1=Died)')
plt.ylabel('Length of Stay (days)')
plt.suptitle('')
plt.show()

# Death rate by admission type
death_by_admtype = adm_pat_icu.groupby('admission_type')['hospital_expire_flag'].mean()
death_by_admtype.plot(kind='bar')
plt.title('Death Rate by Admission Type')
plt.xlabel('Admission Type')
plt.ylabel('Death Rate')
plt.show()

# Death rate by insurance
death_by_insurance = adm_pat_icu.groupby('insurance')['hospital_expire_flag'].mean()
death_by_insurance.plot(kind='bar')
plt.title('Death Rate by Insurance')
plt.xlabel('Insurance')
plt.ylabel('Death Rate')
plt.show()


# Split: grouped and stratified
from sklearn.model_selection import StratifiedGroupKFold

X = adm_pat_icu.drop(columns=['hospital_expire_flag'])
y = adm_pat_icu['hospital_expire_flag']
groups = adm_pat_icu['subject_id']  # <-- use patient ID

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(X, y, groups=groups))  # <-- one fold as test

# create split datasets
adm_pat_icu_train = adm_pat_icu.iloc[train_idx].copy()
adm_pat_icu_test = adm_pat_icu.iloc[test_idx].copy()


# Preprocessing

# Calculate missing value rates
missing_rate = adm_pat_icu_train.isnull().mean().sort_values(ascending=False)

# Drop columns with more than 25% missing data
threshold = 0.25
cols_to_drop = missing_rate[missing_rate > threshold].index.tolist()

adm_pat_icu_train = adm_pat_icu_train.drop(columns=cols_to_drop)
adm_pat_icu_test = adm_pat_icu_test.drop(columns=cols_to_drop, errors='ignore')

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

# Cap LOS at 99th percentile to handle extreme outliers
los_cap = adm_pat_icu_train['los'].quantile(0.99)
adm_pat_icu_train['los'] = adm_pat_icu_train['los'].clip(upper=los_cap)
adm_pat_icu_test['los'] = adm_pat_icu_test['los'].clip(upper=los_cap)

# List of categorical columns
categorical_cols = [col for col in adm_pat_icu_train.select_dtypes(include=['object', 'string']).columns if col != 'hospital_expire_flag']

# Apply one-hot encoding
adm_pat_icu_train = pd.get_dummies(adm_pat_icu_train, columns=categorical_cols, drop_first=True)
adm_pat_icu_test = pd.get_dummies(adm_pat_icu_test, columns=categorical_cols, drop_first=True)

# Align columns
adm_pat_icu_train, adm_pat_icu_test = adm_pat_icu_train.align(adm_pat_icu_test, join='left', axis=1, fill_value=0)

from sklearn.preprocessing import StandardScaler

# Select numeric columns (excluding target and IDs)
num_cols = [col for col in adm_pat_icu_train.columns 
            if adm_pat_icu_train[col].dtype != 'object' 
            and col not in ['hospital_expire_flag', 'subject_id', 'hadm_id', 'icustay_id', 'row_id_x', 'row_id_y', 'row_id']]

scaler = StandardScaler()
adm_pat_icu_train[num_cols] = scaler.fit_transform(adm_pat_icu_train[num_cols])
adm_pat_icu_test[num_cols] = scaler.transform(adm_pat_icu_test[num_cols])

# Remove features with zero variance or that are duplicates
nunique = adm_pat_icu_train.nunique()
zero_var_cols = nunique[nunique <= 1].index.tolist()

adm_pat_icu_train = adm_pat_icu_train.drop(columns=zero_var_cols)
adm_pat_icu_test = adm_pat_icu_test.drop(columns=zero_var_cols, errors='ignore')

# Final X/y
X_train = adm_pat_icu_train.drop(columns=['hospital_expire_flag'])
y_train = adm_pat_icu_train['hospital_expire_flag']

X_test = adm_pat_icu_test.drop(columns=['hospital_expire_flag'])
y_test = adm_pat_icu_test['hospital_expire_flag']



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# group-aware CV
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# Set up grid search for Logistic Regression
logreg = LogisticRegression(max_iter=5000, random_state=42)
param_grid_lr = {'C': [0.01, 0.1, 1, 10, 100]}
grid_lr = GridSearchCV(logreg, param_grid_lr, cv=cv, scoring='roc_auc')
grid_lr.fit(X_train, y_train, groups=groups.iloc[train_idx])

print("Best parameters (Logistic Regression):", grid_lr.best_params_)
logreg_best = grid_lr.best_estimator_

from sklearn.ensemble import RandomForestClassifier

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

import xgboost as xgb

xgb_clf = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7]
}
grid_xgb = GridSearchCV(xgb_clf, param_grid_xgb, cv=cv, scoring='roc_auc')
grid_xgb.fit(X_train, y_train, groups=groups.iloc[train_idx])  # <-- important

print("Best parameters (XGBoost):", grid_xgb.best_params_)
xgb_best = grid_xgb.best_estimator_

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve

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
    plt.show()

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC={results[name]["roc_auc"]:.2f})')

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (All Models)')
plt.legend()
plt.show()