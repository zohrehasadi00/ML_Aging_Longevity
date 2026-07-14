# ML_Aging_Longevity

## Dictionary to the Columns

| Column | Stands For                                                         |
|--------|--------------------------------------------------------------------|
| **subject_id** | Patient unique ID                                                  |
| **hadm_id** | Hospital admission unique ID                                       |
| **icustay_id** | ICU stay unique ID                                                 |
| **admittime** | Hospital admission datetime  (**should be changed to age: age = admittime − dob**)                     |
| **dischtime** | Hospital discharge datetime   (**Temporal Leakage, must be removed**)                               |
| **admission_type** | Type of admission (EMERGENCY, URGENT, ELECTIVE, NEWBORN)           |
| **admission_location** | Where patient came from (ED, CLINIC, TRANSFER, etc.)               |
| **discharge_location** | Where patient went after discharge (HOME, DIED, SNF, etc.)  (**Direct Target Leakage, must be removed**)       |
| **insurance** | Insurance type (Medicare, Medicaid, Private, Government, Self Pay) |
| **language** | Primary language                                                   |
| **religion** | Religion                                                           |
| **marital_status** | Marital status                                                     |
| **ethnicity** | Race/ethnicity                                                     |
| **gender** | Sex (M/F)                                                          |
| **dob** | Date of birth (**should be changed to age: age = admittime − dob**)                                 |
| **dod** | Date of death (**Direct Target Leakage, must be removed**)                                                     |
| **expire_flag** | Patient died during hospital stay (0/1) (**Direct Target Leakage, must be removed**)   |
| **hospital_expire_flag** | **TARGET**: Patient died in hospital (0=survived, 1=died)          |
| **diagnosis** | Primary admission diagnosis code/text                              |
| **intime** | ICU admission time                                                 |
| **outtime** | ICU discharge time  (**Temporal Leakage, must be removed**)                                   |
| **los** | Length of stay (days) (**Temporal Leakage, must be removed**)                           |
| **first_careunit** | First ICU care unit (MED-SURG, CCU, etc.)                          |
| **last_careunit** | Last ICU care unit   (**Temporal Leakage, must be removed**)                                   |
| **first_wardid** | First ward ID                                      |
| **last_wardid** | Last ward ID   (**Temporal Leakage, must be removed**)                                                |
| **has_chartevents_data** | Whether charted events exist (0/1)  (**Proxy, must be removed**)                    |
| **dbsource** | Data source                                                        |

## how to start
- creat a virtual environment, install uv
- uv sync
- uv run .\improved_pipeline.py

## Who did what?
## 📋 Project Audit & Contribution Matrix

This matrix maps out the tasks completed during the audit phase of the **ML_Aging_Longevity** pipeline, aligned with specific lecture guidelines. 

| Process / Task | Zohreh | Ons | Pablo |
| :--- | :---: | :---: | :---: |
| **Repository & Environment Setup** | | | |
| Created GitHub repository, fixed initialization, and managed invitations | ✔ | | |
| Configured system logging (`INFO` level, console output via `setup_logging()`) | ✔ | | |
| Added missing core packages (`numpy`, `matplotlib`, `seaborn`, `missingno`, `scikit-learn`) to `requirements.txt` | | ✔ | |
| Added missing `xgboost` package to `requirements.txt`| | ✔ | |
| Cleaned pipeline script by moving all library imports to the top of the file | ✔ | | |
| **Data Loading & Data Dictionary** | | | |
| Loaded core MIMIC demo CSVs (`PATIENTS`, `ADMISSIONS`, `ICUSTAYS`) and logged shapes | ✔ | | |
| Merged core tables on `subject_id` / `hadm_id` and exported `df.csv` | ✔ | | |
| Created a general column dictionary for team alignment | ✔ | | |
| **Exploratory Data Analysis (EDA) & Plotting** | | | |
| Uploaded `plots/` folder with complete data plots (Histograms, Boxplots, MSNO) | ✔ | ✔ | |
| Generated age histogram/boxplot after outlier cleaning | | ✔ | |
| Plotted target variable distribution for class balance analysis | ✔ | | |
| Plotted Top 10 most influential features based on calculated Odds Ratios | ✔ | | |
| **Data Preprocessing & Outliers** | | | |
| Calculated missing-rates per column and dropped columns with >30% missingness (`drop_thresh = 0.3`) | ✔ | | |
| Adjusted outlier threshold to **2.5** based on lecture guidelines | | ✔ | |
| Adjusted categorical cardinality cutoff boundary from **20** to **15** | | ✔ | |
| Relocated MSNO matrix/bar chart placement to occur post-outlier removal | | ✔ | |
| Added a missing row-wise NaN histogram using `bins=30` | | ✔ | |
| Dropped `intime` before encoding (timestamp, not a real category, not fit for cardinality bucketing) | | ✔ | |
| **Data Leakage & Feature Selection** | | | |
| Identified and dropped direct target leakage columns (`discharge_location`, `expire_flag`, `dod`, `deathtime`, etc.) | ✔ | | ✔ |
| Dropped temporal leakage columns known only post-admission (`dischtime`, `outtime`, `los`, etc.) | ✔ | | ✔ |
| Dropped predictive proxy features (`has_chartevents_data`) | | | ✔ |
| Dropped duplicate ID/identifier columns (`row_id`, `hadm_id`, `icustay_id`) | ✔ | | ✔ |
| Found and removed `subject_id` still present in final model features (missed earlier) | | ✔ | |
| Formulated the engineering of the 'age' feature by calculating `admittime - dob` | | | ✔ |
| Ensured leaky and ID columns are completely dropped *before* splitting data / entering model | ✔ | | |
| **Data Splitting Strategy** | | | |
| Relocated train/test split sequence to occur *before* data preprocessing begins | | | ✔ |
| Implemented `StratifiedGroupKFold` to handle dependent samples and multiple patient admissions | | | ✔ |
| Adapted preprocessing pipeline and GridSearch structure to align with the new split strategy | | | ✔ |
| **Imbalance & Modeling** | | | |
| Evaluated class imbalance (found 33.8% minority class = mild, no resampling required) | ✔ | | |
| Set up binary Logistic Regression to predict survival (`0`) vs. death (`1`) | ✔ | | |
| Kept `lbfgs` solver with L2 regularization for small-to-medium dataset optimization | ✔ | | |
| Extracted and checked clinical plausibility of feature Odds Ratios (e.g., age at ~1.03) | ✔ | | |
