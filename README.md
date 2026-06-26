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
