import pandas as pd
import logging
import sys
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
LOGGER = logging.getLogger("improved_pipeline")

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
setup_logging()


def run_pipeline():
    patients = pd.read_csv('mimic_demo/PATIENTS.csv')
    admissions = pd.read_csv('mimic_demo/ADMISSIONS.csv')
    icustays = pd.read_csv('mimic_demo/ICUSTAYS.csv')
    LOGGER.info("Loaded core tables: patients=%s, admissions=%s, icustays=%s", patients.shape, admissions.shape, icustays.shape)

    # Merge tables (left joins as original)
    adm_pat = pd.merge(admissions, patients, on="subject_id", how="left")
    df = pd.merge(adm_pat, icustays, on=["subject_id", "hadm_id"], how="left")
    LOGGER.info("Merged dataframe shape: %s", df.shape)
    LOGGER.info(df.columns.tolist())

    missing_rate = df.isnull().mean()
    #TO DO: Should we leave this on 0.3? higher? lower?
    drop_thresh = 0.3
    cols_to_drop = missing_rate[missing_rate > drop_thresh].index.tolist()

    if cols_to_drop:
        LOGGER.info("Dropping high-missing columns (>%s): %s", drop_thresh, cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    # droping the IDs, cause they wont help with ML classification
    # remove los, to prevent data leakage, cause this is calculated after the patient is dead
    #TO DO: What else should we remove? How to know?
    df = df.drop(columns=(["subject_id", "hadm_id", "icustay_id", "row_id", "row_id_x", "row_id_y", "los"]))
    LOGGER.info("Non-Null Count of each column and their type: %s", df.info())
    LOGGER.info("Dataframe shape after cleanup: %s", df.shape)
    LOGGER.info("General info from the datafram:\n %s", df.describe())
    LOGGER.info("Sum of duplicates in each row: %s", df.duplicated().sum())
    LOGGER.info(df.head(10))
    LOGGER.info(df.isnull().mean().sort_values(ascending=False))

if __name__ == "__main__":
    run_pipeline()