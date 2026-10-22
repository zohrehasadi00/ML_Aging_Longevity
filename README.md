Machine Learning for Aging and Longevity
Supervised Classification

Files

* `improved_pipeline.py` contains the revised machine learning pipeline.
* `compare_pipelines.py` is used for the comparison between the original-style and revised pipelines in Part C.
* `requirements.txt` contains the required Python packages.

Data

The MIMIC demo data are not included in the submission.

To run the code, the following files are needed:

* `PATIENTS.csv`
* `ADMISSIONS.csv`
* `ICUSTAYS.csv`

They should be placed in a folder called `mimic_demo`.

The folder structure should look like this:

```text
project_folder/
├── improved_pipeline.py
├── compare_pipelines.py
├── requirements.txt
└── mimic_demo/
    ├── PATIENTS.csv
    ├── ADMISSIONS.csv
    └── ICUSTAYS.csv
```

Setup

The project was tested with Python 3.11.9.

Install the required packages with:

```text
pip install -r requirements.txt
```

Running the improved pipeline

```text
python improved_pipeline.py
```

Running the comparison for Part C

```text
python compare_pipelines.py
```

The scripts generate the model results and plots used in the report.
