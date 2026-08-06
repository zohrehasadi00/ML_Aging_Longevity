# Machine Learning for Aging and Longevity

## Group members

- Zohreh Asadi — 519188 — M.Sc. Computer Science
- Ons Srasra — 466203 — M.Sc. Computational Engineering Science
- Pablo Strecker García — 391542 — M.Sc. Human Factors

## Setup and run

1. Create and activate a virtual environment.
2. Install the dependencies:

   ```powershell
   uv sync
   ```

   or:

   ```powershell
   pip install -r requirements.txt
   ```

3. Run the improved pipeline:

   ```powershell
   uv run .\improved_pipeline.py
   ```

4. Run the comparison used in Part C:

   ```powershell
   uv run .\compare_pipelines.py
   ```

## Required data

Place these files inside the `mimic_demo` folder:

```text
PATIENTS.csv
ADMISSIONS.csv
ICUSTAYS.csv
```
