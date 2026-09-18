# Churn Prediction on Imbalanced Data — Precisely Build Sprint

Predicts which customers are at risk of churning (~4% base rate) and turns that into a
ranked, actionable list a retention team can work from.


## Requirements

- Python 3.10+
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- Jupyter (VS Code's built-in Jupyter extension, or `jupyter notebook`/`jupyter lab`)

Install (if needed):
```
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Data

Place the source CSVs in one folder. The notebook does **not** assume fixed filenames — it
scans every `.csv` in the folder and classifies each one by its actual columns:

- a file containing `churned` → treated as the churn labels
- a file containing `cust_id` and `full_name` → treated as the CRM/customer table
- a file containing `billing_id` → treated as billing
- a file containing `ticket_id` → treated as support tickets

If your files don't match this shape, Section 3 of the notebook will print the mismatch
clearly instead of failing silently.

## How to run

1. Open `churn_prediction.ipynb` in VS Code (Jupyter extension) or JupyterLab.
2. In **Section 2 (Imports and Configuration)**, set `DATA_DIR` to the folder holding your CSVs:
   ```python
   DATA_DIR = Path(r"C:\Users\LENOVO\Downloads\Precisely_Hackathon_Datasets")
   ```
3. Run all cells top to bottom (`Run All`). The notebook is sequential — later sections
   depend on variables created earlier, so it's not designed to be run out of order.
4. The final scored file is written next to wherever the notebook is running, as
   `customer_churn_predictions.csv`, and the last cell prints an auto-generated executive
   summary built entirely from values computed in the notebook (nothing is hard-coded).

## What the model does, in one paragraph

Customer, billing, and support records are joined into one row per customer (billing links via
email since it has no customer id; support tickets are resolved to a customer through a
fallback chain since `cust_ref` is inconsistently populated). Every candidate feature is passed
through a leakage audit before use. Because churn is rare (~4%), the model is trained with
class weighting rather than synthetic oversampling, evaluated primarily on PR-AUC and top-K
precision/recall/lift rather than accuracy, and calibrated so the output probabilities are
usable by a commercial team rather than just a ranking.


