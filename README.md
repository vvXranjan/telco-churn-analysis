# Telco Customer Churn Analysis

Customer churn analysis using Python, pandas, and EDA.  
Goal: identify key drivers of churn and recommend actions to reduce churn.

## Dataset
- Telco Customer Churn dataset (raw not tracked in repo)
- Cleaned dataset saved at: `data/cleaned/telco_churn_clean.csv`

## What I did (so far)
### Data Cleaning (Notebook 01)
- Loaded raw data and validated schema
- Converted `TotalCharges` from object → numeric (handled blanks as missing)
- Dropped 11 records with missing `TotalCharges` (mostly tenure=0)
- Dropped `customerID` (identifier only)
- Encoded target `Churn` (Yes/No → 1/0)
- Saved cleaned dataset: **7032 rows × 20 columns**
- Notebook: `notebooks/01_data_cleaning.ipynb`

## Tech Stack
- Python (pandas, numpy)
- JupyterLab
- Git + GitHub

## Project Structure
telco-churn-analysis/
├── data/
│ └── cleaned/
│ └── telco_churn_clean.csv
├── notebooks/
│ └── 01_data_cleaning.ipynb
├── requirements.txt
└── README.md


## Next Steps
- Notebook 02: Exploratory Data Analysis (EDA)
- Churn drivers: contract type, tenure, monthly charges, internet service, payment method
- Visual storytelling + insights
