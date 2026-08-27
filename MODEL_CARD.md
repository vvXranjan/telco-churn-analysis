# Model Card — Telco Churn Prediction (XGBoost)

## 1. Model purpose
Binary classification of whether a telecom customer will churn, used to support
retention targeting and to provide evidence-grounded explanations of each
prediction.

## 2. Intended use
- Flag customers likely to churn from their account/feature profile.
- Provide per-customer attribution (SHAP) and, optionally, a constrained
  natural-language explanation of the model's prediction.
- Intended for analysts and ML engineers exploring churn; not for fully
  autonomous decision-making without human review.

## 3. Out-of-scope use
- Causal inference (the model is predictive, not causal).
- Individual financial or legal decisions about specific people.
- Deployment to a different operator/market/period without revalidation.
- Claims about a person's actual behavior as fact.

## 4. Training data
The Telco Customer Churn dataset. Raw data is not tracked in this repository;
the cleaned version is `data/cleaned/telco_churn_clean.csv`.

## 5. Dataset size
Cleaned dataset: **7,032 rows × 20 columns**. Modeling set: **7,010 rows**
after removing 22 exact duplicate rows from a working copy (see §17). Churn
rate: **26.58%**.

## 6. Target definition
`Churn = 1` if the customer churned, `0` otherwise (converted from Yes/No in
Notebook 01).

## 7. Modeling methodology
- Stratified **70/15/15 train/dev/holdout** split (`random_state=42`).
- Preprocessing: `ColumnTransformer` — median imputation + `StandardScaler`
  for 4 numerical features; most-frequent imputation +
  `OneHotEncoder(handle_unknown="ignore")` for 15 categorical features
  (→ **45 transformed features**).
- Models compared: Logistic Regression, Random Forest, XGBoost; hyperparameters
  selected by 5-fold stratified CV on train scored by F1.
- The **holdout was evaluated exactly once** after all decisions were made.

## 8. Class imbalance handling
- Logistic Regression and Random Forest: `class_weight="balanced"`.
- XGBoost: `scale_pos_weight = 2.7746` (computed from training data only).

## 9. Model selection
Final model: **XGBoost** (`learning_rate=0.05`, `max_depth=3`,
`n_estimators=200`, `subsample=0.8`, `random_state=42`), selected by the
highest development-set F1.

## 10. Threshold selection
Decision threshold **0.525** chosen on the dev set to maximize F1 (never the
default 0.5). Re-select with actual business costs if available.

## 11. Final holdout metrics
| Metric | Value |
|---|---|
| ROC-AUC | 0.8530 |
| PR-AUC | 0.6571 |
| Precision | 0.5369 |
| Recall | 0.7814 |
| F1 | 0.6365 |
| Accuracy | 0.7633 |

Confusion matrix: TN=585, FP=188, FN=61, TP=218 (FPR 0.2432, FNR 0.2186).

## 12. SHAP explainability
`shap.TreeExplainer` (`feature_perturbation="tree_path_dependent"`, shap 0.52.0)
on the persisted pipeline. SHAP values are on the raw (log-odds) scale; additivity
`sum(SHAP) + base = margin` holds (max error ~2.4e-6). Top global features by
mean |SHAP|: `Contract_Month-to-month` (0.6318), `tenure` (0.3952),
`MonthlyCharges` (0.2934), `InternetService_Fiber optic` (0.2305),
`OnlineSecurity_No` (0.2009).

## 13. LLM explanation architecture
Notebook 05 / the API generate natural-language explanations from a **reduced
SHAP evidence set** (top 5 positive + top 5 negative contributors). The LLM is a
constrained verbalization layer, never a source of evidence.

## 14. Evidence-grounding safeguards
A deterministic verifier checks every generated explanation for: grounded
probability and threshold, known features, matching SHAP values, consistent
directions, absence of unsupported evidence, absence of causal claims, and
absence of ground-truth leakage. Valid responses → PASS; fabricated adversarial
responses → FLAG.

## 15. Known limitations
- The verifier reduces/detects classes of unsupported claims; it does not
  mathematically guarantee hallucination-free language.
- Keyword-based causal-language detection is imperfect.
- Observational data: predictive relationships do not establish causality.
- F1-based threshold balances precision and recall equally; a real deployment
  with known FP/FN costs should re-select the threshold.

## 16. Generalization limitations
Results may not generalize to another telecom company, geography, customer
population, pricing structure, or time period. A single 1,052-row holdout
provides one independent estimate with statistical uncertainty.

## 17. Duplicate-row handling
The cleaned dataset contains 22 exact duplicate rows. They were kept in the
source file but removed from a modeling copy **before** splitting (documented
decision in Notebook 03) so no identical record appears in more than one
partition.

## 18. Fairness / subgroup limitations
**No subgroup/fairness analysis was performed.** The dataset contains only
`gender` among standard protected attributes and lacks age, race, income,
location, and other demographic information, so a meaningful demographic
fairness evaluation was not possible. This must not be read as evidence of
fairness.

## 19. Monitoring / drift considerations
A lightweight drift checker (`drift/drift_check.py`) compares incoming batches
against the training distribution (Kolmogorov-Smirnov for numerical features,
chi-square for categorical features) and flags statistically significant
shifts. Statistical significance is not operational importance; flagged shifts
should be reviewed with business context.

## 20. Security considerations
- API keys are read from environment variables (`OPENAI_API_KEY`,
  `OPENAI_MODEL`) and are never logged, committed, or returned.
- The artifact SHA-256 checksum is verified at load time; startup fails loudly
  on mismatch.
- No customer PII beyond model features is processed or stored by the API.