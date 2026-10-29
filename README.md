# MUC02 — Credit Pre-Screening Engine

### Loan Intake Optimisation | Calibo AI Academy · Path 2 · KLU Batch

**Phase:** Phase 2 — Supervised Learning Part 1  
**Industry:** BFSI — Retail Lending (personal loans)  
**Notebook:** `credit_PreScreening(1).ipynb`  
**Dataset required:** `MUC02_Credit_PreScreening_Dataset.csv`

---

## 1. Project overview

A mid-size private bank receives around **3,000 personal-loan applications per month**. Each application takes about **25 minutes** for a credit analyst to review. The bank wants to understand whether a machine-learning model can identify clear-cut historical declines so analysts can spend more time on applications that need human judgement.

The project uses six months of simulated historical application data, from April to September 2026. It explores approval patterns, creates affordability features, trains a Logistic Regression model to predict the historical `decision`, evaluates mistakes, chooses an auto-decline probability threshold, and checks possible gender-related outcome gaps.

### Main business question

> Can a pre-screening model remove some applications from the manual review queue while keeping the share of historically approved applications among auto-declines at or below 5%? How many applications and analyst hours could this represent each month?

**Important limitation:** `decision` is the credit team's historical decision, not the customer's actual repayment or default outcome. The model learns to reproduce historical decisions, including any patterns or bias in them. The results are not proof that an applicant will default or that an automatically declined applicant is not creditworthy. Treat the proposed threshold as an analysis/pilot recommendation, not a production-ready lending policy.

---

## 2. Project files

| File                                        | Purpose                                                                                                                                                                                             |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `credit_PreScreening(1).ipynb`              | Main notebook containing the CBIM Problem Canvas and Steps 5.1–5.7, code, charts, findings and policy briefing.                                                                                     |
| `MUC02_KLU_Credit_PreScreening.docx(1).pdf` | Student assignment brief with the business context, data dictionary, seven required steps, deliverables and evaluation criteria.                                                                    |
| `MUC02_KLU_Trainer_Briefing.pptx(1).pdf`    | Trainer briefing that explains the business story, CBIM canvas, seven steps, setup and submission requirements.                                                                                     |
| `MUC02_KLU_Expected_Outcomes.pptx(1).pdf`   | Reference outcomes for comparing your own results. The source says small differences are expected depending on cleaning choices; conclusions and business interpretation matter.                    |
| `MUC02_KLU_Trainer_Notes.docx(1).pdf`       | Trainer reference only. The document explicitly says **not for student distribution**; do not include it in a student-facing public repository or redistribute it.                                  |
| `MUC02_Credit_PreScreening_Dataset.csv`     | Required input data described in the assignment. Obtain the official CSV from the course materials and place it beside the notebook. It was not included among the files available for this README. |

The notebook's file search may recognise common filename variations, but the recommended setup is to use the exact dataset filename shown above.

---

## 3. Technology and setup

### Tools used

- Python
- Jupyter Notebook
- Pandas and NumPy for data handling
- Matplotlib and Seaborn for charts
- scikit-learn for data preparation, Logistic Regression and evaluation

Install the dependencies in the same Python environment used by Jupyter:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Start Jupyter from the project folder:

```bash
jupyter notebook
```

Open `credit_PreScreening(1).ipynb`.

### Recommended run procedure

1. Put the notebook and official CSV in the same folder.
2. Open the notebook and run the dataset-loading cell. Confirm that it loaded the intended CSV.
3. Select **Kernel → Restart & Run All**.
4. Wait for every cell to finish.
5. Confirm that the plots and printed results appear and no cell reports an error.
6. Save the notebook with its final outputs.

Do not rely only on outputs saved from an earlier run. Restarting and running all cells verifies that the current notebook can execute from beginning to end.

---

## 4. CBIM Problem Canvas

The notebook begins with a problem canvas because the business decision should be defined before writing code.

- **Situation:** Approximately 3,000 applications arrive each month and each receives about 25 minutes of manual review.
- **Complication:** Analysts spend time on applications that may be clear declines, while applicants needing judgement must wait in the same queue.
- **Question:** Which applications could be removed from manual review, and at what probability threshold, without exceeding the agreed 5% wrong-decline limit?
- **SSOT (single source of truth):** The supplied dataset; its `decision` column records the historical judgement of the credit team.
- **Success definition:** Select a threshold that meets the 5% rule and quantify the number of applications and analyst hours potentially freed.
- **Stakeholders:** Credit Policy Analyst, Head of Retail Lending, credit analysts and applicants affected by the decision.

The 5% rule is measured **within the auto-decline group**: at most 5% of applications sent to auto-decline should be ones that historical analysts approved. It is not a guarantee about true creditworthiness or future defaults.

---

## 5. Dataset overview

The assignment describes **12,000 simulated loan applications** received from April through September 2026, with 19 original columns. One row represents one application.

Important fields:

| Column                  | Meaning                                                                   |
| ----------------------- | ------------------------------------------------------------------------- |
| `application_id`        | Application identifier                                                    |
| `application_date`      | Date the application was received                                         |
| `age`, `gender`         | Applicant information                                                     |
| `city`, `city_tier`     | City and city tier                                                        |
| `employment_type`       | Salaried, Self-Employed Professional, Business Owner or Gig / Contract    |
| `years_in_current_job`  | Years in the current job or business                                      |
| `monthly_income`        | Declared gross monthly income in INR                                      |
| `existing_emi`          | Monthly EMI payments on existing loans                                    |
| `credit_score`          | Credit bureau score; blank where there is no credit history               |
| `credit_history_months` | Length of credit history; 0 means no history                              |
| `num_active_loans`      | Number of currently active loans                                          |
| `delinquencies_24m`     | Late or missed repayments in the previous 24 months                       |
| `loan_purpose`          | Stated reason for the loan                                                |
| `loan_amount`           | Amount requested                                                          |
| `tenure_months`         | Requested repayment period                                                |
| `proposed_emi`          | Monthly EMI added by the proposed loan                                    |
| `decision`              | Historical credit-team decision: Approved or Declined; this is the target |

The original notebook reports **6,353 Approved (52.94%)** and **5,647 Declined (47.06%)** before the analysis.

---

## 6. Step 5.1 — Load, inspect and clean

**Built by: Lohitha**

### What happens in this step

The notebook imports the libraries, loads the CSV, checks row/column counts and data types, counts historical decisions, and checks missing values and unusual incomes.

### Issue 1: 441 missing monthly incomes

Missing income values are concentrated among Business Owner and Self-Employed Professional applicants. The notebook fills missing values using the median income for the applicant's employment type. The median is less sensitive to extreme values than the mean, and the employment-type grouping recognises that income distributions can differ across groups.

The notebook records this as a cleaning decision rather than silently changing the data.

### Issue 2: 1,039 blank credit scores

Every application with a blank `credit_score` has `credit_history_months = 0`. These are treated as **new-to-credit (NTC)** applicants, not automatically as bad data.

The notebook creates `is_ntc`:

- `1` means no credit score / no recorded credit history.
- `0` means a credit score is present.

The flag preserves the information that a score was missing even when the missing score is filled for modelling. Dropping these records would remove about 8.7% of applicants and a real customer group.

### Issue 3: 12 unusually high monthly incomes

Twelve Salaried records have monthly incomes above ₹6,00,000. The notebook treats these as likely annual income values entered into a monthly-income field and divides them by 12. The cleaning log explains the assumption. This is a reasoned correction, not proof of what each applicant intended to enter.

### Why this step matters

Cleaning decisions can affect model features and results. A reviewer should be able to understand what was changed, why it was changed, and which customer groups could be affected.

---

## 7. Step 5.2 — What drives approval today?

**Built by: Ashika**

This step analyses historical approval rates by credit-score band and employment type before any model is trained.

### Credit score patterns

The notebook groups scored applicants into bands: Below 600, 600–649, 650–699, 700–749, 750–799 and 800+, while keeping NTC applicants separate.

The reference outcomes report approval rising from about **4.3% for scores 600–649 to 32.8% for scores 650–699**. Applicants with no score have a historical approval rate of about **53.9%**. This suggests the bank's past decisions have a strong score-related pattern, but the NTC result shows that missing scores should not automatically trigger rejection.

### Employment type patterns

The notebook reports these historical approval rates:

| Employment type            | Approval rate |
| -------------------------- | ------------: |
| Salaried                   |        58.37% |
| Business Owner             |        51.17% |
| Self-Employed Professional |        50.56% |
| Gig / Contract             |        34.94% |

Gig / Contract applicants have the lowest historical approval rate and a shorter median time in their current job in this dataset. These are associations in historical decisions, not proof that every applicant in that employment group is risky.

### Why this step matters

It identifies patterns in existing credit decisions and provides context for interpreting the model later. It also shows that NTC applicants are treated differently from applicants with low scores.

---

## 8. Step 5.3 — Affordability feature engineering

**Built by: Ashika**

Feature engineering creates additional variables from the original columns.

### Feature 1: FOIR

**Fixed Obligation to Income Ratio (FOIR)** estimates the share of monthly income used for existing and proposed loan payments:

```text
foir = (existing_emi + proposed_emi) / monthly_income
```

Example: existing EMI ₹8,000 + proposed EMI ₹7,000, divided by monthly income ₹50,000 gives FOIR = 0.30, or **30%**.

A higher FOIR means more of the applicant's monthly income is committed to loan repayments.

### Feature 2: Loan-to-income ratio

```text
loan_to_income = loan_amount / (monthly_income * 12)
```

This compares the requested loan amount with annual income.

### Feature 3: NTC flag

```text
is_ntc = 1 if credit_score is missing, otherwise 0
```

### FOIR findings

| FOIR band | Applicants | Historical approval rate |
| --------- | ---------: | -----------------------: |
| Below 30% |      5,491 |                   65.47% |
| 30–40%    |      2,497 |                   58.99% |
| 40–50%    |      1,766 |                   47.85% |
| 50–55%    |        531 |                   32.39% |
| Above 55% |      1,715 |                   15.63% |

Approval rates decrease as FOIR increases, with a strong drop after the 50% range. The assignment's reference outcome suggests a proposed FOIR policy limit around **50–55%**. This is a policy interpretation of historical data, not a formally validated lending rule.

---

## 9. Step 5.4 — Logistic Regression pre-screener

**Built by: Suresh**

### Target and feature selection

The model predicts whether the historical decision was Declined:

- `Declined = 1`
- `Approved = 0`

A positive coefficient pushes a prediction towards Declined; a negative coefficient pushes it towards Approved.

The assignment-specified model features are `credit_score`, `is_ntc`, `foir`, `loan_to_income`, `delinquencies_24m`, `num_active_loans`, `years_in_current_job`, `age`, `tenure_months`, `city_tier`, `employment_type` and `loan_purpose`.

The notebook excludes `application_id`, `application_date`, `city`, `gender` and `decision` from the features. In particular, `decision` cannot be an input because it is the label being predicted.

### Train/test split and preprocessing

The notebook uses an 80/20 split with `stratify=y` and `random_state=42`, giving 9,600 training applications and 2,400 test applications.

The preprocessing pipeline:

1. Learns imputation values from training data.
2. Fills missing income and credit-score values.
3. Recalculates derived affordability features consistently.
4. Scales numeric features.
5. One-hot encodes employment type and loan purpose.
6. Fits Logistic Regression.

Keeping preprocessing inside a training-fitted pipeline reduces the risk of information from the test set leaking into model preparation.

### Evaluation at the default threshold of 0.50

The notebook's saved output reports:

| Metric                         | Result |
| ------------------------------ | -----: |
| Accuracy                       | 79.08% |
| ROC-AUC                        |  0.867 |
| Precision for Declined         | 79.11% |
| Recall for Declined            | 75.47% |
| Accuracy for scored applicants |  81.2% |
| Accuracy for NTC applicants    |  56.7% |

- **Accuracy:** share of all test predictions that match the historical decision.
- **Precision for Declined:** among applications predicted as Declined, the share that were historically Declined.
- **Recall for Declined:** among all historically Declined applications, the share the model identified.
- **ROC-AUC:** how well the model ranks historically declined applications above approved ones across thresholds. It does not decide the business threshold.

### Confusion matrix

| Actual historical decision | Predicted Approved | Predicted Declined |
| -------------------------- | -----------------: | -----------------: |
| Approved                   |              1,046 |                225 |
| Declined                   |                277 |                852 |

At threshold 0.50, **225 applicants historically approved by the bank are predicted as Declined**. Since the positive class is Declined, these are false positives. The model also predicts Approved for 277 applications that were historically Declined.

This is why accuracy is not the only metric that matters: a wrong decline has a different business impact from a decline the model misses.

### Coefficients in business language

The notebook's leading drivers include:

- **FOIR (positive):** higher repayment obligations relative to income push towards Declined.
- **Credit score (negative):** a higher score pushes towards Approved.
- **Recent delinquencies (positive):** more late or missed payments push towards Declined.
- **Gig / Contract (positive):** this category pushes towards Declined relative to the model's reference category.
- **Salaried (negative):** this category pushes towards Approved relative to the reference category.

Coefficients show the patterns learned from historical decisions. They do not establish cause and effect or prove future repayment risk.

### NTC limitation

Accuracy is **56.7% for NTC applicants**, compared with 81.2% for scored applicants. This weak performance supports routing NTC applicants to a human analyst rather than relying on the model to automatically decline them.

---

## 10. Step 5.5 — Auto-decline threshold

**Built by: Suresh**

Logistic Regression returns a probability of Declined. The threshold is the probability level at which the proposed workflow would place an application into the auto-decline lane.

The bank's rule is that **no more than 5% of the auto-declined group should be applications historically approved by the credit team**. The notebook tests several thresholds and compares the size of the auto-decline group with the wrong-decline share.

| Threshold | Share auto-declined | Share of auto-declines historically approved |
| --------: | ------------------: | -------------------------------------------: |
|      0.50 |               44.8% |                                   20.8–20.9% |
|      0.60 |               37.6% |                                   16.8–16.9% |
|      0.70 |               30.8% |                                   12.0–12.1% |
|      0.80 |               23.5% |                                     6.9–7.1% |
|      0.85 |               19.7% |                                         5.9% |
|  **0.90** |           **15.2%** |                                    **4.66%** |
|      0.95 |               10.0% |                                        2.08% |

Minor differences in rounded percentages may appear between the assignment reference deck and the notebook. Use the results from a fresh run of your own notebook consistently in the final submission.

### Why choose 0.90?

At 0.50, the auto-decline lane is large, but roughly one in five of those applications was historically approved. At 0.90, the saved notebook reports that 4.66% of the auto-decline group was historically approved, meeting the 5% limit. The finer threshold search reports 0.90 as the lowest tested threshold meeting the limit in this run.

### Estimated monthly business impact

Using the assignment's assumptions:

- Monthly volume: 3,000 applications.
- Manual review time: 25 minutes per application.
- At 0.90: 15.2% auto-declined, or about **456 applications per month**.
- Estimated time freed: 456 × 25 minutes = 11,400 minutes, or about **190 analyst hours per month**.
- In the test set, 17 of 365 auto-declined applications were historically approved (**4.66%**), scaling to about 21 applications per month at the stated volume.
- The model catches about 30.8% of all historical declines at this threshold.

These are estimates based on a single held-out test split. They are not guaranteed future savings or a guarantee that the wrong-decline rate will remain below 5%.

### NTC safeguard

The saved notebook found **12 NTC applicants among the 365 test applications in the auto-decline lane**. Because the model's NTC accuracy is low, the recommendation is to send all NTC applicants to manual review. If this safeguard is applied, recompute the number auto-declined, the wrong-decline share and the estimated hours saved. The 456 applications / 190 hours figures are the unadjusted threshold estimates.

---

## 11. Step 5.6 — Fairness check

**Built by: Akhilandra**

The notebook does not use gender as a model feature, but this does not automatically guarantee fair outcomes. It compares historical approval rates by gender, rates within employment type, income and job mix, and model auto-decline rates.

### Main observations

- Historical approval rate: **50.4% female vs 54.1% male**, a 3.7 percentage-point gap.
- Median monthly income: about **₹45,000 for female applicants vs ₹55,000 for male applicants**.
- Female applicants are more likely to be in Gig / Contract work in the dataset.
- Overall model auto-decline rates are reported as almost identical by gender at the selected threshold.
- Within Self-Employed Professional applicants, the historical approval gap is larger than in some other employment groups.

These comparisons are descriptive. They do not prove why the gap exists or show that all fairness concerns have been solved. Income and employment type may act as proxies for group differences, so outcomes should be monitored over time and across relevant subgroups.

---

## 12. Step 5.7 — Credit Policy Briefing

**Written by: Akhilandra, using findings from all sections**

The final notebook section converts technical results into a short recommendation for the Credit Policy Analyst. It should cover:

1. The proposed threshold and estimated applications/hours per month.
2. The two strongest decision drivers, supported by figures.
3. A group the model should not decide for yet, and why.
4. Fairness concerns and one additional data source needed before wider deployment.

The notebook's core recommendation is a **controlled pilot at threshold 0.90**, with NTC applicants routed to humans, gender outcomes monitored, and actual repayment data obtained before wider use. The target in this project is historical analyst decisions, not whether customers repaid.

---

## 13. Team contributions

The notebook identifies these owners:

| Team member | Steps / contribution                                                |
| ----------- | ------------------------------------------------------------------- |
| Lohitha     | CBIM Problem Canvas and Step 5.1 — loading, inspection and cleaning |
| Ashika      | Steps 5.2–5.3 — approval patterns and affordability features        |
| Suresh      | Steps 5.4–5.5 — Logistic Regression and auto-decline threshold      |
| Akhilandra  | Steps 5.6–5.7 — fairness check and Credit Policy Briefing           |

The assignment expects each step to contain both runnable code and a Markdown explanation of what was done and what the result means.

---

## 15. Glossary

- **BFSI:** Banking, Financial Services and Insurance.
- **CBIM Problem Canvas:** Framework for defining the situation, complication, question, source of truth, success definition and stakeholder before modelling.
- **FOIR:** Existing and proposed monthly EMIs divided by monthly income.
- **NTC:** New-to-credit applicant with no recorded credit history.
- **Imputation:** Filling missing data using a defined rule such as a median.
- **One-hot encoding:** Converting categories into binary indicator features.
- **Logistic Regression:** A supervised classification model that estimates the probability of a class.
- **Threshold:** Probability cut-off used to determine when a prediction triggers an action.
- **Confusion matrix:** Table comparing actual labels with predicted labels.
- **False positive in this project:** Historically Approved but predicted Declined, because `Declined = 1`.
- **ROC-AUC:** Metric for ranking the positive class above the negative class across possible thresholds.
- **Data leakage:** When information that should be held out from training influences model fitting or preprocessing.
- **Out-of-time validation:** Testing on a later period to assess how performance may generalise to future applications.

---

## Final takeaway

This use case connects the full supervised-learning workflow: define the business problem, inspect and clean data, explore historical patterns, engineer features, train and evaluate Logistic Regression, select a business threshold, estimate operational impact, check fairness, and communicate a recommendation.

**The central lesson:** a model probability is not a credit policy by itself. The business sets the risk limit, and responsible use requires human-review safeguards, validation on fresh data, actual repayment outcomes and ongoing fairness monitoring.
