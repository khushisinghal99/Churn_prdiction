Problem 2 — Churn Prediction on Imbalanced Data

1. Data Discovery: Schema Over Filenames

Problem: The input directory may contain multiple CSV files, and filenames should not be assumed to be fixed.

Decision: I discovered all CSV files and classified them using their columns:

- "cust_id", "full_name" → CRM
- "billing_id" → Billing
- "ticket_id" → Support
- "churned" → Churn labels

Why: Schema-based discovery is more robust than relying on exact filenames.

Alternative considered: Hard-code filenames.
Why rejected: A renamed file or differently named dataset would break the pipeline or could cause the wrong file to be used.

Cost: Stricter schema validation can stop the pipeline when files are ambiguous, but this is safer than silently using incorrect data.

---

2. Messy Values: Preserve Rows Instead of Dropping Them

Problem: The data contains invalid values such as negative billing amounts, ""-"" in numeric fields, invalid dates, and future dates.

Decision: I converted invalid values to missing ("NaN"/"NaT") and kept the original row. I also created quality flags where useful, such as a negative-amount flag.

Why: Dropping rows could remove specific types of customers and introduce selection bias. Keeping the customer allows the model's imputation step to handle missing values later.

Alternative considered: Delete every row containing an invalid value.
Why rejected: A single bad field should not cause an otherwise useful customer record to disappear.

Cost: Some information is lost for that particular field, but the customer remains available for modelling.

---

3. Customer Matching: Use Multiple Keys Carefully

Problem: Billing does not contain the CRM "cust_id", while support can contain either a customer ID or an email.

Decision: I used a controlled matching strategy:

1. Match support using the direct CRM customer ID when available.
2. If the reference looks like an email, match using normalized email.
3. Otherwise fall back to reporter email.
4. Keep unmatched records rather than forcing a match.

Billing is matched using normalized email.

Why: Customer identifiers are inconsistent, so assuming one universal key would lose data.

Alternative considered: Match using names or approximate/fuzzy matching.
Why rejected: Names are less reliable and fuzzy matching could create false customer matches.

Cost: Some records remain unmatched, but this is safer than incorrectly assigning activity to another customer.

---

4. Aggregate Before Joining

Problem: One customer can have multiple billing records or support tickets.

Decision: I aggregate billing and support information to one row per customer before joining it to CRM.

Why: Directly joining all tickets to customers could create multiple rows for the same customer. That could duplicate the customer's churn label and give customers with many tickets disproportionate influence during training.

Cost: Individual transaction/ticket-level detail is reduced, but customer-level behaviour is appropriate for a customer churn model.

---

5. Missing Relationships: Keep the Customer

Decision: I use an inner join between CRM and churn labels because a supervised model needs both customer information and a known target. Billing and support are left-joined because they are additional sources.

Why: A labelled customer without billing or support activity is still a valid customer. Missing activity should become missing/zero features rather than causing the customer to disappear.

Cost: Some customers have incomplete information, but removing them would unnecessarily reduce the training population.

---

6. Leakage: Check Before Modelling

Problem: Some fields may contain information that would only be known after a customer had already churned.

Decision: I explicitly audit features for leakage. In particular, "account_status" is checked for suspiciously strong alignment with churn. If it appears to represent a post-churn state, it is removed.

Why: A feature that directly reveals the target can produce excellent offline performance but fail in real use.

Alternative considered: Keep every predictive feature.
Why rejected: Predictive power is not useful if the information would not be available when making the prediction.

Important limitation: There is no explicit churn-event date in the supplied data, so billing/support recency cannot be guaranteed to represent only pre-churn behaviour. I therefore treat this as a limitation rather than claiming perfect temporal leakage control.

---

7. Train/Test Split: Stratified Instead of Temporal

Problem: Churn is rare and there is no reliable churn/prediction timestamp.

Decision: I use an 80/20 stratified split with a fixed random seed.

Why: Stratification keeps approximately the same churn proportion in training and testing. Without a reliable prediction date, a temporal split would require inventing a cutoff.

Alternative considered: Temporal split.
Why rejected: It would give a false impression of a real-world time-based evaluation.

What would change my decision: If a genuine prediction/event timestamp became available, I would prefer a temporal evaluation to test performance on future customers.

---

8. Handling the 4% Churn Rate

Problem: Churn is a minority class, so accuracy can be misleading.

Decision: I use class-weighted models rather than SMOTE.

Why: Class weighting gives more importance to churn examples while keeping the original customer data unchanged.

Alternative considered: SMOTE.
Why rejected: With a small number of positive examples, synthetic customers may not represent realistic customer behaviour.

What would make me reconsider: If there were substantially more minority examples and validation showed that synthetic sampling improved minority-class performance without harming generalization.

---

9. Model Selection: Compare Different Types

Decision: I compare:

- Logistic Regression as an interpretable baseline
- Random Forest for nonlinear relationships and interactions
- Histogram Gradient Boosting as another nonlinear approach

I avoid a large hyperparameter search because the objective is to make a reliable, explainable decision within the hackathon time limit.

Why: I want to establish a baseline and then test whether nonlinear models provide useful improvement rather than assuming a complex model is automatically better.

Cost: A larger search might find a slightly better configuration, but it would increase complexity and reduce the time available for data quality, analysis and validation.

---

10. Evaluation: PR-AUC and Top-K Instead of Accuracy

Problem: With only about 4% churn, a model could achieve high accuracy by predicting almost everyone as non-churn.

Decision: PR-AUC is the primary model metric. ROC-AUC is reported as a secondary metric, while Precision, Recall and F1 at 0.5 are treated as reference metrics.

I also calculate Precision, Recall and Lift at the top 5%, 10% and 20% of customers.

Why: A retention team usually has limited capacity. The practical question is not simply "How many customers did the model classify correctly?" but "If we contact the highest-risk customers first, how many churners can we find?"

Example: Recall@10% tells us what fraction of all churners are captured by targeting the top 10% of customers. Lift@10% tells us how much better that targeting is than contacting customers randomly.

---

11. Threshold Decision: Do Not Invent a Business Cost

Problem: There is no supplied cost for contacting a customer, giving a discount, or losing a customer.

Decision: I do not choose an arbitrary probability threshold as the final business policy. Instead, I use ranked top-K targeting and provide threshold analysis separately.

Why: Choosing a threshold such as 0.5 would be a technical default, not a business decision.

What would change my decision: If the business supplied contact capacity and costs, I could choose a threshold using expected cost/value. For example, if the retention team can contact only 10% of customers, top-10% targeting becomes a directly defined operating rule.

---

12. Calibration and Actionable Output

Decision: After selecting the model, I calibrate its probabilities using sigmoid calibration.

Why: Ranking customers is useful, but calibrated probabilities are more meaningful when the output is described as risk.

I then produce a customer-level output with:

- churn probability
- risk band
- ticket activity
- unresolved tickets
- failed payments
- payment recency
- other relevant customer signals

Why: The final result should be usable by a retention team rather than being only a model evaluation table.

Limitation: The supplied customer dataset contains labelled historical customers, so the final risk file is primarily a demonstration of scoring and ranking. In production, the model should score a separate current, unlabeled customer population.

---

13. Explainability and Error Analysis

Decision: I use model coefficients for Logistic Regression and permutation importance for the selected model. I also inspect errors by customer segments.

Why: Feature importance can show which variables are associated with model predictions, while error analysis can reveal groups where the model performs differently.

Important distinction: These are predictive relationships, not proof of causation.
