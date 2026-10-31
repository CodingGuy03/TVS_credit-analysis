# TVS Credit E.P.I.C 8 — Analytics Case Study 2
## Risk-Adjusted Offer Amount Generation for Personal Loans

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-success)
![Case Score](https://img.shields.io/badge/Evaluation%20Score-100.0%20%2F%20100.0-brightgreen)

> **Unlocking Portfolio Growth Through Risk-Adjusted Loan Offers:** Moving from static historical limits to customer-specific capacity estimation, selectively expanding credit lines based on repayment risk.

---

##  Executive Problem Statement & Context
TVS Credit actively reaches out to its existing customer base to communicate eligibility for Personal Loans and quote maximum offer amounts. However, customer feedback indicates that many creditworthy borrowers requiring a Personal Loan find their offered amount insufficient and consequently drop out of the funnel. Conversely, simply increasing loan limits across the board would expose the portfolio to severe credit risk.

###  Core Objective
Develop an alternate, data-driven offer amount generation logic leveraging customer product holdings, repayment performance, and risk metrics such that:
1. **Relatively lower-risk customers** receive **higher loan amounts** than current offers (driving portfolio growth & conversion).
2. **Relatively higher-risk customers** receive **lower or equal loan amounts** (ensuring risk mitigation and capital protection).

###  Mandatory Evaluation Framework (0–100 Score)
| Evaluation Dimension | Weight | Mathematical Objective | Status |
| :--- | :---: | :--- | :---: |
| **Criterion 1: Portfolio Growth** | **30%** | $\text{Mean}(\text{Proposed}) \ge \text{Mean}(\text{Existing})$ overall | ✅ **PASS (30/30)** |
| **Criterion 2: Risk Reduction** | **40%** | For high-risk segment (top 25%), $\text{Mean}(\text{Proposed}) < \text{Mean}(\text{Existing})$ | ✅ **PASS (40/40)** |
| **Criterion 3: Taker Expansion** | **30%** | For Personal Loan takers, $\text{Mean}(\text{Proposed}) \ge \text{Mean}(\text{Existing})$ | ✅ **PASS (30/30)** |

---

##  Solution Architecture: Two-Stage Champion Engine

Instead of relying on historical offers as a rigid anchor with generic percentage adjustments, we decoupled **Borrowing Capacity** from **Default Risk** using a two-stage decisioning framework:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: CAPACITY ENGINE (LightGBM Regressor)                           │
│ "How much credit can this customer reasonably handle?"                  │
│  • Inputs: Multi-product exposures, live/closed balances, EMI velocity  │
│  • Output: BASELINE_CAPACITY (Unconstrained creditworthiness)           │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: RISK ENGINE (Tuned Cost-Sensitive XGBoost Classifier)          │
│ "How safely can we lend more?"                                          │
│  • Target: COMPOSITE_RISK_EVENT_RECENT_VINTAGE (Ever 30+ DPD in MOB 1-6)│
│  • Output: Predicted Probability of Default (Risk %)                    │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: OFFER POLICY LAYER (Risk-Elasticity Multiplier)                │
│  • Scales BASELINE_CAPACITY up for safe tiers & applies haircuts to risk│
│  • Enforces business guardrails (₹40K – ₹600K, rounded to nearest ₹5K)  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Mathematical Formulation: Risk-Elasticity Multiplier
Rather than setting arbitrary step-function cutoffs, we formulated a continuous **Risk-Elasticity Multiplier** anchored around the portfolio median risk:

$$\text{Multiplier}_i = \text{clip}\left( \left( \frac{\text{Median Risk}}{\text{Risk}_i} \right)^{\gamma}, \, 0.55, \, 1.50 \right)$$

$$\text{Proposed Offer}_i = \text{BASELINE\_CAPACITY}_i \times \text{Multiplier}_i \times \text{Capacity Boost}_i$$

* **Low Risk ($\text{Risk}_i < \text{Median}$):** Multiplier $> 1.0 \implies$ **Offer Expansion**.
* **High Risk ($\text{Risk}_i > \text{Median}$):** Multiplier $< 1.0 \implies$ **Prudent Risk Haircut**.
* **Damping Exponent ($\gamma = 0.60$):** Controls elasticity and prevents extreme outlier spikes.
* **Cash-Flow Capacity Boost ($1.08\times$):** Borrowers with a proven track record of servicing high EMIs (`MAX_EMI_SERVED_MOB_0_12 > 0`) and below-median risk receive an additional $+8\%$ boost.

---

##  Data Engineering & Feature Pipeline

The raw dataset (`157,557 rows × 128 columns`) spans multiple domains: Demographics, Delinquency, Repayment Performance, Multi-Product Exposures, Balances, and Bureau Enquiries.

* **Data Segregation:** Partitioned into training/test cohort (`BASE_TYPE == 'TEST'`: $119,891$ rows) and out-of-time/out-of-sample validation cohort (`BASE_TYPE == 'VALIDATION'`: $37,540$ rows).
* **Sentinel & Missing Value Treatment:** Replaced bureau sentinel flags (`-1.0` in `TOTAL_EMI_LIVE` and payment counts) with domain-appropriate medians/zeros.
* **High-Cardinality Consolidation:** Standardized fragmented employment segments (`SELF`, `SENP` $\rightarrow$ `SELF-EMPLOYED`; `SAL` $\rightarrow$ `SALARIED`), grouped rare product categories ($<100$ instances) into `OTHER`, and bucketed tele-calling dispositions (`BEST OUTCOME`, `LAST OUTCOME`) into `Positive`, `Negative`, `Call Dropped`, and `RNR`.
* ** Engineered Risk & Capacity Signals:**
  * `diff_pay_cnt`: Ratio of unsuccessful payments to total payment attempts across all loans.
  * `diff_live_pay_cnt`: Absolute delinquency count on active tradelines.
  * `MAX_EMI_SERVED_MOB_0_12` & `MAX_EMI_SERVED_MOB_12_24`: Rolling peak EMI servicing capacity over recent and historical 12-month windows.

---

##  Model Benchmarking & Explainability

### Risk Classification Models (Predicting `COMPOSITE_RISK_EVENT_RECENT_VINTAGE`)
Due to severe class imbalance ($\sim 0.4\%$ default rate), models were trained using cost-sensitive learning (`scale_pos_weight = 16.525` / balanced class weights) on disbursed loan takers:

| Model Architecture | 5-Fold CV AUC | Holdout AUC | Key Characteristics |
| :--- | :---: | :---: | :--- |
| **Logistic Regression** | $0.5925 \pm 0.0320$ | $0.5899$ | Linear baseline with median imputation & standard scaling |
| **XGBoost (Baseline)** | $0.5408 \pm 0.0098$ | $0.5968$ | Depth-4 trees with inverse class weighting |
| **Random Forest** | $0.5612 \pm 0.0180$ | $0.5988$ | 500 trees, balanced sub-sampling |
| **LightGBM** | $0.5489 \pm 0.0145$ | $0.6085$ | Leaf-wise tree growth (`is_unbalance=True`) |
| **XGBoost (Tuned Champion)** | **$0.6010 \pm 0.0112$** | **$0.6040$** | Regularized shallow trees (`max_depth=3`, `lr=0.012`, L1/L2 penalties) |

* **Top Predictive Risk Drivers:** `PEAK_PAYMENT_DELAY_ALL_LOANS`, `TOTAL_EMI_LIVE`, `CREDIT_HISTORY_DEPTH_FLAG`, `MAX_EMI_SERVED_MOB_0_12`, `RISK_GRADATION_OD`, and `EXTERNAL_RISK_SCORE`.

---

##  Key Results & Policy Performance

### 1. Multi-Iteration Policy Progression (Test Cohort: $N = 119,891$)
| Iteration Strategy | Avg Proposed Offer | C1: Portfolio Growth (30) | C2: Risk Reduction (40) | C3: Taker Expansion (30) | Total Score (100) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Existing Policy)** | ₹178,149 | — | — | — | — |
| **Iteration 1: Naive Flat Boost (+20%)** | ₹213,776 | 30.0 | 0.0 *(Fails Risk)* | 30.0 | **60.0 / 100** |
| **Iteration 2: Single-Stage Risk Multiplier** | ₹184,040 | 30.0 | 40.0 *(-18.4% cut)* | 30.0 *(+1.3% growth)* | **100.0 / 100** |
| **Iteration 3: Two-Stage Champion Engine** | **₹189,931** | **30.0** | **40.0** | **30.0** | **100.0 / 100** |

* **Portfolio Expansion:** Average offer across the portfolio increased from **₹178,149** to **₹184,040+** ($+3.3\%$ to $+6.6\%$).
* **High-Risk Mitigation:** Average offer to the top 25% riskiest customers was cut by **$-18.4\%$** (from **₹191,430** down to **₹156,303**).
* **Taker Ticket Size Growth:** Disbursed Personal Loan takers saw their average offer expand from **₹184,897** to **₹187,388**.

### 2. Final Presentation Output: Validation Cohort Quintiles ($N = 6,791$ Takers)
Evaluating the Two-Stage Champion Engine on unseen `VALIDATION` data demonstrates strong capacity expansion while keeping default rates tightly controlled in the highest offer tier:

| Quintile | Proposed Offer Amount Range | # Volume | Avg Offer Amount | # Risk Event Recent Vintage | % Risk Event Recent Vintage |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Q1 (Lowest)** | `(₹45,000, ₹110,000]` | 1,359 | ₹80,968 | 30 | 2.21% |
| **Q2** | `(₹110,000, ₹160,000]` | 1,358 | ₹134,370 | 23 | 1.69% |
| **Q3** | `(₹160,000, ₹210,000]` | 1,358 | ₹183,737 | 38 | 2.80% |
| **Q4** | `(₹210,000, ₹270,000]` | 1,358 | ₹237,828 | 39 | 2.87% |
| **Q5 (Highest)** | `(₹270,000, ₹575,000]` | 1,358 | **₹343,446** | **25** | **1.84%** |
| **Totals** | **All Validation Takers** | **6,791** | **₹196,053** | **155** | **2.28%** |

> **Key Takeaway:** The highest-offer segment (**Q5**) receives an average offer of **₹343,446** while maintaining a **1.84% default rate**—significantly lower than the portfolio average (**2.28%**) and middle tiers. This proves the engine successfully isolates high-capacity, creditworthy customers for limit expansion without concentrating credit risk.

---



##  Business Impact & Use Cases
* **Personalized Loan Offers at Scale:** Replaces rigid, one-size-fits-all grids with individualized capacity and risk pricing.
* **Higher Funnel Conversion:** Solves the "insufficient offer amount" drop-out problem for prime borrowers.
* **Smarter Capital Allocation:** Actively shifts capital away from high-delinquency segments ($-18.4\%$ exposure reduction) and reallocates it toward low-risk revenue generators.
