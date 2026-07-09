# Executive Summary: Government & Regulatory Briefing
## Customer Engagement & Product Utilization Analytics for Retention Strategy

**Prepared for:** Government Stakeholders & Regulatory Oversight Committees
**Objective:** To inform consumer protection policies, assess retail banking stability, and ensure fair banking practices.

**Portfolio size analyzed:** 10,000 retail banking customers
**Overall churn rate:** 20.4% | **Retention rate:** 79.6%

---

### 1. Headline Finding: Inactivity, Not Product Count, Drives Churn

The single strongest, most actionable signal in this portfolio is **member
activity status**. Inactive members churn at **26.9%**, nearly double the
**14.3%** churn rate of active members. This is a far larger and more
reliable gap than any product-holding or demographic factor observed,
and it is directly addressable through reactivation campaigns.

### 2. Hidden Risk: High-Value Customers Are Not Automatically Loyal

**6.5%** of the customer base qualifies as "Premium" (top-quartile balance
and salary), but **3.5% of the entire portfolio** falls into the "At-Risk
Premium Customer" category — premium clients who are inactive or
under-engaged. Notably, high-balance customers churn at **24.5%**, *above*
the portfolio-wide average of 20.4%. This confirms a critical and
counter-intuitive point: **balance size does not guarantee loyalty**, and
these clients represent a disproportionately large potential revenue loss
if they leave.

### 3. Product Cross-Selling Has a Ceiling

Customers holding 2 products churn the least (7.6%). However, churn rises
sharply for customers holding 3 or 4 products (82.7% and 100% respectively,
though the 4-product segment is a small sample of 60 customers and should
be interpreted cautiously). This suggests that **beyond a certain point,
additional product ownership is a symptom of dissatisfaction or forced
bundling rather than a driver of loyalty** — a nuance that pure cross-sell
targets can miss.

### 4. Weaker-Than-Expected Signals

Two commonly assumed retention levers did not hold up strongly in this
data:
- **Credit card ownership**: retention was nearly identical between holders
  (79.8%) and non-holders (79.2%) — a negligible difference.
- **Relationship Strength Index** (a composite of tenure, product depth, and
  balance-to-salary ratio): showed only a weak correlation with churn
  (r = 0.01), indicating that relationship *depth* alone is not a reliable
  predictor without also accounting for activity status.

### 5. Geographic and Demographic Variation

- **Germany** shows the highest regional churn rate at **32.4%**, notably
  above other markets in the portfolio — warranting local investigation
  into service quality, pricing, or competitive pressure.
- The **46–60 age segment** shows the highest churn rate at **51.1%**,
  suggesting retention messaging and offers should be tailored by life
  stage rather than applied uniformly.

### 6. Data Quality & Methodology

**Data Quality:** A rigorous data quality audit was performed before analysis. Findings included addressing impossible ages, negative balances, and invalid product counts via clipping (e.g. Balance clipped to 0, Age clipped to [18, 100]) and capping extreme outliers for balances and salary using the IQR method to preserve high-net-worth customer records.

**Methodology - Engineered Features:**
- **Engagement Score (0-100):** A composite score derived from activity status (40%), product count (25%), tenure (20%), and credit card ownership (15%).
- **Relationship Strength Index:** Blends normalized tenure (35%), products (35%), and balance-to-salary ratio (30%) to measure how deeply entrenched a customer is with the bank.
- **Premium Customer:** Flags customers in the top quartile of both Balance and Estimated Salary.
- **At-Risk Premium:** Flags Premium Customers who are either inactive or have an Engagement Score below 50.
- **Retention Score:** A 0-100 score summarizing the likelihood a customer will stay (average of Engagement Score and Relationship Strength Index).

---

## Strategic Recommendations

1. **Prioritize reactivation over acquisition.** The activity-status gap is
   the largest lever identified — a targeted campaign for inactive members
   is likely to be the single highest-ROI retention initiative available.
2. **Build a dedicated At-Risk Premium Customer program.** Relationship
   managers should receive proactive alerts for high-balance, low-engagement
   clients before they show explicit churn signals.
3. **Investigate the Germany market specifically.** A churn rate nearly 12
   points above the portfolio average justifies a focused root-cause review.
4. **Reconsider cross-sell targets past 2 products.** Sales incentives tied
   purely to product count may be counter-productive; needs-based selling
   should be favored over volume-based selling beyond this threshold.
5. **Segment retention offers by age group**, given the pronounced
   difference in churn between the 46–60 segment and the rest of the base.

---

*Full KPI detail, all 20 supporting charts, and the complete automatically
generated insight list are available in `processed_data/` and `charts/`,
and interactively via the Streamlit dashboard (`app.py`).*
