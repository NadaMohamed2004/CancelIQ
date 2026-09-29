# CancelIQ — Hotel Booking Cancellation Intelligence Platform

*Predictive analytics · revenue-at-risk modeling · explainable AI · BI dashboard · AI copilot*

Project report — Data Mining & Visualization, MTC Digilians Program
Author: **Nada Mohamed Khalil**

CancelIQ analyses 119,389 cleaned historical hotel bookings, predicts cancellation risk with a calibrated XGBoost model, estimates the financial exposure behind that risk, and makes the results usable through a Power BI dashboard and a bilingual (Arabic/English) browser assistant. An optional local data API supports free‑form questions, with Gemini available when a server‑side key is configured.

---

## Table of contents

1. [Business problem](#1-business-problem)
2. [Data understanding & cleaning](#2-data-understanding--cleaning)
3. [Feature engineering](#3-feature-engineering)
4. [EDA & statistical validation](#4-eda--statistical-validation)
5. [Machine learning](#5-machine-learning)
6. [Benchmark against a reference project](#6-benchmark-against-a-reference-project)
7. [Power BI dashboard](#7-power-bi-dashboard)
8. [Chatbot](#8-chatbot)
9. [Repository structure](#9-repository-structure)
10. [Running it locally](#10-running-it-locally)
11. [Caveats — read before quoting numbers](#11-caveats--read-before-quoting-numbers)
12. [Conclusion](#12-conclusion)

---

## 1. Business problem

The hotel needed answers to six practical questions:

- Which bookings are likely to be cancelled?
- How much revenue is at risk from those cancellations?
- What factors actually drive cancellation behaviour?
- Which specific bookings or customers need proactive attention?
- What operational patterns can the hotel improve?
- How can historical booking data support better day‑to‑day decisions?

## 2. Data understanding & cleaning

The raw dataset (36 columns, 119,390 rows) was cleaned deliberately rather than with a blanket `dropna()`. Missingness itself was treated as signal where relevant, and every decision was justified against the business context.

| Issue | Finding | Decision |
|---|---|---|
| `company` missing | 94.31% missing | Recoded as "No Company" — 38.22% cancel rate when missing vs 17.52% when present |
| `agent` missing | 13.69% missing | Recoded as "No Agent" — 24.66% vs 39.00% cancel rate |
| `country` missing | 0.41% missing | Recoded as "Unknown" |
| `children` missing | 4 rows | Filled with 0, based on booking context, not asserted as certain |
| Duplicates | 0 exact duplicates | None removed |
| Undefined categories | `meal`, `market_segment`, `distribution_channel` | Standardized to "Unknown" |
| Negative ADR | 1 row (adr = ‑6.38) | Removed as invalid |
| Extreme ADR | max 5,400 | Kept — no evidence of entry error |
| Zero‑night bookings | 715 rows | Kept — legitimate for cancellations/no‑shows |
| Outliers | thousands flagged by IQR | Investigated individually, not auto‑removed |
| PII (name, email, phone, card) | present | Dropped entirely from the analytical dataset |
| Leakage (`reservation_status`, date) | near‑perfect mapping to target | Dropped — post‑outcome information |

## 3. Feature engineering

New features were built with business meaning, not just statistical convenience: `total_guests`, `total_nights`, `estimated_revenue` (ADR × nights), `party_size_category`, `stay_type`, `lead_time_category`, `has_booking_changes`, `has_previous_cancellation`, `has_special_requests`, `requires_parking`, `room_type_changed` — plus, at the modeling stage, `guest_cancellation_rate`, `adr_per_person`, and cyclical (sin/cos) encoding of arrival month.

A correlation heatmap caught redundant feature pairs that one‑at‑a‑time testing had missed:

- `requires_parking` ↔ `required_car_parking_spaces` (r = 0.99)
- `has_special_requests` ↔ `total_of_special_requests` (r = 0.86)
- `has_booking_changes` ↔ `booking_changes` (r = 0.80)

The final analytical dataset held 44 columns, trimmed to 38 features for modeling after removing IDs, dates, and redundant pairs.

## 4. EDA & statistical validation

Every visual finding was backed by a formal test — Chi‑square with Cramér's V for categorical associations, Mann‑Whitney U for numeric comparisons — so no claim rests on a bar chart alone. All findings are reported as **association, not causation**.

| Variable | Cancellation‑rate spread | Effect size | Verdict |
|---|---|---|---|
| Deposit type | No Deposit 28.4% · Non‑Refund 99.4% · Refundable 22.2% | V = 0.4815 (strong) | Top priority |
| Previous cancellations | 33.9% → 91.6% | V = 0.2709 (moderate) | High priority |
| Market segment | Groups 61.1% · Online TA 36.7% · Direct 15.3% | V = 0.2668 (moderate) | High priority |
| Lead time | median 45d (kept) vs 113d (cancelled) | 0.3785 | High priority |
| Hotel type | City 41.7% vs Resort 27.8% | V = 0.1365 (weak) | Kept, not standalone |
| Customer type | Transient 40.8% vs Group 10.2% | V = 0.1364 (weak) | Kept, low priority |
| ADR | median 92.5 vs 96.2 | 0.0608 (very small) | Kept, low priority |

## 5. Machine learning

The train/test split was **chronological** (80/20), not random, because bookings have a real time order that a random split would ignore.

- Train: Jul 2015 – Mar 2017 (35.85% cancel rate)
- Test: Mar – Aug 2017 (40.79% cancel rate)

**Preprocessing:** numeric features median‑imputed and standardized; low‑cardinality categoricals one‑hot encoded; high‑cardinality fields (`agent`, `company`, `country`) frequency‑encoded to avoid an exploding, overfit‑prone feature space. All transforms fit on train only.

Six model families were trained and tuned under the exact same evaluation regime — a small random hyper‑parameter search, every candidate scored on pooled out‑of‑time predictions from four expanding‑window temporal folds (Apr 2016 → Mar 2017), so no model gets an easier validation scheme than another.

**Temporal cross‑validation (train set):**

| Model | Accuracy | Precision | Recall | F1 | ROC‑AUC | PR‑AUC |
|---|---|---|---|---|---|---|
| Dummy baseline | 63.63% | 0.00 | 0.00 | 0.00 | 0.482 | 0.353 |
| Logistic Regression (tuned) | 80.05% | 72.41% | 72.94% | 72.67% | 0.869 | 0.818 |
| LightGBM (tuned) | 82.62% | 77.95% | 72.82% | 75.30% | 0.898 | 0.858 |
| Random Forest (tuned) | 82.89% | 78.09% | 73.63% | 75.79% | 0.896 | 0.853 |
| CatBoost (tuned) | 82.94% | 79.50% | 71.56% | 75.32% | 0.901 | 0.861 |
| Blend (XGBoost + LightGBM) | 83.05% | 80.27% | 70.81% | 75.25% | 0.904 | 0.863 |
| **XGBoost (tuned) — winner** | **83.05%** | **80.44%** | **70.56%** | **75.18%** | **0.904** | **0.863** |

XGBoost was selected **programmatically** — not hard‑coded — as the model with the best temporal‑CV accuracy, statistically tied with the Blend. It was then calibrated (Platt scaling) and evaluated **once**, untouched, on the held‑out future test window:

**One‑time holdout test (Mar – Aug 2017, unseen):**

| Metric | Score |
|---|---|
| Accuracy | 79.49% |
| Precision | 76.54% |
| Recall | 71.70% |
| F1 | 74.04% |
| ROC‑AUC | 0.885 |
| PR‑AUC | 0.846 |

The gap between the 83.05% temporal‑CV accuracy and the 79.49% test accuracy is expected and reported honestly: the test window has a meaningfully higher actual cancellation rate (40.8% vs 35.9% in training), which is ordinary drift when forecasting forward in time, not a bug. **This test‑set figure — not the higher cross‑validation numbers — is the one to quote publicly.**

## 6. Benchmark against a reference project

Before building this pipeline, a reference project was used to calibrate expectations for scope and quality: a fellow capstone team's "Hotel Intelligence" dashboard (Snipers Team), covering the same brief on the same booking dataset. Its published result was a Random Forest classifier reaching 81% accuracy, F1 of 0.79, and AUC of 0.86.

| Metric | Reference project (Random Forest) | This project — Random Forest (tuned) | This project — Logistic Regression (tuned) |
|---|---|---|---|
| Accuracy | 81.0% | 82.89% | 80.05% |
| ROC‑AUC | 0.860 | 0.896 | 0.869 |
| F1‑score | 0.79 | 0.758 | 0.727 |

These comparisons are **informal** — the reference project did not use the same train/test split or evaluation protocol. For this project's final, later‑in‑time test, the selected XGBoost achieved 79.49% accuracy and 0.885 ROC‑AUC. The reference result set an informal benchmark; the strongest evidence for the model is the one‑time chronological holdout test above, reported with precision, recall and F1 alongside accuracy.

## 7. Power BI dashboard

Three pages sit on top of the model's outputs.

### Overview
![Dashboard overview](screenshots/01_dashboard_overview.png)

KPIs, revenue by deposit type, booking share by hotel type — 119,389 cleaned bookings, 37.04% cancelled, €42.72M gross booked value (including cancelled bookings), €16.73M estimated value of cancelled bookings, €16.81M model‑expected exposure, €101.83 average ADR, and 40,453 high‑risk bookings at the 50% risk threshold.

### Cancellation drivers
![Cancellation drivers](screenshots/02_cancellation_drivers.png)

Segment/deposit breakdowns next to the model's top feature importances. The updated SHAP feature summary ranks guest country, deposit type, agent, total special requests, lead time and market segment among the leading signals. Feature importance measures contribution to predictions; it does **not** establish a causal effect or, by itself, the direction of an effect.

### Risk calculator & waiting list
![Risk calculator and waiting list](screenshots/03_risk_calculator_waiting_list.png)

A slider‑driven scenario tool for testing hypothetical bookings, paired with a waiting list ranked by expected realised value = (1 − predicted cancellation risk) × ADR × nights. It demonstrates prioritisation logic, not a live queue.

## 8. Chatbot

The bilingual assistant uses updated facts from the 119,389‑row scored booking file and the new model metadata. Its offline answers cover selected dashboard questions; its browser‑based calculator evaluates the embedded XGBoost trees and displays risk plus one‑variable what‑if changes. When the supplied local Python API is running, a Data API mode offers free‑form questions grounded in aggregates calculated from `bookings_scored.csv`. Gemini can optionally help interpret and phrase those answers when `GEMINI_API_KEY` is configured on the server; the model itself continues to calculate booking risk.

### English interface
![Chatbot assistant, English](screenshots/04_chatbot_assistant_en.png)

### Arabic interface
![Chatbot assistant, Arabic](screenshots/05_chatbot_assistant_ar.png)

The same assistant mirrors itself in Arabic — right‑to‑left layout included — with the same risk table, the same caveats, and the same calculator underneath.

Answers are limited by the historical data and available fields. The assistant presents probabilities and estimated exposure for decision support; **the hotel remains responsible for any operational action**. The Gemini connection requires a valid API key and has not been verified with a live key for this report.

## 9. Repository structure

```
CancelIQ/
├── clean_data/
│   └── hotel_bookings_final.csv        # Cleaned historical dataset (119,389 rows, no personal columns)
├── data/
│   └── hotel_booking.csv               # Raw bookings export (input of the cleaning notebook)
├── dashboard_data/
│   └── CancelIQ_dashboard.csv          # Data export used for the Power BI dashboard
├── code/
│   ├── notebooks/                      # Analysis + modeling notebooks
│   └── scripts/
│       ├── app.py                      # Local Data API + chatbot server
│       ├── hotel_utils.py              # Shared helpers (needed to load preprocessor.joblib)
│       └── score_bookings.py           # Batch scoring
├── models/                             # best_model.*, preprocessor.joblib, metadata, feature_importance.csv
├── output/                             # bookings_scored.csv, waiting_list_priority.csv, dashboard_kpis.json
├── visualization/                      # Power BI dashboard, theme, hotel_chatbot.html
├── screenshots/
├── docs/                               # Project report
├── requirements.txt
└── README.md
```

## 10. Running it locally

**Requirements:** Python 3.10+. The saved model and preprocessor only load with the
library versions pinned in `requirements.txt`.

```bash
git clone https://github.com/NadaMohamed2004/CancelIQ.git
cd CancelIQ

python -m venv venv

# Activate the environment (pick ONE):
# Windows (CMD / PowerShell):  venv\Scripts\activate
# Windows (Git Bash):          source venv/Scripts/activate
# macOS/Linux:                 source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **Tip:** use `python -m pip` instead of bare `pip`, so packages are always installed
> into the Python you just activated. If several Python versions are installed and
> `python -m venv venv` picks the wrong one, call the interpreter you want explicitly
> (for example `py -3.11 -m venv venv` on Windows, or the full path to `python.exe`).
> Any Python from 3.10 up works with the pinned versions.

**A. Standalone chatbot (no server, no Python):** open `visualization/hotel_chatbot.html`
in a browser for the built-in dashboard answers and the client-side booking risk
calculator. High-risk threshold: **50%**.

**B. Chatbot with the local Data API:**

```bash
python code/scripts/app.py
# then open http://127.0.0.1:8000
```

Answers come from aggregates computed on `output/bookings_scored.csv`. Optionally set
`GEMINI_API_KEY` on the server so Gemini helps interpret and phrase answers
(**never put the key in browser HTML or a public repository**):

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your_key"
# Windows Git Bash / macOS / Linux
export GEMINI_API_KEY="your_key"
```

Without a key, basic summaries and comparisons still work. The Gemini connection has
not been tested with a live key. Endpoints: `GET /api/health` · `GET /api/metrics` ·
`POST /api/chat`. The CSV loads at server start; restart after replacing it. If port
8000 is busy, set the `PORT` environment variable.

**C. Batch scoring of new bookings** (works from any folder):

```bash
python code/scripts/score_bookings.py input.csv output.csv
```

The input CSV needs these raw columns: `hotel, lead_time, arrival_date_year,
arrival_date_month, arrival_date_day_of_month, stays_in_weekend_nights,
stays_in_week_nights, adults, children, babies, meal, country, market_segment,
distribution_channel, is_repeated_guest, previous_cancellations,
previous_bookings_not_canceled, reserved_room_type, deposit_type, agent, company,
days_in_waiting_list, customer_type, adr, total_of_special_requests`. The output adds
`risk_prob` and `risk_level` (Low / Medium / High).

**Notebooks:** `pip install jupyter matplotlib seaborn`, then open the files in
`code/notebooks/`. **Power BI:** `visualization/CancelIQ_dashboard.pbix` opens in
Power BI Desktop (Windows only); import `Hotel_Intelligence_Theme.json` via
View > Themes > Browse for themes.

**Troubleshooting:**

- An error loading `preprocessor.joblib` / `FrequencyEncoder` means the library
  versions differ from `requirements.txt`, or `hotel_utils.py` was moved.
- `pip: command not found` (Git Bash): Python is not on your PATH. Use
  `python -m pip ...` after activating the venv.
- `Unable to create process using ...python.exe` from `py -m venv`: the default Python
  registered with the `py` launcher no longer exists. Run `py -0p` to list installed
  versions, then create the venv with one that exists (`py -3.11 -m venv venv` or the
  full path to its `python.exe`).
- `venv\Scripts\activate` does nothing in Git Bash: use
  `source venv/Scripts/activate` instead.
- Dependency install fails: make sure the venv is active (you should see `(venv)` in the
  prompt) and that Python is 3.10 or newer (`python --version`).

**Data sources:**

| Dataset | Link | Notes |
|---|---|---|
| Public hotel booking demand dataset | https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand | Public dataset of hotel bookings (arrivals 2015–2017) |
| Cleaned dataset used in this project | https://www.kaggle.com/datasets/nadamohamed25/hotel-bokking?select=hotel_bookings_final.csv | `hotel_bookings_final.csv`, 119,389 rows |

The raw export is included at `data/hotel_booking.csv` and the cleaned file at
`clean_data/hotel_bookings_final.csv`. The cleaning step removes the personal columns
(name, email, phone-number, credit_card) — see Section 2.

## 11. Caveats — read before quoting numbers

- **Test vs. cross‑validation.** The reported 79.49% accuracy, 71.70% recall, 74.04% F1 and 0.885 ROC‑AUC come from a later‑in‑time holdout test. The Random Forest and Logistic Regression figures in Section 6 are training‑only temporal cross‑validation and shouldn't be presented as final test results.
- **External comparison.** The reference project's 81% accuracy and 0.860 AUC were measured on a different split — informal calibration, not a controlled head‑to‑head.
- **Financial exposure is an estimate.** Revenue‑at‑risk figures are ADR × nights, not an audited loss figure.
- **Correlation, not causation.** Feature importance and Cramér's V/Mann‑Whitney results describe association, never a causal mechanism.
- **Data quirks, not policy insight.** Non‑refundable bookings are ~99% cancelled and largely tied to one country in this dataset — an artifact of the data, not a general rule about non‑refundable rates. The "Unknown" market segment covers only 2 bookings; its 100% rate should be ignored. Parking‑related bookings are excluded as leakage (zero cancellations with parking recorded).
- **Scope.** The data covers arrivals from 2015–2017 with no live hotel feed connected. The model shows correlations from that period, which can drift over time — as already seen between the training and test windows.

Source of figures: `best_model_metadata.json`, `dashboard_kpis.json` and `bookings_scored.csv` in this project package.

## 12. Conclusion

CancelIQ brings together data cleaning, leakage checks, feature engineering, statistical analysis, temporal model validation, a Power BI dashboard and a bilingual assistant. The reported test performance is **79.49% accuracy and 0.885 ROC‑AUC**; the assistant provides updated, data‑grounded summaries and browser‑based risk scenarios, with optional Gemini wording through the local API. Historical data and informal external benchmarking remain important limits on operational claims.

---

*Project by **Nada Mohamed Khalil**.*
