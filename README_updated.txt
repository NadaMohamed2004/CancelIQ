CancelIQ — Hotel Booking Cancellation Intelligence Platform
============================================================

DATA AND MODEL
  hotel_bookings_final.csv      cleaned historical dataset: 119,389 rows
  bookings_scored.csv           booking-level risk, classification and estimated exposure
  waiting_list_priority.csv     historical, positive-value waiting-list examples ranked by expected realised value
  dashboard_kpis.json           dashboard totals and risk counts
  feature_importance.csv        model signals; importance is not causation or direction of effect
  best_model.json / .joblib      final tuned XGBoost model
  preprocessor.joblib           fitted preprocessing; requires hotel_utils.py and compatible scikit-learn
  best_model_metadata.json      selected threshold, calibration and temporal test metrics
  score_bookings.py             batch scoring: python score_bookings.py input.csv output.csv
  prediction-hotelbookingcancel.ipynb  modeling notebook supplied with the project

UPDATED CHATBOT
  hotel_chatbot (1).html is the original bilingual assistant updated with the
  new embedded XGBoost trees, preprocessing statistics, calibration and dashboard
  facts. Open it directly in a browser for the built-in dashboard answers and
  client-side booking risk calculator. The high-risk threshold is 50%.

  chatbot_updated_bundle.zip includes the updated interface, historical scored
  data, and a local Python API. Unzip, install pandas (pip install pandas), run
  python app.py in the extracted directory, then open http://127.0.0.1:8000.
  The Data API button answers questions using aggregates computed from the new
  bookings_scored.csv. Optionally set GEMINI_API_KEY on the server for Gemini
  to help interpret questions and phrase responses. The key must not go into
  browser HTML or a public repository. The Gemini connection was not tested
  with a live key. Without it, basic API summaries and comparisons still work.
  Supported endpoints: GET /api/health, GET /api/metrics, POST /api/chat.
  The historical CSV is loaded at server start; restart after replacing it.
  The HTML's embedded model/facts require regeneration if model or data changes.

VALIDATED FIGURES FOR A PUBLIC POST
  119,389 cleaned bookings; 37.04% cancelled; 40,453 high-risk bookings.
  Final XGBoost, later-in-time holdout test: accuracy 79.49%, recall 71.70%,
  F1 74.04%, ROC-AUC 0.885. Training-only temporal CV: Random Forest accuracy
  82.89%, ROC-AUC 0.896; Logistic Regression accuracy 80.05%, ROC-AUC 0.869.
  Do not present the CV figures as final test results. A published reference
  project's 81% accuracy and 0.860 AUC were measured with a different split;
  comparisons with it are informal, not controlled head-to-head results.
  Financial exposure uses ADR × nights and is an estimate, not audited loss.
  Historical data cover arrivals in 2015–2017; no live hotel feed is connected.

Source of figures: best_model_metadata.json, dashboard_kpis.json and
bookings_scored.csv in this project package.
