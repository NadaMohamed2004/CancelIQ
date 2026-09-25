"""Score new bookings with the saved model (no notebook needed).

    python score_bookings.py new_bookings.csv scored_output.csv [models_dir]

Input CSV: the raw booking columns of the original dataset (hotel, lead_time, arrival_date_year, arrival_date_month,
arrival_date_day_of_month, stays_in_weekend_nights, stays_in_week_nights, adults, children, babies, meal, country,
market_segment, distribution_channel, is_repeated_guest, previous_cancellations, previous_bookings_not_canceled,
reserved_room_type, deposit_type, agent, company, days_in_waiting_list, customer_type, adr, total_of_special_requests).
Optional: sibling_count, agent_day_lead_count (computed from the batch if omitted).
Missing agent / company -> "No Agent" / "No Company". Agent & company ids are written like "9.0".
Requires the versions listed in models/best_model_metadata.json and hotel_utils.py next to this file.
"""
import sys, json, numpy as np, pandas as pd, joblib
from pathlib import Path
from hotel_utils import MONTH_MAP


def derive(df):
    d = df.copy()
    d["children"] = d["children"].fillna(0)
    for c, dflt in (("agent", "No Agent"), ("company", "No Company"), ("country", "Unknown")):
        d[c] = d[c].map(lambda x, dflt=dflt: dflt if pd.isna(x) else (str(float(x)) if isinstance(x, (int, float, np.number)) and c != "country" else str(x)))
    guests = d["adults"] + d["children"] + d["babies"]
    d["lead_time_category"] = pd.cut(d["lead_time"], [-1, 7, 30, 90, 180, np.inf], labels=["Last Minute", "Short Term", "Medium Term", "Long Term", "Very Long Term"]).astype(str)
    d["party_size_category"] = np.select([guests == 0, guests == 1, guests <= 3, guests <= 5], ["No Guests", "Solo", "Small", "Medium"], "Large")
    wk, wd = d["stays_in_weekend_nights"], d["stays_in_week_nights"]
    d["stay_type"] = np.select([(wk > 0) & (wd > 0), wk > 0, wd > 0], ["Weekend + Weekday", "Weekend Only", "Weekday Only"], "No Stay")
    d["guest_cancellation_rate"] = d["previous_cancellations"] / (d["previous_cancellations"] + d["previous_bookings_not_canceled"] + 1)
    d["adr_per_person"] = d["adr"] / (d["adults"] + d["children"] + 1)
    d["arrival_month_num"] = d["arrival_date_month"].map(MONTH_MAP)
    d["arrival_month_sin"] = np.sin(2 * np.pi * d["arrival_month_num"] / 12)
    d["arrival_month_cos"] = np.cos(2 * np.pi * d["arrival_month_num"] / 12)
    d["lead_time_x_special_requests"] = d["lead_time"] * d["total_of_special_requests"]
    d["guest_cancellation_rate_x_lead_time"] = d["guest_cancellation_rate"] * d["lead_time"]
    d["has_company"] = (d["company"] != "No Company").astype(int)
    d["has_agent"] = (d["agent"] != "No Agent").astype(int)
    d["no_guests_recorded"] = (guests == 0).astype(int)
    if "sibling_count" not in d or "agent_day_lead_count" not in d:
        d["arrival_date"] = pd.to_datetime({"year": d["arrival_date_year"], "month": d["arrival_month_num"], "day": d["arrival_date_day_of_month"]})
        key = ["hotel", "arrival_date", "market_segment", "agent", "company", "country", "reserved_room_type", "adr", "lead_time",
               "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies", "deposit_type", "customer_type", "meal"]
        d["sibling_count"] = d.groupby(key)["lead_time"].transform("size")
        d["agent_day_lead_count"] = d.groupby(["hotel", "arrival_date", "agent", "lead_time"])["lead_time"].transform("size")
    return d


def score(df, models_dir="models"):
    M = Path(models_dir); meta = json.loads((M / "best_model_metadata.json").read_text())
    pre = joblib.load(M / "preprocessor.joblib"); mdl = joblib.load(M / "best_model.joblib")
    d = derive(df)
    raw = mdl.predict_proba(pre.transform(d[meta["feature_columns_in"]]))[:, 1]
    z = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6)))
    p = 1 / (1 + np.exp(-(meta["calibration"]["a"] * z + meta["calibration"]["b"])))
    out = df.copy(); out["risk_prob"] = p.round(4)
    out["risk_level"] = np.select([p >= meta["risk_levels"]["high_from"], p >= meta["risk_levels"]["low_below"]], ["High", "Medium"], "Low")
    return out


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    score(pd.read_csv(src), sys.argv[3] if len(sys.argv) > 3 else "models").to_csv(dst, index=False)
    print("scored ->", dst)
