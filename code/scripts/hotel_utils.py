"""Shared helpers for the hotel-cancellation project.

Keeping these classes in a real module (instead of defining them inside the notebook)
means `joblib.load("preprocessor.joblib")` works in ANY script / app / dashboard backend.
A class that was defined in a notebook is pickled as `__main__.FrequencyEncoder`, which
cannot be un-pickled anywhere else -- that is why the previous preprocessor.joblib could
not be loaded outside the notebook.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category by its relative frequency learned on the fit data.
    Never looks at the target, so it cannot leak the label. Unseen -> 0.0."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = [str(c) for c in X.columns]
        self.freq_maps_ = [X[c].value_counts(normalize=True).to_dict() for c in X.columns]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for i, c in enumerate(X.columns):
            X[c] = X[c].map(self.freq_maps_[i]).fillna(0.0)
        return X.values.astype("float64")

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else self.columns_, dtype=object)


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
MONTH_MAP = {m: i + 1 for i, m in enumerate(MONTHS)}

# columns that are removed before modelling (identifiers of the target / post-booking info / redundant copies)
DROP_ALWAYS = ["is_canceled", "arrival_date", "arrival_date_month", "arrival_date_week_number",
               "total_guests", "total_nights", "estimated_revenue", "requires_parking",
               "has_special_requests", "has_previous_cancellation"]
# booking-time-unknown fields (recorded at / after check-in)
LEAKAGE_COLS = ["assigned_room_type", "booking_changes", "has_booking_changes", "room_type_changed",
                "required_car_parking_spaces"]
