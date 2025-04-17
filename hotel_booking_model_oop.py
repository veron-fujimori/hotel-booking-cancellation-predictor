import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

class HotelBookingModel:
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.target_encoder = LabelEncoder()
        self.label_encoders = {}
        self.cat_cols = []
        self.num_cols = []
        self.feature_columns = []

    def identify_feature_types(self, X: pd.DataFrame):
        self.cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        self.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    def preprocess_data(self,
                        X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_train: pd.Series,
                        y_test: pd.Series):
        y_train_enc = self.target_encoder.fit_transform(y_train)
        y_test_enc  = self.target_encoder.transform(y_test)

        self.identify_feature_types(X_train)

        for col in self.cat_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            X_test[col]  = X_test[col].map(lambda v: v if v in le.classes_ else le.classes_[0])
            X_test[col]  = le.transform(X_test[col])
            self.label_encoders[col] = le

        X_train[self.num_cols] = self.scaler.fit_transform(X_train[self.num_cols])
        X_test[self.num_cols]  = self.scaler.transform(X_test[self.num_cols])

        self.feature_columns = X_train.columns.tolist()

        return X_train, X_test, y_train_enc, y_test_enc

    def train(self, X_train: pd.DataFrame, y_train: np.ndarray):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray):
        y_pred = self.model.predict(X_test)
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

    def save_model(self, folder="models"):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "hotel_booking_model.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        with open(os.path.join(folder, "label_encoders.pkl"), "wb") as f:
            pickle.dump(self.label_encoders, f)
        with open(os.path.join(folder, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)
        with open(os.path.join(folder, "target_encoder.pkl"), "wb") as f:
            pickle.dump(self.target_encoder, f)
        with open(os.path.join(folder, "feature_columns.pkl"), "wb") as f:
            pickle.dump(self.feature_columns, f)
        print(f"Saved model + preprocessors to `{folder}/`")

    @classmethod
    def load_model(cls, folder="models"):
        inst = cls()
        with open(os.path.join(folder, "hotel_booking_model.pkl"), "rb") as f:
            inst.model = pickle.load(f)
        with open(os.path.join(folder, "label_encoders.pkl"), "rb") as f:
            inst.label_encoders = pickle.load(f)
        with open(os.path.join(folder, "scaler.pkl"), "rb") as f:
            inst.scaler = pickle.load(f)
        with open(os.path.join(folder, "target_encoder.pkl"), "rb") as f:
            inst.target_encoder = pickle.load(f)
        with open(os.path.join(folder, "feature_columns.pkl"), "rb") as f:
            inst.feature_columns = pickle.load(f)
        inst.cat_cols = list(inst.label_encoders.keys())
        inst.num_cols = [c for c in inst.feature_columns if c not in inst.cat_cols]
        print(f"Loaded model + preprocessors from `{folder}/`")
        return inst

    def preprocess_input(self, input_data: dict):
        df = pd.DataFrame([input_data])
        if "Booking_ID" in df.columns:
            df = df.drop(columns=["Booking_ID"])
        df = df.reindex(columns=self.feature_columns, fill_value=0)
        for col, le in self.label_encoders.items():
            df[col] = df[col].map(lambda v: v if v in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        return df

    def predict(self, input_data: dict):
        df = self.preprocess_input(input_data)
        num_pred = self.model.predict(df)
        return self.target_encoder.inverse_transform(num_pred)[0]


if __name__ == "__main__":
    df = pd.read_csv("Dataset_B_hotel.csv")
    df = df.drop(columns=["Booking_ID"])

    X = df.drop(columns=["booking_status"])
    y = df["booking_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    m = HotelBookingModel()
    X_tr, X_te, y_tr, y_te = m.preprocess_data(X_train, X_test, y_train, y_test)
    m.train(X_tr, y_tr)
    m.evaluate(X_te, y_te)
    m.save_model("models")

    loaded = HotelBookingModel.load_model("models")
    sample = {
        "no_of_adults": 2,
        "no_of_children": 0,
        "no_of_weekend_nights": 1,
        "no_of_week_nights": 2,
        "type_of_meal_plan": "Meal Plan 1",
        "required_car_parking_space": 0,
        "room_type_reserved": "Room_Type 1",
        "lead_time": 224,
        "arrival_year": 2018,
        "arrival_month": 10,
        "arrival_date": 2,
        "market_segment_type": "Offline",
        "repeated_guest": 0,
        "no_of_previous_cancellations": 0,
        "no_of_previous_bookings_not_canceled": 0,
        "avg_price_per_room": 65.0,
        "no_of_special_requests": 0
    }
    print("Sample prediction:", loaded.predict(sample))
