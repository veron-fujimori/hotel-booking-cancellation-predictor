import pickle
import pandas as pd

def load_artifacts(folder="models"):
    with open(f"{folder}/hotel_booking_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{folder}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{folder}/label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open(f"{folder}/target_encoder.pkl", "rb") as f:
        target_encoder = pickle.load(f)
    with open(f"{folder}/feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    cat_cols = list(label_encoders.keys())
    num_cols = [c for c in feature_columns if c not in cat_cols]

    return model, scaler, label_encoders, target_encoder, feature_columns, cat_cols, num_cols

def preprocess(input_dict, feature_columns, cat_cols, num_cols, scaler, label_encoders):
    df = pd.DataFrame([input_dict])
    if "Booking_ID" in df.columns:
        df = df.drop(columns=["Booking_ID"])
    df = df.reindex(columns=feature_columns, fill_value=0)
    for col in cat_cols:
        le = label_encoders[col]
        df[col] = df[col].map(lambda v: v if v in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])
    df[num_cols] = scaler.transform(df[num_cols])
    return df

if __name__ == "__main__":
    model, scaler, label_encoders, target_encoder, feature_columns, cat_cols, num_cols = load_artifacts()

    sample_input = {
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

    X = preprocess(sample_input, feature_columns, cat_cols, num_cols, scaler, label_encoders)
    num_pred = model.predict(X)[0]
    human_pred = target_encoder.inverse_transform([num_pred])[0]

    print("Predicted booking_status:", human_pred)
