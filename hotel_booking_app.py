import streamlit as st
import pandas as pd
import pickle

with open('models/hotel_booking_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('models/label_encoders.pkl', 'rb') as encoders_file:
    label_encoders = pickle.load(encoders_file)

with open('models/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def preprocess_input(input_data):
    df = pd.DataFrame(input_data, index=[0])

    categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
    for col in categorical_cols:
        if col in df.columns:
            le = label_encoders[col]
            if df[col].iloc[0] not in le.classes_:
                df[col] = le.transform([le.classes_[0]])
            else:
                df[col] = le.transform(df[col])

    numeric_cols = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
                    'required_car_parking_space', 'lead_time', 'arrival_year', 'arrival_month',
                    'arrival_date', 'repeated_guest', 'no_of_previous_cancellations',
                    'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests']
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df

st.title("Hotel Booking Prediction")

st.subheader("Test Cases")

col1, col2 = st.columns(2)

with col1:
    if st.button('Run Test Case 1'):
        test_case_1 = {
            "no_of_adults": 2,
            "no_of_children": 1,
            "no_of_weekend_nights": 1,
            "no_of_week_nights": 3,
            "type_of_meal_plan": "Breakfast",
            "required_car_parking_space": 1,
            "room_type_reserved": "A",
            "lead_time": 60,
            "arrival_year": 2023,
            "arrival_month": 6,
            "arrival_date": 10,
            "market_segment_type": "Online",
            "repeated_guest": 1,
            "no_of_previous_cancellations": 0,
            "no_of_previous_bookings_not_canceled": 2,
            "avg_price_per_room": 150,
            "no_of_special_requests": 1
        }

        st.write("Test Case 1 Values:")
        st.write(test_case_1)

        processed_test_case_1 = preprocess_input(test_case_1)
        prediction_test_case_1 = model.predict(processed_test_case_1)
        if prediction_test_case_1[0] == 1:
            st.write("Test Case 1: The booking is likely to be canceled.")
        else:
            st.write("Test Case 1: The booking is likely to not be canceled.")

with col2:
    if st.button('Run Test Case 2'):
        test_case_2 = {
            "no_of_adults": 1,
            "no_of_children": 0,
            "no_of_weekend_nights": 0,
            "no_of_week_nights": 2,
            "type_of_meal_plan": "None",
            "required_car_parking_space": 0,
            "room_type_reserved": "B",
            "lead_time": 30,
            "arrival_year": 2023,
            "arrival_month": 8,
            "arrival_date": 5,
            "market_segment_type": "Corporate",
            "repeated_guest": 0,
            "no_of_previous_cancellations": 1,
            "no_of_previous_bookings_not_canceled": 3,
            "avg_price_per_room": 200,
            "no_of_special_requests": 0
        }

        st.write("Test Case 2 Values:")
        st.write(test_case_2)

        processed_test_case_2 = preprocess_input(test_case_2)
        prediction_test_case_2 = model.predict(processed_test_case_2)
        if prediction_test_case_2[0] == 1:
            st.write("Test Case 2: The booking is likely to be canceled.")
        else:
            st.write("Test Case 2: The booking is likely to not be canceled.")

st.subheader("Input Form for Prediction")
with st.form(key='input_form'):
    no_of_adults = st.number_input("Number of adults", min_value=1, max_value=10, value=2)
    no_of_children = st.number_input("Number of children", min_value=0, max_value=10, value=0)
    no_of_weekend_nights = st.number_input("Number of weekend nights", min_value=0, max_value=7, value=1)
    no_of_week_nights = st.number_input("Number of week nights", min_value=0, max_value=7, value=3)
    type_of_meal_plan = st.selectbox("Type of meal plan", ["None", "Breakfast", "Full Board", "All Inclusive"])
    required_car_parking_space = st.selectbox("Required car parking space", [0, 1])
    room_type_reserved = st.selectbox("Room type reserved", ["A", "B", "C", "D"])
    lead_time = st.number_input("Lead time", min_value=1, max_value=365, value=30)
    arrival_year = st.number_input("Arrival year", min_value=2022, max_value=2025, value=2023)
    arrival_month = st.number_input("Arrival month", min_value=1, max_value=12, value=5)
    arrival_date = st.number_input("Arrival date", min_value=1, max_value=31, value=15)
    market_segment_type = st.selectbox("Market segment type", ["Online", "Offline", "Direct", "Corporate"])
    repeated_guest = st.selectbox("Repeated guest", [0, 1])
    no_of_previous_cancellations = st.number_input("No. of previous cancellations", min_value=0, value=0)
    no_of_previous_bookings_not_canceled = st.number_input("No. of previous bookings not canceled", min_value=0, value=1)
    avg_price_per_room = st.number_input("Average price per room", min_value=0, value=100)
    no_of_special_requests = st.number_input("No. of special requests", min_value=0, value=0)

    submit_button = st.form_submit_button("Predict")

if submit_button:
    input_data = {
        "no_of_adults": no_of_adults,
        "no_of_children": no_of_children,
        "no_of_weekend_nights": no_of_weekend_nights,
        "no_of_week_nights": no_of_week_nights,
        "type_of_meal_plan": type_of_meal_plan,
        "required_car_parking_space": required_car_parking_space,
        "room_type_reserved": room_type_reserved,
        "lead_time": lead_time,
        "arrival_year": arrival_year,
        "arrival_month": arrival_month,
        "arrival_date": arrival_date,
        "market_segment_type": market_segment_type,
        "repeated_guest": repeated_guest,
        "no_of_previous_cancellations": no_of_previous_cancellations,
        "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
        "avg_price_per_room": avg_price_per_room,
        "no_of_special_requests": no_of_special_requests
    }
    
    processed_data = preprocess_input(input_data)
    prediction = model.predict(processed_data)
    
    if prediction[0] == 1:
        st.write("The booking is likely to be canceled.")
    else:
        st.write("The booking is likely to not be canceled.")
