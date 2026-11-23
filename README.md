# Hotel Booking Cancellation Predictor

A Streamlit app that predicts whether a hotel reservation will be canceled. The random forest uses booking details such as lead time, room price, and previous cancellations.

## Run locally

From the repository root, install the dependencies and start Streamlit. The saved model was exported with scikit-learn 1.6.1.

```sh
python -m pip install -r requirements.txt "scikit-learn==1.6.1"
python -m streamlit run hotel_booking_app.py
```

Open [localhost:8501](http://localhost:8501) and try a sample booking or fill in the form.

## Model

The [training script](hotel_booking_model_oop.py) fits the random forest on the [hotel bookings dataset](Dataset_B_hotel.csv). The [notebook](hotel_booking_model_training.ipynb) also compares random forest and XGBoost.

The Streamlit app currently reverses the saved class labels. To see a prediction decoded with the saved label encoder, run `python hotel_booking_inference.py`.
