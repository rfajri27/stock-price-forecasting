from modules.helper import *
from modules.get_data import *
from modules.data_preprocessing import *
import joblib

scaler = joblib.load("artifacts/scaler.joblib")

model_name = "stock_price_forecasting"
model_stage = "production"  # or "Staging"

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_stage}"
)

async def predict():
    df = await get_new_stock_data()
    df = await scaling(scaler, df)
    input_x, output_y = await data_preprocessing_regression(df)
    y_pred = model.predict(input_x)
    
    final_result = pd.DataFrame({
        "Date": df["Date"],
        "Close": df["Close"],
        "Baseline_Prediction": df["Close"],
        "Predicted_Close": scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    })
    return y_pred

y_pred = predict()