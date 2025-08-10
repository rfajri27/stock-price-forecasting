import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import joblib
import mlflow
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings('ignore')

# Import your custom modules
from modules.helper import *
from modules.get_data import *
from modules.data_preprocessing import *

# Page configuration
st.set_page_config(
    page_title="Stock Price Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

last_14_days_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')

@st.cache_data
def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        scaler = joblib.load("artifacts/scaler.joblib")
        
        model_name = "stock_price_forecasting"
        model_stage = "production"
        
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_stage}"
        )
        
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_data
def get_historical_data():
    """Get historical stock data from database"""
    try:
        # Get stock data with volume from database
        df = get_stock_data_with_volume(from_date=last_14_days_date)
        
        if df.empty:
            st.warning("No historical data found in database")
            return pd.DataFrame()
        
        # Convert Date column to datetime if it's not already
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date
        df = df.sort_values('Date')
        
        return df
    except Exception as e:
        st.error(f"Error getting historical data: {e}")
        return pd.DataFrame()

@st.cache_data
def get_predicted_data():
    """Get predicted stock data from database"""
    try:
        # Get predicted data from database
        df = get_predicted_stock_data(from_date=last_14_days_date)
        
        if df.empty:
            st.warning("No predicted data found in database")
            return pd.DataFrame()
        
        # Convert Date column to datetime if it's not already
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date
        df = df.sort_values('Date')
        
        return df
    except Exception as e:
        st.error(f"Error getting predicted data: {e}")
        return pd.DataFrame()

def create_prediction_chart(actual_df, predicted_df):
    """Create interactive chart showing actual vs predicted values"""
    fig = go.Figure()
    
    # Actual values
    if not actual_df.empty:
        fig.add_trace(go.Scatter(
            x=actual_df['Date'],
            y=actual_df['Close'],
            mode='lines+markers',
            name='Actual Close Price',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
    
    # Predicted values
    if not predicted_df.empty:
        fig.add_trace(go.Scatter(
            x=predicted_df['Date'],
            y=predicted_df['Predicted_Close'],
            mode='lines+markers',
            name='Predicted Close Price',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title='Stock Price: Actual vs Predicted',
        xaxis_title='Date',
        yaxis_title='Price (Rp)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def predict_next_day(model, scaler, historical_data):
    """Predict the next day's stock price"""
    try:
        historical_data = get_historical_data()
        
        if historical_data.empty:
            return None, None
            
        # Get the last row for prediction
        last_row = historical_data.iloc[-1:]
        
        # Scale the features
        historical_data["Close"] = scaler.transform(historical_data["Close"].values.reshape(-1, 1))

        scaled_features, output_y = data_preprocessing_regression(historical_data)
        scaled_features = scaled_features.tail(5)
        # Make prediction
        prediction = model.predict(scaled_features[input_columns])
        
        # Inverse transform to get actual price
        predicted_price = scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]
        
        return predicted_price, last_row['Close'].iloc[0]
    
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Price Forecasting</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Date range selector
    st.sidebar.subheader("Date Range")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(start_date, end_date),
        max_value=end_date
    )
    
    # Load model and scaler
    with st.spinner("Loading ML model..."):
        model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        st.error("Failed to load the ML model. Please check your model configuration.")
        return
    
    # Get historical data from database
    with st.spinner("Fetching historical data from database..."):
        historical_data = get_historical_data()
    
    if historical_data.empty:
        st.error("Failed to load historical data from database.")
        return
    
    # Get predicted data from database
    with st.spinner("Fetching predicted data from database..."):
        predicted_data = get_predicted_data()
    
    # Filter data based on selected date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        historical_data = historical_data[
            (historical_data['Date'] >= pd.Timestamp(start_date)) &
            (historical_data['Date'] <= pd.Timestamp(end_date))
        ]
        
        if not predicted_data.empty:
            predicted_data = predicted_data[
                (predicted_data['Date'] >= pd.Timestamp(start_date)) &
                (predicted_data['Date'] <= pd.Timestamp(end_date))
            ]
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        current_price = historical_data['Close'].iloc[-1] if not historical_data.empty else 0
        prev_price = historical_data['Close'].iloc[-2] if len(historical_data) > 1 else current_price
        price_change_pct = ((current_price / prev_price - 1) * 100) if prev_price > 0 else 0
        
        st.metric(
            "Current Price",
            f"Rp{current_price:,.0f}",
            f"{price_change_pct:.2f}%"
        )
    
    with col2:
        volume = historical_data['Volume'].iloc[-1] if not historical_data.empty and 'Volume' in historical_data.columns else 0
        st.metric(
            "Volume",
            f"{volume:,}"
        )
    
    # Prediction section
    st.markdown("---")
    st.subheader("🔮 Next Day Prediction")
    
    if st.button("Predict Next Day Price", type="primary"):
        with st.spinner("Making prediction..."):
            predicted_price, current_price = predict_next_day(model, scaler, historical_data)
            
            if predicted_price is not None:
                price_change = predicted_price - current_price
                price_change_pct = (price_change / current_price) * 100
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Current Price", f"Rp{current_price:,.0f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Predicted Price", f"Rp{predicted_price:,.0f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Expected Change", f"Rp{price_change:,.0f}", f"{price_change_pct:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts section
    st.markdown("---")
    st.subheader("📊 Price Analysis")
    
    fig1 = create_prediction_chart(historical_data, predicted_data)
    st.plotly_chart(fig1, use_container_width=True)
    
    if predicted_data.empty:
        st.info("No predicted data available. The chart shows only historical data.")

if __name__ == "__main__":
    # Run the async main function
    main()