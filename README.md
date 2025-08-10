# Stock Price Forecasting ML Pipeline

A comprehensive machine learning pipeline for stock price forecasting using Prefect, MLflow, and Streamlit. This project includes data ingestion, preprocessing, model training, evaluation, and a web-based prediction interface.

## 🚀 Features

- **ML Pipeline**: Automated training pipeline with Prefect orchestration
- **Data Pipeline**: ETL pipeline for stock data collection and storage
- **Web Interface**: Streamlit app for real-time predictions and analysis
- **Model Management**: MLflow integration for experiment tracking and model versioning
- **Multiple Data Sources**: Support for yfinance API and database ingestion
- **Technical Indicators**: Advanced feature engineering with technical analysis

## 📁 Project Structure

```
stock-price-forecasting/
├── app/                          # Streamlit web application
│   ├── main.py                  # Main Streamlit app
│   ├── modules/                 # App-specific modules
│   └── artifacts/               # Model artifacts
├── data_pipeline/               # ETL pipeline for data collection
│   ├── main-data-pipeline.py    # Main data pipeline orchestration
│   └── modules/                 # ETL modules (extract, transform, load)
├── ml_pipeline/                 # ML training pipeline
│   ├── main_ml_pipeline.py      # Main ML pipeline orchestration
│   └── modules/                 # ML pipeline modules
├── notebooks/                    # Jupyter notebooks for exploration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip or conda
- Git

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd stock-price-forecasting
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```bash
DB_URL=postgresql://user:pass@host:port/db 
DAGSHUB_TOKEN=YOUR_TOKEN
```

## 🚀 Quick Start

### 1. Data Pipeline

Run the data collection pipeline to gather stock data:

```bash
python data_pipeline/main-data-pipeline.py
```

This will:
- Extract stock data from yfinance or your database
- Transform and clean the data
- Load it into your configured database

### 2. ML Training Pipeline

Train the forecasting model:

```bash
python ml_pipeline/main_ml_pipeline.py
```

This will:
- Ingest and preprocess the data
- Engineer technical features
- Train the model using scikit-learn
- Log experiments to MLflow
- Save the trained model

### 3. Web Application

Launch the Streamlit web interface:

```bash
streamlit run app/main.py
```

The web app provides:
- Real-time stock price predictions
- Historical data visualization
- Interactive charts and analysis

## 📊 Model Details

### Features

The model uses engineered features including:
- **Price Features**: Open, High, Low, Close, Volume
- **Lag Features**: Previous day prices and volumes

### Algorithms

- **Default**: scikit-learn GradientBoostingRegressor
- **Evaluation**: Mean Absolute Error, Root Mean Square Error, R² Score

## 🔧 Development

### Project Components

#### ML Pipeline (`ml_pipeline/`)
- **Data Ingestion**: `modules/data_ingestion.py`
- **Preprocessing**: `modules/data_preprocessing.py`
- **Training**: `modules/training.py`
- **Evaluation**: `modules/evaluation.py`
- **Model Pusher**: `modules/pusher.py`

#### Data Pipeline (`data_pipeline/`)
- **Extract**: `modules/extract.py` - Data extraction from APIs/databases
- **Transform**: `modules/transform.py` - Data cleaning and transformation
- **Load**: `modules/load.py` - Data loading into target database

#### Web Application (`app/`)
- **Main App**: `main.py` - Streamlit interface
- **Helper Modules**: `modules/` - Utility functions and data processing

### Adding New Features

1. **New Technical Indicators**: Add to `ml_pipeline/modules/data_preprocessing.py`
2. **New Models**: Extend `ml_pipeline/modules/training.py`
3. **New Data Sources**: Modify `data_pipeline/modules/extract.py`

## 🆘 Troubleshooting

### Common Issues

1. **MLflow Connection Error**: Ensure the tracking URI is accessible
2. **Model Loading Error**: Check if the model is registered in MLflow
3. **Data Source Error**: Verify API keys and database connections
4. **Dependency Issues**: Ensure all requirements are installed in the virtual environment

### Getting Help

- Check the logs in the respective pipeline directories
- Verify environment variable configuration
- Ensure database connectivity (if using database source)
- Check MLflow model registry for model availability

## 🔮 Future Enhancements

- [ ] Real-time streaming data integration
- [ ] Advanced ensemble methods
- [ ] Automated hyperparameter tuning
- [ ] API endpoints for model serving
- [ ] Docker containerization
