
# mlflow-implementation

# Apple Stock Price Forecasting

This project demonstrates end-to-end time series forecasting for Apple (AAPL) stock prices using an ARIMA model, MLflow for experiment tracking, and FastAPI for serving the model through an API deployed on AWS EC2.

## Tasks Completed

### 1. Data Collection
- Collected Apple (AAPL) stock price data using `yfinance`
- Retrieved at least two years of historical closing prices

### 2. Model Development
- Implemented ARIMA time series forecasting model
- Performed hyperparameter tuning
- Split data into training (80%) and testing (20%) sets

### 3. MLflow Tracking
- Logged all model hyperparameters
- Tracked performance metrics: RMSE, MAE
- Logged model artifacts
- Registered the best-performing model

### 4. API and Deployment

#### FastAPI Endpoints
- `/predict` endpoint for forecasting
- `/health` endpoint for health check
- Accepts forecast step parameter

#### Cloud Deployment
- Deployed on AWS EC2 t2.micro instance
- Configured security groups for public API access

### 5. Submission Requirements
- GitHub repository with complete code
- This README with setup instructions
- MLflow tracking screenshots included in `mlflow_screenshots/`
- Deployed API endpoint accessible publicly

## Setup Instructions

### Prerequisites
- Python 3.8+
- AWS account and an EC2 instance (t2.micro recommended)
- Security group allowing inbound TCP on port 8000

### Clone Repository
git clone https://github.com/your-username/apple-stock-forecasting.git
cd apple-stock-forecasting




### Install Dependencies
pip install -r requirements.txt

### Run MLflow Tracking Server (Optional)
mlflow ui

Open [http://localhost:5000](http://localhost:5000) to view runs.

### Start FastAPI Server
uvicorn app:app --host 0.0.0.0 --port 8000


## API Usage

- Health check: `http://<EC2_PUBLIC_IP>:8000/health`
- Prediction: `http://<EC2_PUBLIC_IP>:8000/predict?forecast_steps=5`

Replace `<EC2_PUBLIC_IP>` with your actual EC2 public IP.

## MLflow Tracking

All experiments, parameters, metrics, and model artifacts are logged using MLflow. Screenshots are provided in the `mlflow_screenshots/` folder.

## Deployed API Endpoint

Visit your API at: `http://<EC2_PUBLIC_IP>:8000`

## Contact

For any questions or issues, please open an issue in this repository.
