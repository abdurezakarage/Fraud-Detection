# Fraud Detection System

A comprehensive machine learning system for detecting fraudulent transactions using advanced data analysis, feature engineering, and machine learning techniques.

## �� Project Overview

This project implements a complete fraud detection pipeline that processes transaction data, performs exploratory data analysis, engineers relevant features, and builds machine learning models to identify fraudulent activities. The system is designed to handle imbalanced datasets and provides comprehensive insights into fraud patterns with model explainability capabilities.

## ✨ Key Features

### 🔍 **Exploratory Data Analysis (EDA)**
- **Class Distribution Analysis**: Comprehensive analysis of fraud vs legitimate transaction distribution
- **Temporal Pattern Analysis**: Fraud rate analysis by hour and day of the week
- **Geographic Analysis**: Fraud patterns by country and location
- **Amount Distribution Analysis**: Statistical analysis of transaction amounts
- **Feature Correlation Analysis**: Correlation heatmaps and feature importance
- **Interactive Visualizations**: Matplotlib and Seaborn-based plots

### 🛠️ **Data Preprocessing**
- **Missing Value Handling**: Multiple strategies (drop/impute) for handling missing data
- **Duplicate Removal**: Automatic detection and removal of duplicate transactions
- **Data Type Correction**: Proper conversion of timestamps, categories, and numerical data
- **Geolocation Integration**: IP address to country mapping for geographic analysis
- **Data Validation**: Comprehensive data quality checks

### ⚙️ **Feature Engineering**
- **Time-based Features**: Hour, day, month, year, weekend/business hours flags
- **Transaction Velocity**: Time between transactions, rapid transaction detection
- **User Behavior Features**: Transaction frequency, user lifetime analysis
- **Seasonal Features**: Day of week, month patterns, holiday detection
- **Risk Indicators**: Immediate transaction flags, burst transaction detection

### 🔄 **Data Transformation**
- **Class Imbalance Handling**: SMOTE, ADASYN, and other resampling techniques
- **Feature Encoding**: Label encoding, one-hot encoding for categorical variables
- **Feature Scaling**: StandardScaler for numerical features
- **Pipeline Creation**: Automated preprocessing pipelines
- **Memory Optimization**: Efficient handling of large datasets

### 🤖 **Machine Learning & Model Explainability**
- **Model Training**: Advanced ML algorithms for fraud detection
- **Model Evaluation**: Comprehensive performance metrics and validation
- **SHAP Analysis**: Model interpretability and feature importance
- **Model Persistence**: Saved models for production deployment
- **Explainable AI**: Transparent model decision-making process

### 📊 **Advanced Analytics**
- **Fraud Pattern Detection**: Identification of peak fraud hours and days
- **Risk Scoring**: Geographic and temporal risk assessment
- **Anomaly Detection**: Statistical outlier detection
- **Performance Metrics**: Comprehensive model evaluation metrics

## 🏗️ Project Structure

```
Fraud-Detection/
├── 📁 data/                    # Data files and processed datasets
├── 📁 models/                  # Trained models and features
│   ├── 📄 best_model.joblib    # Best performing model
│   └── 📄 model_features.joblib # Model feature definitions
├── 📁 src/                     # Source code modules
│   ├── 📄 eda_analysis.py      # Exploratory Data Analysis
│   ├── 📄 data_preprocessing.py # Data cleaning and preprocessing
│   ├── 📄 data_transformation.py # Feature encoding and scaling
│   ├── 📄 feature_enfineering.py # Feature engineering pipeline
│   ├── 📄 data_merger.py       # Data merging utilities
│   ├── 📄 model.py             # Machine learning model training
│   ├── 📄 ModelExplainability.py # SHAP analysis and model interpretation
│   └── 📄 __init__.py
├── 📁 notbooks/               # Jupyter notebooks for analysis
│   ├── 📄 fraud_analysis.ipynb # Main fraud analysis notebook
│   ├── 📄 data_merging.ipynb   # Data merging examples
│   ├── 📄 data_transformation.ipynb # Data transformation examples
│   ├── 📄 feature_engineeering.ipynb # Feature engineering examples
│   ├── 📄 dataProcess.ipynb    # Data processing examples
│   ├── 📄 model.ipynb          # Model training and evaluation
│   └── 📄 model_interprablity.ipynb # Model explainability analysis
├── 📁 venv/                   # Virtual environment
├── 📄 requirements.txt        # Python dependencies
├── 📄 README.md              # This file
└── 📄 .gitignore             # Git ignore file
```


### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Fraud-Detection
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📋 Dependencies

The project requires the following key packages:

- **Data Processing**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`, `imbalanced-learn`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Development**: `jupyter`, `ipykernel`

## 🎯 Key Insights

The system provides comprehensive insights into fraud patterns:

### Temporal Patterns
- **Peak Fraud Hours**: Identifies hours with highest fraud rates
- **Day of Week Analysis**: Shows fraud patterns across weekdays/weekends
- **Seasonal Trends**: Monthly and yearly fraud patterns

### Geographic Patterns
- **High-Risk Countries**: Countries with elevated fraud rates
- **Geographic Clustering**: Regional fraud pattern analysis

### Behavioral Patterns
- **Transaction Velocity**: Rapid transaction detection
- **User Behavior**: User-specific fraud indicators
- **Amount Patterns**: Transaction value analysis

## 🔧 Model Performance

The system includes:
- **Trained Models**: Pre-trained models saved in the `models/` directory
- **Feature Engineering**: Comprehensive feature extraction pipeline
- **Model Explainability**: SHAP-based model interpretation
- **Performance Metrics**: Accuracy, precision, recall, F1-score, and ROC-AUC

