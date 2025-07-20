
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        self.features_data = None
        
    def load_data(self) -> None:

        try:
            logger.info(f"Loading data from {self.data_path}...")
            self.data = pd.read_csv(self.data_path)
            
            # Convert timestamp columns to datetime
            self.data['signup_time'] = pd.to_datetime(self.data['signup_time'])
            self.data['purchase_time'] = pd.to_datetime(self.data['purchase_time'])
            
            logger.info(f"Data loaded successfully: {self.data.shape}")
            logger.info(f"Date range: {self.data['purchase_time'].min()} to {self.data['purchase_time'].max()}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_time_based_features(self) -> pd.DataFrame:
     
        logger.info("Creating time-based features...")
        
        df = self.data.copy()
        
        # Extract hour of day (0-23)
        df['hour_of_day'] = df['purchase_time'].dt.hour
        
        # Extract day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        
        # Extract day of week name
        df['day_of_week_name'] = df['purchase_time'].dt.day_name()
        
        # Extract month
        df['month'] = df['purchase_time'].dt.month
        
        # Extract year
        df['year'] = df['purchase_time'].dt.year
        
        # Extract day of month
        df['day_of_month'] = df['purchase_time'].dt.day
        
        # Extract week of year
        df['week_of_year'] = df['purchase_time'].dt.isocalendar().week
        
        # Create time periods
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 17)).astype(int)
        df['is_night_hours'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)
        
        # Create time of day categories
        df['time_of_day'] = pd.cut(df['hour_of_day'], 
                                  bins=[0, 6, 12, 18, 24], 
                                  labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                  include_lowest=True)
        
        logger.info("Time-based features created successfully")
        return df
    
    def calculate_time_since_signup(self, df: pd.DataFrame) -> pd.DataFrame:
      
        logger.info("Calculating time since signup features...")
        
        df_with_time = df.copy()
        
        # Calculate time difference in various units
        df_with_time['time_since_signup_seconds'] = (df_with_time['purchase_time'] - df_with_time['signup_time']).dt.total_seconds()
        df_with_time['time_since_signup_minutes'] = df_with_time['time_since_signup_seconds'] / 60
        df_with_time['time_since_signup_hours'] = df_with_time['time_since_signup_minutes'] / 60
        df_with_time['time_since_signup_days'] = df_with_time['time_since_signup_hours'] / 24
        
        # Create time since signup categories
        df_with_time['time_since_signup_category'] = pd.cut(
            df_with_time['time_since_signup_hours'],
            bins=[0, 1, 24, 168, float('inf')],  # 1 hour, 1 day, 1 week, more
            labels=['Immediate', 'Same Day', 'Same Week', 'Long Term'],
            include_lowest=True
        )
        
        # Flag immediate transactions (within 1 hour of signup)
        df_with_time['is_immediate_transaction'] = (df_with_time['time_since_signup_hours'] <= 1).astype(int)
        
        # Flag same-day transactions
        df_with_time['is_same_day_transaction'] = (df_with_time['time_since_signup_days'] <= 1).astype(int)
        
        logger.info("Time since signup features calculated successfully")
        return df_with_time
    
    def create_transaction_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:

        logger.info("Creating transaction frequency features...")
        
        df_with_freq = df.copy()
        
        # Sort by user_id and purchase_time
        df_with_freq = df_with_freq.sort_values(['user_id', 'purchase_time'])
        
        # Calculate transaction count per user
        user_transaction_counts = df_with_freq.groupby('user_id').size().reset_index(name='total_transactions')
        df_with_freq = df_with_freq.merge(user_transaction_counts, on='user_id', how='left')
        
        # Calculate transaction frequency (transactions per day)
        user_first_last_dates = df_with_freq.groupby('user_id').agg({
            'purchase_time': ['min', 'max']
        }).reset_index()
        user_first_last_dates.columns = ['user_id', 'first_transaction', 'last_transaction']
        
        user_first_last_dates['user_lifetime_days'] = (
            user_first_last_dates['last_transaction'] - user_first_last_dates['first_transaction']
        ).dt.total_seconds() / (24 * 3600)
        
        # Avoid division by zero
        user_first_last_dates['user_lifetime_days'] = user_first_last_dates['user_lifetime_days'].fillna(1)
        user_first_last_dates['user_lifetime_days'] = user_first_last_dates['user_lifetime_days'].clip(lower=1)
        
        df_with_freq = df_with_freq.merge(user_first_last_dates, on='user_id', how='left')
        df_with_freq['transactions_per_day'] = df_with_freq['total_transactions'] / df_with_freq['user_lifetime_days']
        
        # Create frequency categories
        df_with_freq['transaction_frequency_category'] = pd.cut(
            df_with_freq['transactions_per_day'],
            bins=[0, 0.1, 1, 10, float('inf')],
            labels=['Very Low', 'Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        logger.info("Transaction frequency features created successfully")
        return df_with_freq
    
    def create_transaction_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
     
        logger.info("Creating transaction velocity features...")
        
        df_with_velocity = df.copy()
        
        # Sort by user_id and purchase_time
        df_with_velocity = df_with_velocity.sort_values(['user_id', 'purchase_time'])
        
        # Calculate time since previous transaction for each user
        df_with_velocity['time_since_previous_transaction'] = df_with_velocity.groupby('user_id')['purchase_time'].diff()
        df_with_velocity['time_since_previous_transaction_seconds'] = df_with_velocity['time_since_previous_transaction'].dt.total_seconds()
        df_with_velocity['time_since_previous_transaction_minutes'] = df_with_velocity['time_since_previous_transaction_seconds'] / 60
        df_with_velocity['time_since_previous_transaction_hours'] = df_with_velocity['time_since_previous_transaction_minutes'] / 60
        
        # Calculate velocity metrics per user
        user_velocity_stats = df_with_velocity.groupby('user_id').agg({
            'time_since_previous_transaction_minutes': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        user_velocity_stats.columns = [
            'user_id', 'avg_time_between_transactions_min', 'std_time_between_transactions_min',
            'min_time_between_transactions_min', 'max_time_between_transactions_min'
        ]
        
        df_with_velocity = df_with_velocity.merge(user_velocity_stats, on='user_id', how='left')
        
        # Create velocity categories
        df_with_velocity['transaction_velocity_category'] = pd.cut(
            df_with_velocity['avg_time_between_transactions_min'],
            bins=[0, 1, 60, 1440, float('inf')],  # 1 min, 1 hour, 1 day, more
            labels=['Very Fast', 'Fast', 'Normal', 'Slow'],
            include_lowest=True
        )
        
        # Flag rapid transactions (less than 1 minute between transactions)
        df_with_velocity['is_rapid_transaction'] = (
            (df_with_velocity['time_since_previous_transaction_minutes'] <= 1) & 
            (df_with_velocity['time_since_previous_transaction_minutes'].notna())
        ).astype(int)
        
        # Flag burst transactions (multiple transactions in short time)
        df_with_velocity['is_burst_transaction'] = (
            (df_with_velocity['time_since_previous_transaction_minutes'] <= 5) & 
            (df_with_velocity['time_since_previous_transaction_minutes'].notna())
        ).astype(int)
        
        logger.info("Transaction velocity features created successfully")
        return df_with_velocity
    
    def create_user_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
      
        logger.info("Creating user behavior features...")
        
        df_with_behavior = df.copy()
        
        # User activity patterns
        df_with_behavior['user_activity_hours'] = df_with_behavior.groupby('user_id')['hour_of_day'].transform('nunique')
        df_with_behavior['user_activity_days'] = df_with_behavior.groupby('user_id')['day_of_week'].transform('nunique')
        
        # User device diversity
        df_with_behavior['user_device_count'] = df_with_behavior.groupby('user_id')['device_id'].transform('nunique')
        df_with_behavior['user_browser_count'] = df_with_behavior.groupby('user_id')['browser'].transform('nunique')
        df_with_behavior['user_source_count'] = df_with_behavior.groupby('user_id')['source'].transform('nunique')
        
        # User location diversity
        df_with_behavior['user_country_count'] = df_with_behavior.groupby('user_id')['country'].transform('nunique')
        df_with_behavior['user_ip_count'] = df_with_behavior.groupby('user_id')['ip_address'].transform('nunique')
        
        # Purchase value statistics per user
        user_value_stats = df_with_behavior.groupby('user_id').agg({
            'purchase_value': ['mean', 'std', 'min', 'max', 'sum']
        }).reset_index()
        
        user_value_stats.columns = [
            'user_id', 'avg_purchase_value', 'std_purchase_value',
            'min_purchase_value', 'max_purchase_value', 'total_purchase_value'
        ]
        
        df_with_behavior = df_with_behavior.merge(user_value_stats, on='user_id', how='left')
        
        # Create user risk indicators
        df_with_behavior['is_high_value_user'] = (df_with_behavior['total_purchase_value'] > df_with_behavior['total_purchase_value'].quantile(0.95)).astype(int)
        df_with_behavior['is_multi_device_user'] = (df_with_behavior['user_device_count'] > 1).astype(int)
        df_with_behavior['is_multi_country_user'] = (df_with_behavior['user_country_count'] > 1).astype(int)
        df_with_behavior['is_multi_ip_user'] = (df_with_behavior['user_ip_count'] > 1).astype(int)
        
        logger.info("User behavior features created successfully")
        return df_with_behavior
    
    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
      
        logger.info("Creating seasonal features...")
        
        df_with_seasonal = df.copy()
        
        # Cyclical encoding for hour of day (sin and cos)
        df_with_seasonal['hour_sin'] = np.sin(2 * np.pi * df_with_seasonal['hour_of_day'] / 24)
        df_with_seasonal['hour_cos'] = np.cos(2 * np.pi * df_with_seasonal['hour_of_day'] / 24)
        
        # Cyclical encoding for day of week
        df_with_seasonal['day_of_week_sin'] = np.sin(2 * np.pi * df_with_seasonal['day_of_week'] / 7)
        df_with_seasonal['day_of_week_cos'] = np.cos(2 * np.pi * df_with_seasonal['day_of_week'] / 7)
        
        # Cyclical encoding for month
        df_with_seasonal['month_sin'] = np.sin(2 * np.pi * df_with_seasonal['month'] / 12)
        df_with_seasonal['month_cos'] = np.cos(2 * np.pi * df_with_seasonal['month'] / 12)
        
        # Day of month cyclical encoding
        df_with_seasonal['day_of_month_sin'] = np.sin(2 * np.pi * df_with_seasonal['day_of_month'] / 31)
        df_with_seasonal['day_of_month_cos'] = np.cos(2 * np.pi * df_with_seasonal['day_of_month'] / 31)
        
        # Week of year cyclical encoding
        df_with_seasonal['week_of_year_sin'] = np.sin(2 * np.pi * df_with_seasonal['week_of_year'] / 53)
        df_with_seasonal['week_of_year_cos'] = np.cos(2 * np.pi * df_with_seasonal['week_of_year'] / 53)
        
        logger.info("Seasonal features created successfully")
        return df_with_seasonal
    
    def engineer_all_features(self) -> pd.DataFrame:
     
        logger.info("Starting comprehensive feature engineering...")
        
        if self.data is None:
            self.load_data()
        
        # Step 1: Create time-based features
        df = self.create_time_based_features()
        
        # Step 2: Calculate time since signup
        df = self.calculate_time_since_signup(df)
        
        # Step 3: Create transaction frequency features
        df = self.create_transaction_frequency_features(df)
        
        # Step 4: Create transaction velocity features
        df = self.create_transaction_velocity_features(df)
        
        # Step 5: Create user behavior features
        df = self.create_user_behavior_features(df)
        
        # Step 6: Create seasonal features
        df = self.create_seasonal_features(df)
        
        # Store the features data
        self.features_data = df
        
        logger.info(f"Feature engineering completed. Final shape: {df.shape}")
        logger.info(f"Total features created: {len(df.columns)}")
        
        return df
    
    def get_feature_summary(self) -> Dict:
        """
        Get a summary of all created features.
        
        Returns:
            Dictionary with feature summary information
        """
        if self.features_data is None:
            raise ValueError("No features data available. Run engineer_all_features() first.")
        
        df = self.features_data
        
        # Categorize features
        original_features = ['user_id', 'signup_time', 'purchase_time', 'purchase_value', 
                           'device_id', 'source', 'browser', 'sex', 'age', 'ip_address', 
                           'class', 'ip_integer', 'country']
        
        time_features = [col for col in df.columns if any(x in col for x in ['hour', 'day', 'month', 'year', 'week'])]
        frequency_features = [col for col in df.columns if 'frequency' in col or 'transactions_per_day' in col]
        velocity_features = [col for col in df.columns if 'velocity' in col or 'time_since_previous' in col]
        signup_features = [col for col in df.columns if 'time_since_signup' in col]
        behavior_features = [col for col in df.columns if 'user_' in col and col not in original_features]
        seasonal_features = [col for col in df.columns if any(x in col for x in ['sin', 'cos'])]
        
        summary = {
            'total_features': len(df.columns),
            'original_features': len(original_features),
            'engineered_features': len(df.columns) - len(original_features),
            'feature_categories': {
                'time_based': len(time_features),
                'transaction_frequency': len(frequency_features),
                'transaction_velocity': len(velocity_features),
                'time_since_signup': len(signup_features),
                'user_behavior': len(behavior_features),
                'seasonal': len(seasonal_features)
            },
            'feature_lists': {
                'time_based': time_features,
                'transaction_frequency': frequency_features,
                'transaction_velocity': velocity_features,
                'time_since_signup': signup_features,
                'user_behavior': behavior_features,
                'seasonal': seasonal_features
            }
        }
        
        return summary
    
    def save_features(self, output_path: str) -> None:
       
        if self.features_data is None:
            raise ValueError("No features data available. Run engineer_all_features() first.")
        
        try:
            logger.info(f"Saving engineered features to {output_path}...")
            self.features_data.to_csv(output_path, index=False)
            logger.info(f"Features saved successfully to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            raise
    
    def print_feature_summary(self) -> None:
        """Print a formatted summary of the engineered features."""
        summary = self.get_feature_summary()
        
        print("\n" + "="*60)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*60)
        print(f"Total Features: {summary['total_features']}")
        print(f"Original Features: {summary['original_features']}")
        print(f"Engineered Features: {summary['engineered_features']}")
        
        print("\nFeature Categories:")
        for category, count in summary['feature_categories'].items():
            print(f"  • {category.replace('_', ' ').title()}: {count} features")
        
        print("\nKey Features Created:")
        print("  • hour_of_day, day_of_week, month, year")
        print("  • time_since_signup (seconds, minutes, hours, days)")
        print("  • transaction frequency and velocity metrics")
        print("  • user behavior patterns")
        print("  • seasonal/cyclical encodings")
        print("="*60)


