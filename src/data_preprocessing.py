"""
Data Preprocessing Module for Fraud Detection
Handles missing values, data cleaning, and dataset merging for geolocation analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Comprehensive data preprocessing class for fraud detection datasets.
    Handles missing values, data cleaning, and geolocation analysis.
    """
    
    def __init__(self):
        self.fraud_data = None
        self.ip_country_data = None
        self.credit_card_data = None
        
    def load_data(self, fraud_path: str, ip_country_path: str, credit_card_path: str) -> None:
        """
        Load all datasets for preprocessing.
        
        Args:
            fraud_path: Path to Fraud_Data.csv
            ip_country_path: Path to IpAddress_to_Country.csv
            credit_card_path: Path to creditcard.csv
        """
        logger.info("Loading datasets...")
        
        try:
            self.fraud_data = pd.read_csv(fraud_path)
            logger.info(f"Loaded fraud data: {self.fraud_data.shape}")
            
            self.ip_country_data = pd.read_csv(ip_country_path)
            logger.info(f"Loaded IP country data: {self.ip_country_data.shape}")
            
            self.credit_card_data = pd.read_csv(credit_card_path)
            logger.info(f"Loaded credit card data: {self.credit_card_data.shape}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input dataframe
            strategy: 'drop' to remove rows with missing values, 'impute' to fill them
            
        Returns:
            DataFrame with handled missing values
        """
        logger.info(f"Handling missing values using strategy: {strategy}")
        
        if strategy == 'drop':
            df_cleaned = df.dropna()
            logger.info(f"Dropped {len(df) - len(df_cleaned)} rows with missing values")
        elif strategy == 'impute':
            # Impute numeric columns with median, categorical with mode
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            df_cleaned = df.copy()
            
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    df_cleaned[col].fillna(df[col].median(), inplace=True)
                    
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    df_cleaned[col].fillna(df[col].mode()[0], inplace=True)
                    
            logger.info("Imputed missing values")
        else:
            raise ValueError("Strategy must be 'drop' or 'impute'")
            
        return df_cleaned
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with duplicates removed
        """
        initial_count = len(df)
        df_cleaned = df.drop_duplicates()
        removed_count = initial_count - len(df_cleaned)
        
        logger.info(f"Removed {removed_count} duplicate rows")
        return df_cleaned
    
    def correct_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Correct data types for optimal processing.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with corrected data types
        """
        df_cleaned = df.copy()
        
        # Convert timestamp columns to datetime
        timestamp_cols = ['signup_time', 'purchase_time']
        for col in timestamp_cols:
            if col in df_cleaned.columns:
                df_cleaned[col] = pd.to_datetime(df_cleaned[col])
        
        # Convert categorical columns
        categorical_cols = ['source', 'browser', 'sex', 'device_id']
        for col in categorical_cols:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].astype('category')
        
        # Convert target variable to int
        if 'class' in df_cleaned.columns:
            df_cleaned['class'] = df_cleaned['class'].astype(int)
            
        logger.info("Corrected data types")
        return df_cleaned
    
    def convert_ip_to_integer(self, ip_address: str) -> int:
        """
        Convert IP address string to integer for comparison.
        
        Args:
            ip_address: IP address string (e.g., "192.168.1.1")
            
        Returns:
            Integer representation of IP address
        """
        try:
            parts = ip_address.split('.')
            return sum(int(part) << (24 - 8 * i) for i, part in enumerate(parts))
        except:
            return 0
    
    def merge_with_geolocation(self) -> pd.DataFrame:
        """
        Merge fraud data with IP country mapping for geolocation analysis.
        
        Returns:
            Merged DataFrame with country information
        """
        if self.fraud_data is None or self.ip_country_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Merging fraud data with geolocation information...")
        
        # Convert IP addresses to integers for comparison
        fraud_data = self.fraud_data.copy()
        fraud_data['ip_integer'] = fraud_data['ip_address'].apply(self.convert_ip_to_integer)
        
        ip_country_data = self.ip_country_data.copy()
        ip_country_data['lower_bound_int'] = ip_country_data['lower_bound_ip_address'].apply(self.convert_ip_to_integer)
        ip_country_data['upper_bound_int'] = ip_country_data['upper_bound_ip_address'].apply(self.convert_ip_to_integer)
        
        # Merge using IP range matching
        merged_data = []
        
        for _, fraud_row in fraud_data.iterrows():
            ip_int = fraud_row['ip_integer']
            
            # Find matching country
            matching_country = ip_country_data[
                (ip_country_data['lower_bound_int'] <= ip_int) & 
                (ip_country_data['upper_bound_int'] >= ip_int)
            ]
            
            if not matching_country.empty:
                country = matching_country.iloc[0]['country']
            else:
                country = 'Unknown'
            
            fraud_row_dict = fraud_row.to_dict()
            fraud_row_dict['country'] = country
            merged_data.append(fraud_row_dict)
        
        merged_df = pd.DataFrame(merged_data)
        logger.info(f"Merged data shape: {merged_df.shape}")
        logger.info(f"Countries found: {merged_df['country'].nunique()}")
        
        return merged_df
    
    def preprocess_fraud_data(self, strategy: str = 'impute') -> pd.DataFrame:
     
        logger.info("Starting fraud data preprocessing...")
        
        if self.fraud_data is None:
            raise ValueError("Fraud data not loaded. Call load_data() first.")
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(self.fraud_data, strategy)
        
        # Step 2: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 3: Correct data types
        df = self.correct_data_types(df)
        
        # Step 4: Merge with geolocation data
        df = self.merge_with_geolocation()
        
        logger.info("Fraud data preprocessing completed")
        return df
    
    def preprocess_credit_card_data(self, strategy: str = 'impute') -> pd.DataFrame:
      
        logger.info("Starting credit card data preprocessing...")
        
        if self.credit_card_data is None:
            raise ValueError("Credit card data not loaded. Call load_data() first.")
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(self.credit_card_data, strategy)
        
        # Step 2: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 3: Correct data types
        df = self.correct_data_types(df)
        
        logger.info("Credit card data preprocessing completed")
        return df
    
    def get_class_distribution(self, df: pd.DataFrame, target_col: str = 'class') -> dict:
       
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        class_counts = df[target_col].value_counts()
        total_samples = len(df)
        
        distribution = {
            'class_counts': class_counts.to_dict(),
            'total_samples': total_samples,
            'imbalance_ratio': class_counts.min() / class_counts.max(),
            'fraud_percentage': (class_counts.get(1, 0) / total_samples) * 100
        }
        
        logger.info(f"Class distribution: {distribution}")
        return distribution

 